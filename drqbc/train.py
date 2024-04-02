# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
import tqdm
import datetime

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
from dm_env import specs
from collections import deque

import dmc
import utils
import wandb
from numpy_replay_buffer import EfficientReplayBuffer, EfficientLatentReplayBuffer
from video import TrainVideoRecorder, VideoRecorder
from utils import load_offline_dataset_into_buffer, load_generated_dataset_into_buffer
import pyrootutils

from dm_env import StepType
step_type_lookup = {
    0: StepType.FIRST,
    1: StepType.MID,
    2: StepType.LAST
}


path = pyrootutils.find_root(search_from = __file__, indicator=".project_root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)



torch.backends.cudnn.benchmark = True
EPISODE_LENGTH = 500

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger

        wandb.init(
            # config=self.cfg,
            project=f'VD4RL_{self.cfg.task_name}',
            entity='gda-for-orl',
            group=f'{self.cfg.experiment}',
            name=f'DrQ-v2'
        )

        wandb.run.save()
        
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, self.cfg.distracting_mode)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, self.cfg.distracting_mode)
        # create replay buffer
        self.data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_buffer = EfficientReplayBuffer(self.cfg.replay_buffer_size,
                                                   self.cfg.batch_size,
                                                   self.cfg.nstep,
                                                   self.cfg.discount,
                                                   self.cfg.frame_stack,
                                                   self.data_specs)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

        self.eval_on_distracting = self.cfg.eval_on_distracting
        self.eval_on_multitask = self.cfg.eval_on_multitask

    @property
    def global_step(self):
        return self._global_step
    
    @global_step.setter
    def global_step(self, value):
        self._global_step=value

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            # self.video_recorder.save(f'{self.global_frame}.mp4')
        wandb.log({
            'episode_reward': total_reward / episode,
            'episode_length': step * self.cfg.action_repeat / episode,
            'episode': self.global_episode},
            step = self.global_step  
            )

    def eval_distracting(self, record_video):
        distraction_modes = ['easy', 'medium', 'hard', 'fixed_easy', 'fixed_medium', 'fixed_hard']
        if not hasattr(self, 'distracting_envs'):
            self.distracting_envs = []
            for distraction_mode in distraction_modes:
                env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                               self.cfg.action_repeat, self.cfg.seed, distracting_mode=distraction_mode)
                self.distracting_envs.append(env)
        for env, env_name in zip(self.distracting_envs, distraction_modes):
            self.eval_single_env(env, env_name, record_video)

    def eval_multitask(self, record_video):
        multitask_modes = [f'len_{i}' for i in range(1, 11, 1)]
        if not hasattr(self, 'multitask_envs'):
            self.multitask_envs = []
            for multitask_mode in multitask_modes:
                env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                               self.cfg.action_repeat, self.cfg.seed, multitask_mode=multitask_mode)
                self.multitask_envs.append(env)
        for env, env_name in zip(self.multitask_envs, multitask_modes):
            self.eval_single_env(env, env_name, record_video)

    def eval_single_env(self, env, env_name, save_video):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = env.reset()
            self.video_recorder.init(env, enabled=((episode == 0) and save_video))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = env.step(action)
                self.video_recorder.record(env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{env_name}_{self.global_frame}.mp4')
        wandb.log({f'eval/{env_name}_episode_reward', total_reward / episode}, step=self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    wandb.log({
                        'fps', episode_frame / elapsed_time,
                        'total_time', total_time,
                        'episode_reward', episode_reward,
                        'episode_length', episode_frame,
                        'episode', self.global_episode,
                        'buffer_size', len(self.replay_storage)},
                        step = self.global_step            
                    )

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                wandb.log({'eval_total_time': self.timer.total_time()}, step=self.global_step)
                if self.eval_on_distracting:
                    self.eval_distracting(eval_save_vid_every_step(self.global_step))
                if self.eval_on_multitask:
                    self.eval_multitask(eval_save_vid_every_step(self.global_step))
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                wandb.log(metrics, step=self.global_step)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def train_offline(self, offline_dir):
        # Open dataset, load as memory buffer
        load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, self.cfg.frame_stack,
                                         self.cfg.replay_buffer_size)

        if self.replay_buffer.index == -1:
            raise ValueError('No offline data loaded, check directory.')

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, 1) # 1100000
        eval_every_step = utils.Every(self.cfg.eval_every_frames, 1) #   10000
        show_train_stats_every_step = utils.Every(self.cfg.show_train_stats_every_frames, 1)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        metrics = None
        step = 0
        with tqdm.tqdm(total=self.cfg.num_train_frames) as pbar:
            while train_until_step(self.global_step):
                if show_train_stats_every_step(self.global_step):
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        elapsed_time, total_time = self.timer.reset()
                        wandb.log({
                            'fps': step / elapsed_time,
                            'total_time': total_time,
                            'buffer_size': len(self.replay_buffer)},
                            step = self.global_step
                        )
                        step = 0
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                step += 1
                # try to evaluate
                if eval_every_step(self.global_step):
                    wandb.log({'eval_total_time': self.timer.total_time()}, step=self.global_step)
                    if self.eval_on_distracting:
                        self.eval_distracting(eval_save_vid_every_step(self.global_step))
                    if self.eval_on_multitask:
                        self.eval_multitask(eval_save_vid_every_step(self.global_step))
                    self.eval()

                # try to update the agent
                metrics = self.agent.update(self.replay_buffer, self.global_step) #  agent OBS DIM: torch.Size([256, 9, 84, 84])
                if show_train_stats_every_step(self.global_step):
                    wandb.log(metrics, step=self.global_step)

                self._global_step += 1
                pbar.update(1)

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        snapshot = Path(snapshot)
        print("Load snapshot from: ", snapshot)
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            # if k != "replay_buffer":
            self.__dict__[k] = v
            
    def merge_dictionary(self, list_of_Dict):
        merged_data = {}

        for d in list_of_Dict:
            for k, v in d.items():
                if k not in merged_data.keys():
                    merged_data[k] = [v]
                else:
                    merged_data[k].append(v)

        for k, v in merged_data.items():
            merged_data[k] = np.concatenate(merged_data[k])

        return merged_data
    
    def get_timestep_from_idx(self, offline_data: dict, idx: int):
        return dmc.ExtendedTimeStep(
            step_type=step_type_lookup[offline_data['step_type'][idx]],
            reward=offline_data['reward'][idx],
            observation=offline_data['observation'][idx],
            discount=offline_data['discount'][idx],
            action=offline_data['action'][idx]
        )
    
    def add_offline_data_to_buffer(self, offline_data, framestack):
        offline_data_length = offline_data['reward'].shape[0]
        for v in offline_data.values():
            assert v.shape[0] == offline_data_length # 데이터 길이 맞는지 검정
        for idx in range(offline_data_length):
            time_step = self.get_timestep_from_idx(offline_data, idx) # idx 번의 step_type, reward, observation, discount, actions
            if not time_step.first(): # 맨 마지막이거나 맨 처음일 때
                stacked_frames.append(time_step.observation)
                time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
                # print(time_step.observation.shape)
                # self.replay_buffer.add(time_step_stack)
                self.add(time_step_stack)
            else: # 일반 가동중일때
                stacked_frames = deque(maxlen=framestack)
                while len(stacked_frames) < framestack:
                    stacked_frames.append(time_step.observation)
                time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
                # self.replay_buffer.add(time_step_stack)
                self.add(time_step_stack)
                
    def add(self, time_step):
        if self.replay_buffer.index == -1:
            self.replay_buffer.index = 0
            self.replay_buffer.obs_shape = list(time_step.observation.shape)
            self.replay_buffer.ims_channels = self.replay_buffer.obs_shape[0] // self.replay_buffer.frame_stack
            self.replay_buffer.act_shape = time_step.action.shape

            if self.replay_buffer.gta:
                # self.replay_buffer.obs = np.zeros([self.replay_buffer.buffer_size, *self.replay_buffer.obs_shape[1:]], dtype=np.uint8)
                self.replay_buffer.obs = np.zeros([self.replay_buffer.buffer_size, time_step.observation.shape[0]], dtype=np.float32)
            else:
                self.replay_buffer.obs = np.zeros([self.replay_buffer.buffer_size, self.replay_buffer.ims_channels, *self.replay_buffer.obs_shape[1:]], dtype=np.uint8)
            self.replay_buffer.act = np.zeros([self.replay_buffer.buffer_size, *self.replay_buffer.act_shape], dtype=np.float32)
            self.replay_buffer.rew = np.zeros([self.replay_buffer.buffer_size], dtype=np.float32)
            self.replay_buffer.dis = np.zeros([self.replay_buffer.buffer_size], dtype=np.float32)
            self.replay_buffer.valid = np.zeros([self.replay_buffer.buffer_size], dtype=np.bool_)
        # self.replay_buffer.add_data_point(time_step)
        
        # first = time_step.first()
        latest_obs = time_step.observation
        np.copyto(self.replay_buffer.obs[self.replay_buffer.index], latest_obs)  # Check most recent image
        np.copyto(self.replay_buffer.act[self.replay_buffer.index], time_step.action)
        self.replay_buffer.rew[self.replay_buffer.index] = time_step.reward
        self.replay_buffer.dis[self.replay_buffer.index] = time_step.discount
        self.replay_buffer.valid[(self.replay_buffer.index + self.replay_buffer.frame_stack) % self.replay_buffer.buffer_size] = False
        if self.replay_buffer.traj_index >= self.replay_buffer.nstep:
            self.replay_buffer.valid[(self.replay_buffer.index - self.replay_buffer.nstep + 1) % self.replay_buffer.buffer_size] = True
        self.replay_buffer.index += 1
        self.replay_buffer.traj_index += 1
        if self.replay_buffer.index == self.replay_buffer.buffer_size:
            self.replay_buffer.index = 0
            self.replay_buffer.full = True


    def fine_tune_with_GTA(self, offline_dir, gta_dir):
        # Open dataset, load as memory buffer
        self.replay_buffer = EfficientLatentReplayBuffer(
                                                    self.cfg.replay_buffer_size,
                                                    self.cfg.batch_size,
                                                    1,
                                                    self.cfg.discount,
                                                    1, # self.cfg.frame_stack
                                                    self.data_specs)

        # load_generated_dataset_into_buffer(Path(offline_dir), self.replay_buffer, 1,
        #                                  self.cfg.replay_buffer_size)

        load_generated_dataset_into_buffer(Path(gta_dir), self.replay_buffer, 1,
                                         self.cfg.replay_buffer_size)
        
        print(self.replay_buffer.__len__(), " <- Length of the replay buffer")

        self.agent.fine_tune_mode()
        # predicates
        
        # self.global_step=0 # initialize the training step
        train_until_step = utils.Until(self.cfg.num_train_frames, 1)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, 1)
        show_train_stats_every_step = utils.Every(self.cfg.show_train_stats_every_frames, 1)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        metrics = None
        step = 0
        # self.global_step=0
        
        with tqdm.tqdm(total=self.cfg.num_train_frames) as pbar:
            while train_until_step(self.global_step):
                if show_train_stats_every_step(self.global_step):
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        elapsed_time, total_time = self.timer.reset()
                        wandb.log({
                            'fps':step / elapsed_time,
                            'total_time':total_time,
                            'buffer_size': len(self.replay_buffer)},
                            step = self.global_step
                        )
                        step = 0
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                step += 1
                # try to evaluate
                if eval_every_step(self.global_step):
                    wandb.log({'eval_total_time': self.timer.total_time()}, step=self.global_step)
                    if self.eval_on_distracting:
                        self.eval_distracting(eval_save_vid_every_step(self.global_step))
                    if self.eval_on_multitask:
                        self.eval_multitask(eval_save_vid_every_step(self.global_step))
                    self.eval()

                # try to update the agent
                metrics = self.agent.tune_update(self.replay_buffer, self.global_step) #  agent
                if show_train_stats_every_step(self.global_step):
                    wandb.log(metrics, step=self.global_step)

                self._global_step += 1
                pbar.update(1)
                
    def save_latent_trajectory(self, offline_dir):
        print("STARTING TRANSFORMING THE IMAGE INTO LATENT")
        if self.replay_buffer.index == -1:
            load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, self.cfg.frame_stack,
                                         self.cfg.replay_buffer_size)
            
        latent_dataset = []
        for i in tqdm.tqdm(range(self.replay_buffer.index//(EPISODE_LENGTH + 1)), desc='encoding observations'):
            batch = self.replay_buffer.ordered_sampling(np.arange(EPISODE_LENGTH * i, EPISODE_LENGTH * (i+1))) # 501 for the episode length
            obs, action, reward, discount, next_obs = utils.to_torch(
                batch, self.device)
            # augment
            obs = self.agent.aug(obs.float())
            next_obs = self.agent.aug(next_obs.float())
            # encode
            obs = self.agent.encoder(obs)
            with torch.no_grad():
                next_obs = self.agent.encoder(next_obs)

            obs_critic_trunk = self.agent.critic.trunk(obs) # (batch_size, 50)
            obs_actor_trunk = self.agent.actor.trunk(obs)
            obs_latent_state = torch.cat([obs_critic_trunk, obs_actor_trunk], axis=-1) # (batch_size, 100)
            next_obs_critic_trunk = self.agent.critic.trunk(next_obs)
            next_obs_actor_trunk = self.agent.actor.trunk(next_obs)
            next_obs_latent_state = torch.cat([next_obs_critic_trunk, next_obs_actor_trunk], axis=-1)
            trajectory={
                'observations':obs_latent_state.detach().cpu().numpy(),
                'actions': action.detach().cpu().numpy(),
                'rewards': reward.detach().cpu().numpy(),
                'next_observations': next_obs_latent_state.detach().cpu().numpy()
            } 
            latent_dataset.append(trajectory)
        timestr = datetime.datetime.now().strftime('%Y%m%d%H%M')
        np.save(f"/home/jaewoo/research/v-d4rl/encoded_trajectory/{self.cfg.task_name}_{timestr}.npy", latent_dataset)

    def latent_eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        wandb.log({
            'result/episode_reward': total_reward / episode,
            'episode_length': step * self.cfg.action_repeat / episode,
            'episode': self.global_episode},
            step = self.global_step  
            )




@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    
    if cfg.snapshot_dir:
        load_flag=True
    else:
        snapshot = root_dir / 'snapshot.pt'
        load_flag=False

    if cfg.gta and load_flag:
        print("\n\n============<GTA>============\n\n")
        workspace.load_snapshot(cfg.snapshot_dir)
        workspace.fine_tune_with_GTA(cfg.offline_dir, cfg.gta)

    else:
        if load_flag:
            print(f'resuming training')
            workspace.load_snapshot(cfg.snapshot_dir)
            if cfg.save_latent_trajectory:
                workspace.save_latent_trajectory(cfg.offline_dir)

        if cfg.offline:
            workspace.train_offline(cfg.offline_dir)
            if cfg.save_latent_trajectory:
                workspace.save_latent_trajectory(cfg.offline_dir)

        else :
            workspace.train()
            if cfg.save_latent_trajectory:
                workspace.save_latent_trajectory(cfg.offline_dir)



        

if __name__ == '__main__':
    main()

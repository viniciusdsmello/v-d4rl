# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import h5py
from collections import deque
import dmc
from dm_env import StepType
from drqbc.numpy_replay_buffer import EfficientReplayBuffer

import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


step_type_lookup = {
    0: StepType.FIRST,
    1: StepType.MID,
    2: StepType.LAST
}

def merge_dictionary(list_of_Dict):
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


def load_offline_dataset_into_buffer(offline_dir, replay_buffer, frame_stack, replay_buffer_size):
    print("Loading offline dataset from : ", offline_dir)
    filenames = sorted(offline_dir.glob('*.hdf5'))
    num_steps = 0
    for filename in filenames:
        try:
            episodes = h5py.File(filename, 'r')
            episodes = {k: episodes[k][:] for k in episodes.keys()}
            add_offline_data_to_buffer(episodes, replay_buffer, framestack=frame_stack)
            length = episodes['reward'].shape[0]
            num_steps += length
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        print("Loaded {} offline timesteps so far...".format(int(num_steps)))
        if num_steps >= replay_buffer_size:
            break
    print("Finished, loaded {} timesteps.".format(int(num_steps)))

def load_generated_dataset_into_buffer(gta_dir, replay_buffer, frame_stack, replay_buffer_size):
    print("Loading generated dataset")
    num_steps = 0
    if 'npy' in str(gta_dir):
        episodes = np.load(gta_dir, allow_pickle=True) # list of dictionaries
    elif 'npz' in str(gta_dir):
        episodes = np.load(gta_dir, allow_pickle=True)['data'] # list of dictionaries
    else:
        raise RuntimeError("Wrong directory for the generated dataset")
    for episode in episodes:
        episode['step_type'] = np.ones_like(episode['rewards'], dtype=np.int).reshape(-1)
        episode['step_type'][0] = 0
        episode['step_type'][-1] = 2
    
    episodes = merge_dictionary(episodes)
    episodes['action'] = episodes['actions']
    episodes['observation'] = episodes['observations']
    episodes['reward'] = episodes['rewards']
    episodes['discount'] = np.ones_like(episodes['reward']).reshape(-1) ######## 여기 1로 원본데이터에는 되어 있음. 하드코딩. TODO

    replay_buffer.on_gta_flag()
    add_offline_data_to_buffer(episodes, replay_buffer, framestack=frame_stack)
    length = episodes['reward'].shape[0]
    num_steps += length
    if num_steps >= replay_buffer_size:
        raise RuntimeError("data is larger than buffer size")
    print("Finished, loaded {} timesteps.".format(int(num_steps)))

def add_offline_data_to_buffer(offline_data: dict, replay_buffer: EfficientReplayBuffer, framestack: int = 3):
    offline_data_length = offline_data['reward'].shape[0]
    for v in offline_data.values():
        assert v.shape[0] == offline_data_length # 데이터 길이 맞는지 검정
    for idx in range(offline_data_length):
        time_step = get_timestep_from_idx(offline_data, idx) # idx 번의 step_type, reward, observation, discount, actions
        if not time_step.first(): # 맨 마지막이거나 맨 처음일 때
            stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
            # print(time_step.observation)
            replay_buffer.add(time_step_stack)
            # print(kyle)
        else: # 일반 가동중일때
            stacked_frames = deque(maxlen=framestack)
            while len(stacked_frames) < framestack:
                stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
            replay_buffer.add(time_step_stack)


def get_timestep_from_idx(offline_data: dict, idx: int):
    return dmc.ExtendedTimeStep(
        step_type=step_type_lookup[offline_data['step_type'][idx]],
        reward=offline_data['reward'][idx],
        observation=offline_data['observation'][idx],
        discount=offline_data['discount'][idx],
        action=offline_data['action'][idx]
    )

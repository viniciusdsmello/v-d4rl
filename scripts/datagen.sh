ENVNAME=walker_walk # choice in ['walker_walk', 'cheetah_run', 'humanoid_walk']
TYPE=medium # choice in ['random', 'medium_replay', 'medium', 'medium_expert', 'expert']
gta= ## 이게 있으면 여기서 저장된 pt를 불러온다. '/home/taeyoung/v-d4rl/encoded_trajectory/5M-1_1x-smallmixer-10-sar-temp2_0_50.npz'
save_latent_trajectory=True
num_train_frames=2200000 # 1100000
offline_dir=/home/jaewoo/research/v-d4rl/vd4rl_data/main/walker_walk/medium ## /home/taeyoung/v-d4rl/vd4rl_data/main/walker_walk/medium or /home/taeyoung/v-d4rl/encoded_trajectory/offline_walker_walk_medium_ver1.npy

device=cuda:0

python drqbc/train.py  offline_dir=$offline_dir device=$device task_name=offline_${ENVNAME}_${TYPE}  nstep=3 seed=0 gta=$gta num_train_frames=$num_train_frames save_latent_trajectory=$save_latent_trajectory &
wait

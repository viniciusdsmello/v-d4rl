ENVNAME=walker_walk # choice in ['walker_walk', 'cheetah_run', 'humanoid_walk']
TYPE=medium # choice in ['random', 'medium_replay', 'medium', 'medium_expert', 'expert']
gta=/home/jaewoo/research/v-d4rl/encoded_trajectory/offline_walker_walk_medium_202403240152.npy ## 이게 있으면 여기서 저장된 pt를 불러온다. 
save_latent_trajectory=True
num_train_frames=1100000 # 1100000


python drqbc/train.py task_name=offline_${ENVNAME}_${TYPE} offline_dir=/home/jaewoo/research/v-d4rl/vd4rl_data/main/walker_walk/medium nstep=3 seed=0 gta=$gta num_train_frames=$num_train_frames save_latent_trajectory=$save_latent_trajectory

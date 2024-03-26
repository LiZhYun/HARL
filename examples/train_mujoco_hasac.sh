#!/bin/bash

#SBATCH --job-name=mujoco-hasac
#SBATCH --output=./out/mujoco-hasac%A_%a.out # Name of stdout output file
#SBATCH --error=./out/mujoco-hasac%A_%a.txt  # Name of stderr error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4

env="mamujoco"
scenario=$1
agent_conf=$2

algo="hasac"
exp="check"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/zhaow7/mujoco_football_triton.sif /bin/sh -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhaow7/.mujoco/mujoco210/bin; python ./train.py \
--env ${env} --algo ${algo} --exp_name ${exp} \
--scenario ${map} --agent_conf ${agent_conf} --seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 40 \
--num_env_steps 10000000 --eval_episodes 5"


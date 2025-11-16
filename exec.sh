#!/bin/bash
#SBATCH --job-name=run_vllm_cfg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=h100n2
#SBATCH --nodelist=dgx-H100-02


HF_HOME="/raid/user_iago/.cache/"
OMP_NUM_THREADS=1

source /cm/shared/apps/conda/etc/profile.d/conda.sh
conda activate /raid/user_iago/conda_envs/vllm_iago

export HF_HOME="/raid/user_iago/guardrail/mijabench/hf_cache"

# Set the port
VLLM_PORT=8000

echo "LOADING VLLM"
# Add --host 0.0.0.0 and --port $VLLM_PORT
vllm serve google/gemma-3-27b-it \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --max-model-len 12000 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --max_num_batched_tokens 32000 & # Run in background

echo "Waiting for vLLM server to start..."
while ! curl -s http://localhost:$VLLM_PORT/v1/models >/dev/null; do
    sleep 2  # Check every 2 seconds
done

echo "********************************************"
echo "*** vLLM server started ***"
echo "Access it from inside the cluster at:"
echo "http://$(hostname):$VLLM_PORT"
echo "********************************************"

# Keep the job alive
wait

---
description: Use this file to provide instructions to the agent for running jobs on the AI cluster managed via Slurm. It includes best practices for GPU usage, job submission, and monitoring.
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

Slurm Job Submission & GPU Usage Best Practices (AI Cluster Guide)
Overview
This document defines the standard practices for running jobs on the AI cluster managed via Slurm.
The goals are:
Efficient GPU utilization
Avoiding idle GPU reservation
Proper CPU vs GPU workload separation
Fair resource sharing across users
Preventing low-utilization GPU jobs
A monitoring system is active:
If GPU utilization is below 20% for more than 1 hour, users will receive an automatic email alert
1. Critical Rule
If your job does NOT require a GPU
You MUST follow BOTH rules:
Rule 1: Use CPU-only node
#SBATCH --nodelist=orion

Rule 2: Do NOT request GPU
#SBATCH --gres=gpu:1   # NOT allowed for CPU-only jobs







Example: CPU-only Job (Correct)

#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --output=cpu_out.txt
#SBATCH --error=cpu_err.txt
#SBATCH --time=2-00:00
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=orion


cd $SLURM_SUBMIT_DIR
python script.py



2. When to Use GPU
Use GPU ONLY for:
Deep learning training
GPU inference
CUDA-based computation
Large tensor/matrix workloads
Example: GPU Job
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=gpu_out.txt
#SBATCH --error=gpu_err.txt
#SBATCH --time=3-00:00
#SBATCH --mem=60000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=tesla2


cd $SLURM_SUBMIT_DIR
python train.py
3. GPU Usage Monitoring System
Automatic Email Alerts
Users will receive an email if:
GPU utilization is < 20% for more than 1 hour
This is automatically enforced.


How Users Can Check Their GPU Utilization And Allocation Details?
1. GPU usage log file
Generated inside every GPU job:
gpu_usage.csv

Check after job completion:
cat gpu_usage.csv

Every GPU job SHOULD include the following code snippet:
echo "timestamp,gpu_index,utilization_percent,memory_used_MB,memory_total_MB" > gpu_usage.csv


(
while true; do
   nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
              --format=csv,noheader,nounits >> gpu_usage.csv
   sleep 60
done
) &


GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID" EXIT
Full Script Example:
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_out.txt
#SBATCH --error=gpu_test_err.txt
#SBATCH --time=00:20:00
#SBATCH --mem=4000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=tesla2
cd $SLURM_SUBMIT_DIR
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "timestamp,gpu_index,utilization_percent,memory_used_MB,memory_total_MB" > gpu_usage.csv
(
while true; do
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits >> gpu_usage.csv
  sleep 60   # better interval for real monitoring
done
) &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID" EXIT
#Your training command here
python train.py
echo "Job finished at: $(date)"

What this does:
Logs GPU utilization every 60 seconds
Stores data in gpu_usage.csv
Automatically stops when job ends
2. Slurm job inspection
scontrol show job <jobid>
Shows:
Node assignment
GPU allocation
3. Post-job accounting
sacct -j <jobid>
Shows:
Runtime
Resource usage
Allocation details



4. Recommended Workflow Design
Many projects have both CPU and GPU bottlenecks.
Correct Approach: Split pipeline into stages
Step 1: CPU Preprocessing (Orion)
If your workload includes:
Data cleaning
Tokenization
Feature extraction
Dataset building
Run it on CPU node
#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess_out.txt
#SBATCH --error=preprocess_err.txt
#SBATCH --time=1-00:00
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=orion


python preprocess.py --output processed_data/

Saves processed dataset for later steps.
Step 2: GPU Training
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train_out.txt
#SBATCH --error=train_err.txt
#SBATCH --time=3-00:00
#SBATCH --mem=60000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=tesla2


python train.py --data processed_data/ --save checkpoints/





Step 3: CPU Evaluation (Orion)
#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=eval_out.txt
#SBATCH --error=eval_err.txt
#SBATCH --time=1-00:00
#SBATCH --mem=30000
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=orion


python evaluate.py --model checkpoints/




5. Why splitting is required
If everything is done in one GPU job:
Bad practice:
GPU sits idle during preprocessing
Low utilization (<20%)
Waste of expensive resources
Email alerts triggered
Good practice:
CPU handles data preparation
GPU handles computation only
CPU handles evaluation separately


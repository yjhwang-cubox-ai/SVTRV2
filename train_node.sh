#!/bin/bash -l
#SBATCH --job-name=SVTR
#SBATCH --nodelist=hpe160,hpe162
#SBATCH --nodes=2            # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=8     # This needs to match Trainer(devices=...)
#SBATCH --gpus=16
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_logs/mylog-%j.out

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "Number of nodes:= " "$SLURM_JOB_NUM_NODES"
echo "Ntasks per node:= "  "$SLURM_NTASKS_PER_NODE"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

srun --container-image /purestorage/project/yjhwang/torch_lightning_241202.sqsh \
    --container-mounts /purestorage:/purestorage \
    --no-container-mount-home \
    --container-writable \
    --container-workdir /purestorage/project/yjhwang/SVTRV2 \
    bash -c "
    hostname -I;
    python train.py"
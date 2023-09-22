#!/bin/bash --login
#SBATCH --job-name=gemm_exhaustive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu-dev
#SBATCH --account=director2178-gpu
#SBATCH --time=04:00:00

if [ ! -f "$INPUT" ]; then
  echo "INPUT FILE \"$INPUT\" DOES NOT EXIST"
  exit 1
fi
if [ ! -f "$EXE" ]; then
  echo "EXE FILE \"$EXE\" DOES NOT EXIST"
  exit 1
fi

RUN=`readlink -f run.sh`

DIR=exhaustive/${SLURM_JOB_NAME}${SLURM_JOBID}
mkdir $DIR && cd $DIR
srun -N 1 -n 1 -c 8 --gpus-per-node=1 --gpus-per-task=1 $RUN $INPUT $EXE

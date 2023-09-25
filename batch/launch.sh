#!/bin/bash
export INPUT=`readlink -e $1`
export DIR=$2
export EXE=`readlink -e ../build/src/app/run_tests_exhaustive`

if [ ! -f "$INPUT" ]; then
  echo "INPUT FILE \"$INPUT\" DOES NOT EXIST"
  exit 1
fi
if [ ! -f "$EXE" ]; then
  echo "EXE FILE \"$EXE\" DOES NOT EXIST"
  exit 1
fi
if [ -z "$DIR" ]; then
  echo "PROVIDE DIRECTORY PLEASE"
  exit 1
fi

echo "Running input=" $INPUT " exe=" $EXE
if which sbatch &> /dev/null; then 
  sbatch --export=ALL,DIR="${DIR}",INPUT="${INPUT}",EXE="${EXE}" slurm_job.sh 
elif which qsub &> /dev/null; then
  qsub -v DIR="${DIR}",INPUT="${INPUT}",EXE="${EXE}" pbs_job.sh 
fi

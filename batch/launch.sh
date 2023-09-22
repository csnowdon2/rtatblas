#!/bin/bash
export INPUT=`readlink -f $1`
export EXE=`readlink -f ../build/src/app/run_tests_exhaustive`

if [ ! -f "$INPUT" ]; then
  echo "INPUT FILE \"$INPUT\" DOES NOT EXIST"
  exit 1
fi
if [ ! -f "$EXE" ]; then
  echo "EXE FILE \"$EXE\" DOES NOT EXIST"
  exit 1
fi

echo "Running input=" $INPUT " exe=" $EXE
sbatch --export=ALL,INPUT="${INPUT}",EXE="${EXE}" slurm_job.sh 

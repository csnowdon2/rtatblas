#!/bin/bash
#PBS -q gpuvolta
#PBS -j oe
#PBS -l walltime=01:00:00,mem=90GB
#PBS -l wd
#PBS -l jobfs=20GB
#PBS -l ncpus=12
#PBS -l ngpus=1

if [ ! -f "$INPUT" ]; then
  echo "INPUT FILE \"$INPUT\" DOES NOT EXIST"
  exit 1
fi
if [ ! -f "$EXE" ]; then
  echo "EXE FILE \"$EXE\" DOES NOT EXIST"
  exit 1
fi

RUN=`readlink -f run-experiment.sh`

$RUN $EXE $DIR $INPUT

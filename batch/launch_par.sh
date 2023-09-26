#!/bin/bash
export INPUT=`readlink -e $1`
export DIR=$2
export EXE=`readlink -e ../build/src/app/run_tests_exhaustive`

ROOT=`pwd`

if [ ! -f "$INPUT" ]; then
  echo "INPUT FILE \"$INPUT\" DOES NOT EXIST"
  exit 1
fi
if [ ! -f "$EXE" ]; then
  echo "EXE FILE \"$EXE\" DOES NOT EXIST"
  exit 1
fi

SUFFIX=".part"
if ! [ -d ${DIR} ]; then
  # Check input
  echo "Directory not found, creating directory and splitting input..."
  echo "Input: ${INPUT}"

  mkdir ${DIR} && cd ${DIR}
  split -l 5 --additional-suffix=${SUFFIX} ${INPUT} $(basename ${INPUT})

  for FILE in `find . -type f`
  do
    SUBDIR=subdir_$(basename ${FILE})
    mkdir $SUBDIR
    mv $FILE $SUBDIR

    echo "Running input=" $INPUT " exe=" $EXE
    if which sbatch &> /dev/null; then 
      sbatch --export=ALL,DIR="$(readlink -f ${SUBDIR})",INPUT="",EXE="${EXE}" -D $ROOT $ROOT/slurm_job.sh
    elif which qsub &> /dev/null; then
      echo "CALUM NEED TO SET WORKING DIRECTORY IN QSUB"
      exit 1
      qsub -v DIR="$(readlink -f ${SUBDIR})",INPUT="",EXE="${EXE}" $ROOT/pbs_job.sh 
    fi
  done
else
  echo "Directory already exists, aborting"
fi


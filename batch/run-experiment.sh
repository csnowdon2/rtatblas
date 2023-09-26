#!/bin/bash
# All paths absolute
EXE=$1
DIR=$2
INPUT=$3

# An experiment is defined by an input along with a directory name
# and an executable

# Check EXE
if ! readlink -e ${EXE}; then
  echo "ERROR: Expected absolute path to executable, got: ${EXE}"
  exit 1
else
  echo "Found executable: ${EXE}"
fi

# Check DIR
if ! [[ "${DIR}" = /* ]]; then
  echo "ERROR: Expected absolute path to working directory, got: ${DIR}"
else
  echo "Working directory: ${DIR}"
fi

# Crate directory and inputs if need be
SUFFIX=".part"
if ! [ -d ${DIR} ]; then
  # Check input
  echo "Directory not found, creating directory and splitting input..."
  if [ -z "${INPUT}" ] || ! readlink -e ${INPUT}; then
    echo "ERROR: Expected absolute path to input, got: ${INPUT}"
    exit 1
  else
    echo "Input: ${INPUT}"
  fi

  mkdir ${DIR} && cd ${DIR}
  split -l 5 --additional-suffix=${SUFFIX} ${INPUT} $(basename ${INPUT})
else
  echo "Directory already exists, checking remaining inputs"
  cd ${DIR}
fi

# Run inputs
INPUTS=`find . -type f -iname "*${SUFFIX}"`
echo "Running inputs" ${INPUTS}
for FILE in $INPUTS
do
 echo $EXE $FILE
 $EXE $FILE
 rm $FILE
done

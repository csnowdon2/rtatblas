#!/bin/bash
FILENAME=$1
EXE=$2

SUFFIX=".part"

split -l 5 ${FILENAME} $(basename ${FILENAME})${SUFFIX}
INPUTS=`find . -type f -iname "*${SUFFIX}*"`
for FILE in $INPUTS
do
 echo $EXE $FILE
 $EXE $FILE
 rm $FILE
done

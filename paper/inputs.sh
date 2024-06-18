#!/bin/bash

GENERATE=../scripts/input.py

OUTPUT_DIR=$1
MEM=$2
re='^[0-9]+$'
if ! [[ $MEM =~ $re ]] ; then
  echo "Error: must input a positive integer"
  exit 1
fi

SMALL=$(echo "scale=8 ; $MEM / 4096" | bc),$(echo "scale=3 ; $MEM / 1024" | bc)
LARGE=$(echo "scale=8 ; $MEM / 32" | bc),$(echo "scale=3 ; $MEM / 4" | bc)

EX_REPS=5
AT_REPS=20

# EXHAUSTIVE TESTS: determine the extent to which 
# RTAT transformations can improve performance
# RECTANGULAR 3
for METHOD in gemm trsm syrk; do
  for PRECISION in float double; do
    for TYPE in exhaustive; do
      python3 $GENERATE -s 500 -n 100 -r ${EX_REPS} -m ${SMALL} -a 3 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_rectangle_small.json
      python3 $GENERATE -s 600 -n 100 -r ${EX_REPS} -m ${LARGE} -a 3 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_rectangle_large.json
    done
  done
done

# SQUARE
for METHOD in gemm trsm syrk; do
  for PRECISION in float double; do
    for TYPE in exhaustive; do
      python3 $GENERATE -s 500 -n 30 -r ${EX_REPS} -m ${SMALL} -a 0 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_square_small.json
      python3 $GENERATE -s 600 -n 30 -r ${EX_REPS} -m ${LARGE} -a 0 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_square_large.json
    done
  done
done

# AUTOTUNE TESTS: For the same shapes as before, run 20 reps with 
# autotuning. Determine speedups relative to exhaustive tests
# RECTANGULAR 3
for METHOD in gemm trsm syrk; do
  for PRECISION in float double; do
    for TYPE in autotune; do
      python3 $GENERATE -s 500 -n 100 -r ${AT_REPS} -m ${SMALL} -a 3 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_rectangle_small.json
      python3 $GENERATE -s 600 -n 100 -r ${AT_REPS} -m ${LARGE} -a 3 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_rectangle_large.json
    done
  done
done

# SQUARE
for METHOD in gemm trsm syrk; do
  for PRECISION in float double; do
    for TYPE in autotune; do
      python3 $GENERATE -s 500 -n 30  -r ${AT_REPS} -m ${SMALL} -a 0 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_square_small.json
      python3 $GENERATE -s 600 -n 30  -r ${AT_REPS} -m ${LARGE} -a 0 --opts random --rt $TYPE --dt $PRECISION --method $METHOD -o ${OUTPUT_DIR}/${METHOD}_${PRECISION}_${TYPE}_square_large.json
    done
  done
done


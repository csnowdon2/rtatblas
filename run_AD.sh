#!/bin/bash


shapes=("960 324480 960" "120 2957880 120" "192 738048 192")
reps=20
DIR="./build/src/app/"

echo "Running computations with ${reps} repetitions each"

for shape in "${shapes[@]}"
do
  echo "( m k n ) = (" ${shape} ")"
  for options in "N N" "N T" "T N" "T T"
  do
    echo -n "${options}: "
    ${DIR}/simple_test ${shape} ${options} ${reps}
  done
  #echo -n "Autotuned: "
  #./build/src/app/autotune ${shape} N N ${reps}
  echo
done

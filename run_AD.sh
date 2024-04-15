#!/bin/bash


shapes=("960 324480 960" "120 2957880 120" "192 738048 192")
reps=20

for shape in "${shapes[@]}"
do
  echo ${shape}
  for options in "N N" "N T" "T N" "T T"
  do
    echo -n "${options}: "
    ./build/src/app/simple_test ${shape} ${options} ${reps}
  done
  echo -n "Autotuned: "
  ./build/src/app/autotune ${shape} N N ${reps}
  echo
done

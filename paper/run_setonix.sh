#!/bin/bash

mkdir setonix_tests
./inputs.sh $(readlink -f setonix_tests) 64
cd setonix_tests

for file in *.json ; do
  python3 ../../scripts/launch_setonix.py $file
done

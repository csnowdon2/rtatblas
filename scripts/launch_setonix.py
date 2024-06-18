import subprocess
import sys
import json
from estimate_time import estimate_time

filename = sys.argv[1]


input_json = None
with open(filename) as file:
    input_json = json.load(file)

hours = 2*estimate_time(input_json, 26, 52)

cmd = ["sbatch", "--account=director2178-gpu", "--nodes=1",
       "--gpus-per-node=1", "--partition=gpu", 
       f"--time={hours*60}",
       "--ntasks=1", 
       f'--wrap="../../build/src/app/run_tests {filename} > out_{filename}"']
       #f'--wrap="../build/src/app/run_tests {filename}"']
#print(" ".join(cmd))
subprocess.run(" ".join(cmd), shell=True)

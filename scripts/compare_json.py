import pandas as pd
import itertools
import json
import sys
import os

naive=sys.argv[1]
autotune=sys.argv[2]
columns=['m','k','n','opA','opB','ms']
opcols=columns[5:11]

ix_pad = [(*x,*y) for x in itertools.product(["N","T"],repeat=3) 
                  for y in itertools.product(["N","P"],repeat=3)]
ix_nopad = [(*x,*y) for x in itertools.product(["N","T"],repeat=3)
                    for y in itertools.product(["N"],repeat=3)]

def readdir(directory):
  rows = []
  for subdir, dirs, files in os.walk(directory):
    for file in files:
      if file.split('.')[-1][0] == 'o':
        data = []
        with open(os.path.join(subdir, file), 'r') as f:
          data = json.load(f)
        for o in data:
          rowbase = [o[col] for col in columns[:5]]
  
          avg = 0.0
          tot = 0
          for opts in o['results']:
            for val in o['results'][opts]['data']:
              avg += val
              tot += 1
          avg = avg/tot
          rowext = [avg]
          rows.append(rowbase+rowext)
  return rows

df = pd.DataFrame(readdir(autotune), columns=columns)
df_naive = pd.DataFrame(readdir(naive), columns=columns)

df['tflops'] = df['m']*df['k']*df['n']*2/(df['ms']/1000)/1e12
df['naive_ms'] = df_naive['ms']
df['naive_tflops'] = df['m']*df['k']*df['n']*2/(df['naive_ms']/1000)/1e12
df['speedup'] = df['naive_ms']/df['ms']

print(df.describe())

#enabled_options = dict()
for (opA, opB) in itertools.product(['N','T'], repeat=2):
  print("OPS", opA, opB)
  op_df = df[(df['opA'] == opA) & (df['opB'] == opB)]
  op_df = op_df.set_index(columns[:5])
  print(op_df.describe())

print(df.sort_values(by=['speedup']))

print(f"AGGREGATE SPEEDUP OVER NAIVE: {sum(df['naive_ms'])/sum(df['ms']):.2f}")
print(f"Total naive time: {sum(df['naive_ms'])}")
print(f"Total autot time: {sum(df['ms'])}")

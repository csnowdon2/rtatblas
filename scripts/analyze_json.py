import pandas as pd
import json
import sys
import os

directory=sys.argv[1]


rows = []
for subdir, dirs, files in os.walk(directory):
  for file in files:
    print(file)
    if file.split('.')[-1][0] == 'o':
      data = []
      with open(os.path.join(subdir, file), 'r') as f:
        data = json.load(f)
      for m in data:
        for k in data[m]:
          for n in data[m][k]:
            for op in data[m][k][n]:
              for options in data[m][k][n][op]:
                rows.append([m,k,n,op,options,data[m][k][n][op][options]['mean']])

df = pd.DataFrame(rows)
df = df.rename(columns={0: "m", 1: "k", 2: "n", 3: "op", 4: "options" , 5: "mean"})

print(df.head())

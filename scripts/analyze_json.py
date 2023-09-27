import pandas as pd
import itertools
import json
import sys
import os

directory=sys.argv[1]
columns=['m','k','n','opA','opB','tA','tB','tC','pA','pB','pC','ms']
opcols = columns[5:11]

ix_pad = [(*x,*y) for x in itertools.product(["N","T"],repeat=3) 
                  for y in itertools.product(["N","P"],repeat=3)]
ix_nopad = [(*x,*y) for x in itertools.product(["N","T"],repeat=3)
                    for y in itertools.product(["N"],repeat=3)]

  
def hmean(ser):
  return ser.size/(1/ser).sum()

# goodness(opts) = mean performance of best options among opts
def goodness(df, opts):
  data = pd.DataFrame()
  for options in opts:
    data[str(options)] = df.unstack(level=opcols)[('tflops', *options)]

  data['best'] = data.max(axis=1)
  return hmean(data['best'])

def greedy_select(df, ix, n):
  current_options = []
  possibilities = set(ix)
  for i in range(0,n):
    best = 0
    best_op = None
    for op in possibilities:
      good = goodness(df, current_options + [op])
      if good > best:
        best = good
        best_op = op
    possibilities.remove(best_op)
    current_options.append(best_op)
    print("Best group of", i+1, " = ", best, current_options)
  return current_options

rows = []
for subdir, dirs, files in os.walk(directory):
  for file in files:
    if file.split('.')[-1][0] == 'o':
      data = []
      with open(os.path.join(subdir, file), 'r') as f:
        data = json.load(f)
      for o in data:
        rowbase = [o[col] for col in columns[:5]]

        for opts in o['results']:
          rowext = list(opts) + [o['results'][opts]['mean']]
          rows.append(rowbase+rowext)

if rows:
  print("DIR", subdir)
  # Transform data, add tflops
  df = pd.DataFrame(rows, columns=columns)
  df['tflops'] = df['m']*df['k']*df['n']*2/(df['ms']/1000)/1e12

  enabled_options = dict()
  for (opA, opB) in itertools.product(['N','T'], repeat=2):
    print("OPS", opA, opB)
    op_df = df[(df['opA'] == opA) & (df['opB'] == opB)]
    op_df = op_df.set_index(columns[:11])

    # Calculate best options for each shape
    idx = op_df.groupby(level=columns[:5])['ms'].transform(min) == op_df['ms']
    print("Mean of best", hmean(op_df[idx]['tflops']))
    print("Baseline    ", hmean(op_df.unstack(level=opcols)[('tflops', 'N', 'N', 'N', 'N', 'N', 'N')]))

    # Find options with best coverage
    print("With pad:")
    best_opts = greedy_select(op_df,ix_pad,4)
    #print("Without pad:")
    #best_opts = greedy_select(op_df,ix_nopad,4)
    print()

    enabled_options[(opA,opB)] = best_opts
  print("Predicates:")
  for key in enabled_options:
    for opts in enabled_options[key]:
      print(''.join(list(key)), ''.join(list(opts)))

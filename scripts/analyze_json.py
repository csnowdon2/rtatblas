import pandas as pd
import itertools
import json
import sys
import os

directory=sys.argv[1]
columns=['m','k','n','opA','opB','tA','tB','tC','pA','pB','pC','ms']
opcols = columns[5:11]

ix = [(*x,*y) for x in itertools.product(["N","T"],repeat=3) 
              for y in itertools.product(["N","P"],repeat=3)]

  
def hmean(ser):
  return ser.size/(1/ser).sum()

# goodness(opts) = mean performance of best options among opts
def goodness(df, opts):
  data = pd.DataFrame()
  for options in opts:
    data[str(options)] = df.unstack(level=opcols)[('tflops', *options)]

  data['best'] = data.max(axis=1)
  return hmean(data['best'])

def greedy_select(df, n):
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

for subdir, dirs, files in os.walk(directory):
  rows = []
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
    for (opA, opB) in itertools.product(['N','T'], repeat=2):
      print("OPS", opA, opB)
      op_df = df[(df['opA'] == opA) & (df['opB'] == opB)]
      op_df = op_df.set_index(columns[:11])

      # Calculate best options for each shape
      idx = op_df.groupby(level=columns[:5])['ms'].transform(min) == op_df['ms']
      print("Mean of best", hmean(op_df[idx]['tflops']))
      print("Baseline    ", hmean(op_df.unstack(level=opcols)[('tflops', 'N', 'N', 'N', 'N', 'N', 'N')]))

      # Find options with best coverage
      best_opts = greedy_select(op_df,4)
      print()
    print() 
      # Exhaustive selection
      #best = 0
      #best_opts = None
      #for i in range(1,4):
      #  for options in itertools.combinations(ix,i):
      #    good = goodness(df, options)
      #    if good > best:
      #      best = good
      #      best_opts = options
      #  print("Best group of", i, " = ", best, best_opts)

      # Sort options by performance
      #perf_by_opts = dict()
      #for options in ix:
      #  perf_by_opts[options] = df.unstack(level=opcols)[('tflops', *options)].mean()
      #print([(a,perf_by_opts[a]) for a in sorted(list(perf_by_opts),key = lambda x:perf_by_opts[x], reverse=True)])

      #print(df.groupby(level=columns[:5]).apply(max))
      #print(df[df[('ms','N')] > 1.05*df[('ms','T')]])



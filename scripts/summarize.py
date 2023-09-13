import pandas as pd
import sys

def read_result_file(filename):
  df = pd.read_csv(filename, sep=' ', header=None)
  df = df.rename(columns={0: "m", 1: "k", 2: "n", 3: "opA", 4: "opB", 5: "TFLOPs"})
  return df

def split_by_op(df):
  ret = dict()
  ret["NN"] = df[(df['opA'] == 'N') & (df['opB'] == 'N')].reset_index()
  ret["NT"] = df[(df['opA'] == 'N') & (df['opB'] == 'T')].reset_index()
  ret["TN"] = df[(df['opA'] == 'T') & (df['opB'] == 'N')].reset_index()
  ret["TT"] = df[(df['opA'] == 'T') & (df['opB'] == 'T')].reset_index()
  return ret

filename = sys.argv[1]
df = read_result_file(filename)
dfs = split_by_op(df)
#for key in dfs:
#  print(f'Type {key} mean TFLOP/s {dfs[key]["TFLOPs"].mean():.2f}')

transformed = dfs["NN"][['m','k','n']].copy()
for key in dfs:
  transformed[key] = dfs[key]['TFLOPs']

transformed['best'] = transformed[['NN','NT','TN','TT']].max(axis=1)
transformed['bestarg'] = transformed[['NN','NT','TN','TT']].idxmax(axis=1)
print(transformed.describe())
print(transformed['bestarg'].value_counts())

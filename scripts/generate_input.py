import argparse
import random
import sys
import itertools
from enum import Enum

class Args:
  def __init__(self):
    parser = argparse.ArgumentParser(description="Randomly generate GEMM problems.")
    parser.add_argument('-t', default='double', choices=['float','double'], help='Matrix data type')
    parser.add_argument('-a', default=3, type=int, help='Aspect ratio bound')
    parser.add_argument('-n', default=10, type=int, help='Number of problems to generate')
    parser.add_argument('--axis', default='mem', choices=['mem','flop'])
    parser.add_argument('--transposes', default='exhaustive', 
                        choices=['exhaustive','random','NN','NT','TN','TT'])
    parser.add_argument('-r', '--range', required=True, help='Memory footprint lower/upper bounds')
    parser.add_argument('-o', default='', help='Output file name')
    parser.add_argument('-s', default='0', type=int, help='Output file name')

    args = parser.parse_args()

    if args.t == 'float':
      self.dt_size = 4
    else:
      self.dt_size = 8

    self.aspect_bound = args.a
    self.num_problems = args.n
    self.random_axis = args.axis
    self.transposes = args.transposes

    mem_bounds = args.range.split(',')
    self.mem_lb = int(float(mem_bounds[0])*1024**3)
    self.mem_ub = int(float(mem_bounds[1])*1024**3)

    self.filename = args.o
    self.seed = args.s


class Op:
  op_vals = ['N', 'T']

  def __init__(self, c):
    if not c in Op.op_vals:
      raise ValueError("Bad Op value")
    self.val = c

  def __str__(self):
    return self.val

  def __repr__(self):
    return self.val

  def random():
    return Op(random.choice(Op.op_vals))

  def enumerate():
    return itertools.product(Op.op_vals, repeat=2)


class AspectDims:

  def get_dims(self, N):
    return (int(N), int(N*self.aspect1), int(N*self.aspect1*self.aspect2))

  def calc_flops(self, N):
    (m,k,n) = get_dims(self, N)
    return 2*m*k*n

  def calc_footprint(self, N, dt_size):
    (m,k,n) = get_dims(self, N)
    return self.dt_size*(m*k+m*n+k*n)

  def N_from_flops(self, flopcount):
    return int((flopcount/(2*self.aspect1*self.aspect1*self.aspect2))**(1/3))

  def N_from_mem(self, mem):
    return int((mem/(self.dt_size*self.aspect1*(1+self.aspect1*self.aspect2+self.aspect2)))**(1/2))

  def generate_aspect(bound):
    return 2**(random.uniform(-bound,bound))

  def __init__(self):
    self.aspect1 = 1.0
    self.aspect2 = 1.0

  def __init__(self, bound, dt_size):
    self.aspect1 = AspectDims.generate_aspect(bound)
    self.aspect2 = AspectDims.generate_aspect(bound)
    self.dt_size = dt_size


class Problem:

  def __init__(self, m, k, n, opA, opB):
    self.m = m
    self.k = k
    self.n = n
    self.opA = opA
    self.opB = opB

  def __repr__(self):
    ret = f"{self.m} {self.k} {self.n} {self.opA} {self.opB} 0"
    return ret

def random_dims_mem(mem_lb, mem_ub, aspect_bound, dt_size):
  aspects = AspectDims(aspect_bound, dt_size)

  mem = random.uniform(mem_lb, mem_ub)
  N = aspects.N_from_mem(mem)
  return aspects.get_dims(N)

def random_dims_flop(mem_lb, mem_ub, aspect_bound, dt_size):
  aspects = AspectDims(aspect_bound, dt_size)

  n_lb = aspects.N_from_footprint(mem_lb)
  n_ub = aspects.N_from_footprint(mem_ub)
  
  flop_lb = aspects.calc_flops(n_lb)
  flop_ub = aspects.calc_flops(n_ub)

  flopcount = random.uniform(flop_lb, flop_ub)
  N = aspects.N_from_flops(flopcount)
  return aspects.get_dims(N)


# BEGIN
args = Args()
random.seed(args.seed)

file = None
if args.filename:
  file = open(args.filename, 'w')
else:
  file = sys.stdout

generate = None
if args.random_axis == "mem":
  generate = random_dims_mem
elif args.random_axis == "flop":
  generate = random_dims_flop
else:
  raise ValueError("Bad random axis")


for i in range(args.num_problems):
  dims = generate(args.mem_lb, args.mem_ub, args.aspect_bound, args.dt_size)  
  oplist = [(Op.random(), Op.random())]

  if args.transposes == 'exhaustive':
    oplist = Op.enumerate()
  elif args.transposes == 'random':
    pass
  else:
    oplist = [(Op(args.transposes[0]), Op(args.transposes[1]))]

  for ops in oplist:
    problem = Problem(*dims, *ops)
    file.write(str(problem)+'\n')

if args.filename:
  file.close()

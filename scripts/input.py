# Input Generation script
# Take some parameters to describe input set
# Output json file with randomly generated problems


# GEMM:
# m, k, n, transA, transB
# FLOPS=2*m*k*n

# SYRK:
# n, k, transA, uploC
# FLOPS=k*n*(n+1)

# TRSM:
# m, n, side, uploA, transA, diagA
# FLOPS=n*m^2 (LEFT), m*n^2 (RIGHT)
import random
import argparse
import json

class Opt:
    def __init__(self, op):
        if op not in type(self).ops:
            raise Exception(f"Bad input {op} to Opt ({self.ops})")
        self.op = op

    def randomize(self):
        self.op = random.choice(type(self).ops)

    def __repr__(self):
        return self.op
            

class Trans_Opt(Opt):
    ops = ["N","T"]
    def __init__(self, op = ops[0]):
        Opt.__init__(self,op)

class Side_Opt(Opt):
    ops = ["Left","Right"]
    def __init__(self, op = ops[0]):
        Opt.__init__(self,op)

class Diag_Opt(Opt):
    ops = ["Non-Unit","Unit"]
    def __init__(self, op = ops[0]):
        Opt.__init__(self,op)

class Uplo_Opt(Opt):
    ops = ["Lower","Upper"]
    def __init__(self, op = ops[0]):
        Opt.__init__(self,op)

def enumerate(Opt_Type):
    return [Opt_Type(op) for op in Opt_Type.ops]

class GEMM_Key:
    def __init__(self, m, k, n, 
                 transA = None, 
                 transB = None):
        self.m = m
        self.k = k
        self.n = n
        self.transA = transA
        self.transB = transB
        if transA is None:
            op = Trans_Opt()
            self.transA = op
        if transB is None:
            op = Trans_Opt()
            self.transB = op

    def randomize_opts(self):
        self.transA.randomize()
        self.transB.randomize()

    def enumerate_opts(self):
        ret = []
        for transA in enumerate(Trans_Opt):
            for transB in enumerate(Trans_Opt):
                ret.append(GEMM_Key(self.m,self.k,self.n,transA,transB))
        return ret

    def flop_count(self):
        return 2*self.m*self.k*self.n

    def memory_footprint(self):
        return self.m*self.k + self.k*self.n + self.m*self.n

    def json(self):
        ret = {}
        ret["m"] = self.m
        ret["k"] = self.k
        ret["n"] = self.n
        ret["transA"] = str(self.transA)
        ret["transB"] = str(self.transB)
        return ret

class GEMM_Aspects:
    def __init__(self):
        self.aspect1 = 1.0
        self.aspect2 = 1.0

    def __init__(self, bound):
        self.aspect1 = 2**random.uniform(-bound,bound)
        self.aspect2 = 2**random.uniform(-bound,bound)

    def dims(self, N):
        return (int(N), int(N*self.aspect1), 
                 int(N*self.aspect1*self.aspect2))

    def flops(self, N):
        (m,k,n) = self.dims(N)
        return 2*m*k*n

    def memory(self, N):
        (m,k,n) = self.dims(N)
        return m*k+k*n+m*n

    def N_from_flops(self, flopcount):
        return int((flopcount/(2*self.aspect1*self.aspect1*self.aspect2))**(1/3))
    def N_from_mem(self, mem):
        return int((mem/(self.aspect1*(1+self.aspect1*self.aspect2+self.aspect2)))**(1/2))


class SYRK_Key:
    def __init__(self, n, k, 
                 trans = None, 
                 uplo = None):
        self.n = n
        self.k = k
        self.trans = trans
        self.uplo = uplo
        if trans is None:
            self.trans = Trans_Opt()
        if uplo is None:
            self.uplo = Uplo_Opt()

    def randomize_opts(self):
        self.trans.randomize()
        self.uplo.randomize()

    def enumerate_opts(self):
        ret = []
        for trans in enumerate(Trans_Opt):
            for uplo in enumerate(Uplo_Opt):
                ret.append(SYRK_Key(self.n, self.k, trans, uplo))
        return ret

    def flop_count(self):
        return k*n*(n+1)

    def memory_footprint(self):
        return self.n*self.n + self.k*self.n

    def json(self):
        ret = {}
        ret["n"] = self.n 
        ret["k"] = self.k 
        ret["trans"] = str(self.trans)
        ret["uplo"] = str(self.uplo)
        return ret

class SYRK_Aspects:
    def __init__(self):
        self.aspect = 1.0

    def __init__(self, bound):
        self.aspect = 2**random.uniform(-bound,bound)

    def dims(self, N):
        return (int(N), int(N*self.aspect))

    def flops(self, N):
        (n,k) = self.dims(N)
        return k*n*(n+1)

    def memory(self, N):
        (n,k) = self.dims()
        return n*n+k*n

    def N_from_flops(self, flopcount):
        return int((flopcount/self.aspect)**(1/3))

    def N_from_mem(self, mem):
        return int(((mem/(1+self.aspect)))**(1/2))

class TRSM_Key:
    def __init__(self, m, n, 
                 side = None, 
                 uplo = None, 
                 trans = None, 
                 diag = None):
        self.m = m
        self.n = n
        self.side = side
        self.uplo = uplo
        self.trans = trans
        self.diag = diag
        if side is None:
            self.side = Side_Opt()
        if uplo is None:
            self.uplo = Uplo_Opt()
        if trans is None:
            self.trans = Trans_Opt()
        if diag is None:
            self.diag = Diag_Opt()

    def randomize_opts(self):
        old_side = self.side
        self.side.randomize()
        if old_side != self.side:
            (self.m,self.n) = (self.n,self.m)
        self.uplo.randomize()
        self.trans.randomize()
        self.diag.randomize()

    def enumerate_opts(self):
        ret = []
        for uplo in enumerate(Uplo_Opt):
            for trans in enumerate(Trans_Opt):
                for diag in enumerate(Diag_Opt):
                    for side in enumerate(Side_Opt):
                        m = self.m if side.op == "Left" else self.n
                        n = self.n if side.op == "Left" else self.m
                        ret.append(TRSM_Key(m,n,side,uplo,trans,diag))
        return ret


    def flop_count(self):
        if self.side.op == "Left":
            return self.n*self.m*self.m
        else:
            return self.n*self.n*self.m

    def memory_footprint(self):
        if self.side.op == "Left":
            return self.m*self.m + self.m*self.n
        else:
            return self.n*self.n + self.m*self.n

    def json(self):
        ret = {}
        ret["m"] = self.m
        ret["n"] = self.n
        ret["side"] = str(self.side)
        ret["uplo"] = str(self.uplo)
        ret["trans"] = str(self.trans)
        ret["diag"] = str(self.diag)
        return ret

class TRSM_Aspects:
    def __init__(self):
        self.aspect = 1.0

    def __init__(self, bound):
        self.aspect = 2**random.uniform(-bound,bound)

    def dims(self, N):
        return (int(N), int(N*self.aspect))

    def flops(self, N):
        (m,n) = self.dims(N)
        return m*m*n

    def memory(self, N):
        (m,n) = self.dims(N)
        return m*m + n*m

    # Assumes Left side
    def N_from_flops(self, flopcount):
        return int((flopcount/self.aspect)**(1/3))

    def N_from_mem(self, mem):
        return int(((mem/(1+self.aspect)))**(1/2))

def random_gemm(mem, aspect_bound):
    aspects = GEMM_Aspects(aspect_bound)
    N = aspects.N_from_mem(mem)
    (m,k,n) = aspects.dims(N)
    return GEMM_Key(m,k,n)

def random_syrk(mem, aspect_bound):
    aspects = SYRK_Aspects(aspect_bound)
    N = aspects.N_from_mem(mem)
    (n,k) = aspects.dims(N)
    return SYRK_Key(n,k)

def random_trsm(mem, aspect_bound):
    aspects = TRSM_Aspects(aspect_bound)
    N = aspects.N_from_mem(mem)
    (m,n) = aspects.dims(N)
    return TRSM_Key(m,n)


opt = Uplo_Opt("Upper")
#print(random_gemm(10000, 3).json())
#print(random_syrk(10000, 3).json())
#print(random_trsm(10000, 3).json())

def generate(generator, count, mem_lower, mem_upper, aspect_bound,
             opts = "default", seed = 0):
    keys = []
    random.seed(seed)
    for _ in range(count):
        mem = random.uniform(mem_lower, mem_upper)
        key = generator(mem, aspect_bound)

        if opts == "default":
            keys.append(key)
        elif opts == "random":
            key.randomize_opts()
            keys.append(key)
        elif opts == "all":
            keys.extend(key.enumerate_opts())
        else:
            raise Exception(f"Invalid opts setting: {opts}")

    return keys

class Args:
  def __init__(self):
    parser = argparse.ArgumentParser(description="Randomly generate GEMM problems.")
    parser.add_argument('-n', default=10, type=int, help='Number of problems to generate')
    parser.add_argument('-m', required=True, help='Memory footprint lower/upper bounds')
    parser.add_argument('--dt', default='double', choices=['float','double'], help='Matrix data type')
    parser.add_argument('--rt', default='autotune', choices=['exhaustive','autotune'], help='Run type')
    parser.add_argument('-a', default=3, type=int, help='Aspect ratio bound')
    parser.add_argument('--method', choices=['gemm','syrk','trsm'])
    parser.add_argument('--opts', default='default', choices=['default','random','all'])
    parser.add_argument('-o', default='', help='Output file name')
    parser.add_argument('-s', default='0', type=int, help='Random Seed')
    parser.add_argument('-r', default='5', type=int, help='Repetitions')

    args = parser.parse_args()

    self.data_type = args.dt
    self.run_type = args.rt

    self.num_problems = args.n
    self.aspect_bound = args.a
    self.opts = args.opts
    self.method = args.method
    self.repetitions = args.r

    mem_bounds = args.m.split(',')
    self.mem_lb = int(float(mem_bounds[0])*1024**3)
    self.mem_ub = int(float(mem_bounds[1])*1024**3)

    self.filename = args.o
    self.seed = args.s

args = Args()
generator = None
if args.method == "gemm":
    generator = random_gemm
if args.method == "syrk":
    generator = random_syrk
if args.method == "trsm":
    generator = random_trsm

dt_size = 0 
if args.data_type == "double":
    dt_size = 8
elif args.data_type == "float":
    dt_size = 4
else:
    raise Exception(f"Bad data type {args.data_type}")

keys = generate(generator, args.num_problems, 
                args.mem_lb//dt_size, args.mem_ub//dt_size, 
                args.aspect_bound, opts=args.opts,
                seed=args.seed)

output_json = {}
output_json["keywords"] = {}
output_json["keywords"]["method"] = args.method
output_json["keywords"]["data_type"] = args.data_type
output_json["keywords"]["run_type"] = args.run_type
output_json["keywords"]["repetitions"] = args.repetitions

output_json["metadata"] = {}
output_json["metadata"]["seed"] = args.seed
output_json["metadata"]["mem_lb"] = args.mem_lb
output_json["metadata"]["mem_ub"] = args.mem_ub
output_json["metadata"]["num_problems"] = args.num_problems
output_json["metadata"]["aspect_bound"] = args.aspect_bound
output_json["metadata"]["opts"] = args.opts

output_json["problems"] = [key.json() for key in keys]

if args.filename != "":
    with open(args.filename,'w') as file:
        json.dump(output_json, file, indent=2)
else:
    print(json.dumps(output_json, indent=2))
#print(output_json)

# EXAMPLE: python3 scripts/input.py -n 10 -r 5 -m 0.1,1 --rt autotune--dt double -a 2 --method gemm --opts random

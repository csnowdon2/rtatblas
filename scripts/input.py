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
                 transA = Trans_Opt(), 
                 transB = Trans_Opt()):
        self.m = m
        self.k = k
        self.n = n
        self.transA = transA
        self.transB = transB

    def randomize_opts(self):
        self.opA.randomize()
        self.opB.randomize()

    def flop_count(self):
        return 2*self.m*self.k*self.n

    def memory_footprint(self):
        return self.m*self.k + self.k*self.n + self.m*self.n

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
                 trans = Trans_Opt(), 
                 uplo = Uplo_Opt()):
        self.n = n
        self.k = k
        self.trans = trans
        self.uplo = uplo

    def randomize_opts(self):
        self.trans.randomize()
        self.uplo.randomize()

    def flop_count(self):
        return k*n*(n+1)

    def memory_footprint(self):
        return self.n*self.n + self.k*self.n

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
        return int((mem/(1+self.aspect)))**(1/2))

class TRSM_Key:
    def __init__(self, m, n, 
                 side = Side_Opt(), 
                 uplo = Uplo_Opt(), 
                 trans = Trans_Opt(), 
                 diag = Diag_Opt()):
        self.m = m
        self.n = n
        self.side = side
        self.uplo = uplo
        self.trans = trans
        self.diag = diag

    def randomize_opts(self):
        self.side.randomize()
        self.uplo.randomize()
        self.trans.randomize()
        self.diag.randomize()

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
        return int((mem/(1+self.aspect)))**(1/2))



opt = Uplo_Opt("Upper")
print(enumerate(Uplo_Opt))

import math
import numpy as np
import pandas as pd
from funcoesTermosol import *

E = 200e9
A = 2e-2
L = 2
P = 50e3
nodes = 11
l = L/10


# substituir list -> matriz
c = []
for i in range(0, (nodes-1)):
    m = np.zeros((11,1))
    m[i][0] = -1
    m[i+1][0] = 1
    c.append(m)
c

S = (E*A)/l

K = []
for i in range(0, (nodes-1)):
    K.append((c[i] * (np.transpose(c[i]))) * S)

K_global = 0
for i in range(0, (nodes-1)):
    K_global += K[i]

f = np.zeros((10,1))
f[9][0] = 50e3

K_del = np.delete(K_global, 0, 0)
K_del = np.delete(K_del, 0, 1)
K_del.shape

x = np.linalg.solve(K_del, f)
u = np.zeros((11,1))
u[1:] = x

[nn,N,nm,Inc,nc,F,nr,R] = importa('entrada.xlsx')

# def calc_sin_cos(x2, x1, y2, y1):
#   sin = (y2 - y1)/L
#   cos = (x2 - x1)/L
#   return sin, cos

# Ke = np.zeros((4, 4))
# Ke = 

plota(N,Inc)
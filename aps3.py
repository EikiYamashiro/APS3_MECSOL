import math
import numpy as np
import pandas as pd
from funcoesTermosol import *

# Modulo de Elasticidade Longitudinal (GPa)
E = 200e9

# Área da seção transversal (m²)
A = 2e-2

# Comprimento da barra (m)
L = 2

# Esforços externos (N)
P = 50e3

# Número de nós  
nodes = 11

# Tamanho de cada elemento
l = L/10

# Leitura da colunda de entradas (Arquivo excel)
[nn,N,nm,Inc,nc,F,nr,R] = importa('entrada.xlsx')
# nn: número de nós
# N: matriz dos nos
# nm: número de membros
# Inc: matriz de incidencia
# nc: numero de cargas
# F: vetor carregamento
# nr: número de restrições
# R: vetor de restricoes


def calc_sin_cos(x2, x1, y2, y1):
  s = (y2 - y1)/L
  c = (x2 - x1)/L
  return s, c
# Hardcoded, mudar dps
C_t = [[-1, 0, -1], [1, -1, 0], [0, 1, 1]]
C_t = np.array([np.array(xi) for xi in C_t])
C = np.transpose(C_t)
M = N.dot(C_t)
K_e = []
for element in range(0,nm-1):
  tem_zero = False
  for i in M[element]:
    if i == 0:
      tem_zero= True

  if tem_zero:
    S = 0
  else:
    S = (E*A/L) * (np.matmul(M[element], np.transpose(M[element])))/np.absolute(M[element])**2
  K_e.append((C[element] * (C_t[element])) * S)

# [array([[-1.,  0.,  0.],
#         [ 1.,  0.,  0.],
#         [ 0.,  0.,  0.]]),
#  array([[ 0.,  0.,  0.],
#         [-1.,  0.,  0.],
#         [ 1.,  0.,  0.]])]


# 4 - MATRIZES DE RIGIDEZ
# 5 - MARIZ DE RIGIDEZ GLOBAL ( -> N)
# 6 - VETOR GLOBAL DE FORÇAS CONCENTRADAS #



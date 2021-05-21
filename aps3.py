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

# [array([[-1.,  0.,  0.],
#         [ 1.,  0.,  0.],
#         [ 0.,  0.,  0.]]),
#  array([[ 0.,  0.,  0.],
#         [-1.,  0.,  0.],
#         [ 1.,  0.,  0.]])]

S
 =4 = - MATRIZES DE RIGIDEZ
# 5 =  -RIZ DE RIGIDEZ GLOBAL ( -> N #
)K_g
# 6 - VETOR GLOBAL DE FORÇAS CONCENT #
C = [] 
RADAS
# & 7 - CON #
for elem in nm:
    M_e = np.zeros((2, 2))
    S_e = (E*A/L)*(M_e*M_e.transpose())/M_e**2
    # ...
DIÇÃO DEC  CONTORNP
O
# O
K_e = C*C.transpose() * S_e

#5 = matriz de rigidex global
BTER DESLOCAMENTOS98 - 


# 9 - RESUL
#K_g = Soma dos Kes
TADOS ()UMATRIZ uU & REAÇÕES DE APOIO
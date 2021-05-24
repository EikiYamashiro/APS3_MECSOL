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

#[nn,N,nm,Inc,nc,F,nr,R] = importa('entrada.xlsx')
[nn,N,nm,Inc,nc,F,nr,R] = importa('aps3_entrada.xlsx')
plota(N, Inc)
# nn: número de nós
# N: matriz dos nos
# nm: número de membros
# Inc: matriz de incidencia
# nc: numero de cargas
# F: vetor carregamento
# nr: número de restrições
# R: vetor de restricoes

def vet_conec(ne):
  conec_array = np.array(nm*[0])
  n1 = Inc[ne-1, 0]
  n2 = Inc[ne-1, 1]
  conec_array[n1 - 1] = -1
  conec_array[n2 - 1] = 1
  return conec_array

def mat_conec():
  mat_conec = np.zeros((nn, nm))
  for i in range(0, nn - 1):
    mat_conec[i] = vet_conec(i)
  return mat_conec


def calc_sin_cos(x2, x1, y2, y1):
  s = (y2 - y1)/L
  c = (x2 - x1)/L
  return s, c

def calc_tensao_G(c, s, mat_u):
  return np.matmul((E/l*[-c, -s, c, s]), mat_u)

def calc_deform_esp_G(c, s, mat_u):
  return np.matmul((1/l*[-c, -s, c, s]), mat_u)

def calc_mat_rig_G(c, s):
  mat_rig = (E*A/l)*np.array([[c**2,c*s,-c**2,-c*s],[c*s,s**2,-c*s,-s**2],[-c**2,-c*s,c**2,c*s],[-c*s,-s**2,c*s,s**2]])
  return mat_rig

def calc_mat_rig_L(c_e, S_e):
  K_e = np.kron((np.matmul(c_e, c_e.T)), S_e)
  return K_e
      
def calc_Se(m_e):
      S_e = (E*A/L)*(np.matmul(m_e, m_e.T)/(np.linalg.det(m_e))**2)
      return S_e

# Hardcoded, mudar dps
#C_t = [[-1, 0, -1], [1, -1, 0], [0, 1, 1]]
#C = C_t.transpose()

for element in range(0, nm - 1):
  c_e = mat_conec()
  m_e = np.matmul(N, c_e.T)
  S_e = calc_Se(m_e)
  list_K_e = []
  K_e = calc_mat_rig_L(c_e, S_e)
  list_K_e.append(K_e)

print(list_K_e)


# 4 - MATRIZES DE RIGIDEZ
# 5 - MARIZ DE RIGIDEZ GLOBAL ( -> N)
# 6 - VETOR GLOBAL DE FORÇAS CONCENTRADAS #



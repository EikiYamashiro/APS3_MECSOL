import math
import numpy as np
import pprint as pp
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
  n1 = int(Inc[ne-1, 0])
  n2 = int(Inc[ne-1, 1])
  conec_array[n1 - 1] = -1
  conec_array[n2 - 1] = 1
  return conec_array

def mat_conec():
  mat = np.zeros((nn, nm))
  for i in range(0, nn - 1):
    mat[i] = vet_conec(i)
  return mat


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
  S_e = (E*A/L)*(np.matmul(m_e.T, m_e)/(np.linalg.norm(m_e))**2)
  if (np.linalg.norm(m_e))**2 == 0:
    S_e = 0
  return S_e

C = mat_conec()
C = C.T
M = np.matmul(N, C.T)
# M deve ser 2X8
print(f'M deve ser (2, {nm}) e esta {M.shape}')

for element in range(0, nm - 1):
  #m_e = M.T[:, element]
  #c_e = C[:, element]
  m_e = M[:, element]
  m_e = np.array(m_e)[np.newaxis]
  c_e = C[element, :]
  S_e = calc_Se(m_e)
  K_e = calc_mat_rig_L(c_e, S_e)
  print(K_e)
  #pp.pprint(f'Shape do c_e: {c_e.shape}, shape do S_e: {S_e.shape}')
  K = np.zeros((c_e.shape))

# geraSaida(trelica, )
# 4 - MATRIZES DE RIGIDEZ
# 5 - MARIZ DE RIGIDEZ GLOBAL ( -> N)
# 6 - VETOR GLOBAL DE FORÇAS CONCENTRADAS #



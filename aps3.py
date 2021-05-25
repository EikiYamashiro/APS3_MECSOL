import math
import numpy as np
import pprint as pp
from funcoesTermosol import *

# Modulo de Elasticidade Longitudinal (GPa)
E = 210e9

# Área da seção transversal (m²)
A = 2e-4

# Comprimento da barra (m)
L = 1
# Esforços externos (N)
P = 50e3
# Tamanho de cada elemento
l = L/10

# Leitura da coluna de entradas (Arquivo excel) 
# [nn,N,nm,Inc,nc,F,nr,R] = importa('entrada.xlsx')
[nn,N,nm,Inc,nc,F,nr,R] = importa('entrada.xlsx')

print(N)
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
  conec_array = np.array(nn*[0])
  n1 = int(Inc[ne, 0])
  n2 = int(Inc[ne, 1])
  conec_array[n1 - 1] = -1
  conec_array[n2 - 1] = 1
  return conec_array

def mat_conec():
  mat = []
  for i in range(nm):
    mat.append(vet_conec(i))
  return np.array(mat)

def calc_sin_cos(x2, x1, y2, y1):
  s = (y2 - y1)/l
  c = (x2 - x1)/l
  return s, c

def calc_l(element):
  x1 = N[0][int(Inc[:,0][element]) - 1]
  x2 = N[0][int(Inc[:, 1][element]) - 1]
  y1 = N[1][int(Inc[:, 0][element]) - 1]
  y2 = N[1][int(Inc[:, 1][element]) - 1]
  l = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  return l

def calc_tensao_G(c, s, mat_u):
  return np.matmul((E/l*[-c, -s, c, s]), mat_u)

def calc_deform_esp_G(c, s, mat_u):
  return np.matmul((1/l*[-c, -s, c, s]), mat_u)

def calc_mat_rig_G(c, s, l):
  mat_rig = (E*A/l)*np.array([[c**2,c*s,-c**2,-c*s],[c*s,s**2,-c*s,-s**2],[-c**2,-c*s,c**2,c*s],[-c*s,-s**2,c*s,s**2]])
  return mat_rig

def calc_Ke(c_e, S_e):
  print(c_e)
  print(c_e.T)
  mul_C = np.matmul(c_e.T, c_e)
  print(mul_C)
  K_e = (np.kron(mul_C, S_e))
  return K_e
      
def calc_Se(m_e, l):
  S_e = (E*A/l)*(np.matmul(m_e.T, m_e)/(np.linalg.norm(m_e))**2)
  if (np.linalg.norm(m_e))**2 == 0:
    S_e = np.zeros((2, 2))
  return S_e

C = mat_conec()
C_Transposto = C.T
print(f'C:\n {C}')
print(f'C Transposto:\n {C_Transposto}')
print("__________________")
M = np.matmul(N, C_Transposto)

print(f'Matriz M:\n {M}')
print("__________________")

shape_Kg = 2*nn
K_g = np.zeros((shape_Kg, shape_Kg))
for element in range(0, nm):
  #####
  m_e = (M[:, element])[np.newaxis]
  #print(f'm_e:\n {m_e}')
  #print("__________________")
  #####
  m_e_transposed = m_e.T
  #print(f'm_e_transposed:\n {m_e.T}')
  #print("__________________")
  #####
  c_e = (C[:, element])[np.newaxis]
  print(f'c_e: {c_e}')
  S_e = calc_Se(m_e, calc_l(element))
  #####
  print(f'Matriz S{element+1}:\n {S_e}')
  print("__________________")
  #####
  K_e = calc_Ke(c_e, S_e)
  print(f'Matriz K_e[{element}]:\n {K_e}')
  print("__________________")
  
  K_g += K_e

print(f'Matriz K_g:\n {K_g}')
print("__________________")
# geraSaida(trelica, )
# 4 - MATRIZES DE RIGIDEZ
# 5 - MARIZ DE RIGIDEZ GLOBAL ( -> N)
# 6 - VETOR GLOBAL DE FORÇAS CONCENTRADAS
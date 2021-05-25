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
  mul_C = np.matmul(c_e.T, c_e)
  K_e = (np.kron(mul_C, S_e))
  return K_e
      
def calc_Se(m_e, l):
  S_e = (E*A/l)*(np.matmul(m_e.T, m_e)/(np.linalg.norm(m_e))**2)
  if (np.linalg.norm(m_e))**2 == 0:
    S_e = np.zeros((2, 2))
  return S_e

def sol_Jacobi(ite, tol, K, F):
  x = np.zeros((1, 3)).T
  D = np.diag(K)
  R = A - np.diagflat(D)
  for i in range(ite):
    x = (F - np.matmul(F, x[i]))/D
  return x

def cond_contorno(Kg):
  print((R).astype(int))
  try:
    K_g = np.delete(Kg, R.astype(int), 0)
    K_g_new = np.delete(K_g, R.astype(int), 1)
    F_new = np.delete(F, R.astype(int), 0)
    print(f'Kg com condicoes de contorno:\n{K_g_new}')
    print(f'F com condicoes de contorno:\n{F_new}')
  except ValueError:
      print( "Not Defined")
  return K_g_new, F_new

def desloc_nodais(K_g, F):
  u = np.linalg.solve(K_g, F)
  return u

def reac_apoio(K_g, u):
  R = np.matmul(K_g, u)
  return R

# def sol_Gauss(ite, tol, K, F):
#     return u

C = mat_conec()
#C_Transposto = np.array([[-1, 0, -1],[1, -1, 0],[0, 1, 1]])
#C = C_Transposto.T
C_Transposto = C.T
print(f'C:\n {C}')
print(f'C Transposto:\n {C_Transposto}')
print("__________________")
M = np.matmul(N, C_Transposto)

#print(f'Matriz M:\n {M}')
#print("__________________")

shape_Kg = 2*nn
K_g = np.zeros((shape_Kg, shape_Kg))
# sol_Jacobi(1, 1, 1, 1)
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
  c_e = (C[element, :])[np.newaxis]
  c_e_transposed = c_e.T
  #print(f'c_e: {c_e}')
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

K_g_new, F_new = cond_contorno(K_g)
u = desloc_nodais(K_g_new, F_new)
R = reac_apoio(K_g_new, u)
print(u)
print(R)

# geraSaida(trelica, )
# 4 - MATRIZES DE RIGIDEZ
# 5 - MARIZ DE RIGIDEZ GLOBAL ( -> N)
# 6 - VETOR GLOBAL DE FORÇAS CONCENTRADAS
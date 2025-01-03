import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('../')
from molecular_vib import vibration_space

# case study, a 2D square molecuar of symmetry C4v 
molecular = {"na":4, 
             "position":np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]),
             "type": [0,0,0,0]
            }
square = vibration_space(molecular)
from group import MatrixGroup
from spherical import R_Z 

#build C4v in x,y,z spapce 
Id = np.eye(3)
th = np.pi/2
Rz90 = R_Z(np.pi/2)
sigma_d = np.array([[0,1,0],[1,0,0],[0,0,1]])
C4v = MatrixGroup(generator = [Id,Rz90,sigma_d])
print(f'|C4v| = {C4v.nG}')
if C4v.nG!=8:
   raise ValueError('# of elements in C4v is inconsistent')
C4v.constructMultiplicationTable()
C4v.conjugacy_class()
if C4v.nClass!=5:
   raise ValueError('# of conjugacy classes in C4v is inconsistent')
print('C4v ConjClass = ',C4v.ConjClass)
#
from BDS_solver import character_solver
ChiSolver = character_solver(C4v.MultilicationTable, C4v.ConjClass)
chi_table = ChiSolver.solve()
print('Character table of C4v = ')
for i in range(C4v.nClass):
   print(" ".join(f"{x:10.2f}" for x in chi_table[:,i]))  # Format numbers to 2 decimal places
C4v.decompose(C4v.G, chi_table)
C4v.build_explicit_IRmatrix(chi_table)

#
square.RepOfGroup(C4v)
Multiplicity = C4v.decompose(square.G, chi_table)
print('Multiplicity = ',Multiplicity)
basis = C4v.basis_function_break_multiplicity(square.G, chi_table, excluded_space=square.acoustic_translation_mode_2D())

for i in range(len(basis)):
   print(basis[f'{i}-th basis'])
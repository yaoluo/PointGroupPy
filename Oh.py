# case study of Oh point group 
import numpy as np 
from group import MatrixGroup
from spherical import R_X, R_Y, R_Z 

from molecular_vib import vibration_space

# case study, a 3D Octahedral molecuar of symmetry Oh 
molecular = {"na":7, 
             "position":np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),
             "type": [1,0,0,0,0,0,0]
            }
Octahedral  = vibration_space(molecular)
#Oh 
Id = np.eye(3)
th = np.pi/2
Rx90 = R_X(np.pi/2)
Ry90 = R_Y(np.pi/2)
Rz90 = R_Z(np.pi/2)
sigma_d = np.array([[0,1,0],[1,0,0],[0,0,1]])
Oh = MatrixGroup(generator = [Id,Rx90,Ry90,Rz90,sigma_d])
print(f'|Oh| = {Oh.nG}')
if Oh.nG!=48:
   raise ValueError('# of elements in Oh is inconsistent')
Oh.constructMultiplicationTable()
Oh.conjugacy_class()
if Oh.nClass!=10:
   raise ValueError('# of conjugacy classes in Oh is inconsistent')
print('Oh ConjClass = ',Oh.ConjClass)
#
from BDS import character_solver
ChiSolver = character_solver(Oh.MultilicationTable, Oh.ConjClass)
chi_table = ChiSolver.solve()
print('Character table of Oh = ')
for i in range(Oh.nClass):
   print(" ".join(f"{x:10.2f}" for x in chi_table[:,i]))  # Format numbers to 2 decimal places
#apply C4v to the polynomial of (x,y)
# build the matrix representtaion of C4v on linear function of (x,y)
# Here, it is coincidently the same as the original C4v, for cubic or quadratic, it will be different. 
Oh.decompose(Oh.G, chi_table)
Oh.build_explicit_IRmatrix(chi_table)

#
Octahedral .RepOfGroup(Oh)
Multiplicity = Oh.decompose(Octahedral .G, chi_table)
print('Multiplicity = ',Multiplicity)
basis = Oh.basis_function_break_multiplicity(Octahedral .G, chi_table, excluded_space=Octahedral .acoustic_translation_mode())
for i in range(len(basis)):
   print(basis[f'{i}-th subspace'])
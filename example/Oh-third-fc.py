# case study of Oh point group 
import numpy as np 
import sys 
sys.path.append('../')
from group import MatrixGroup
from spherical import R_X, R_Y, R_Z 
from molecular_vib import vibration_space


#[1] Oh: build character table and IR matrix rep  
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
from BDS_solver import character_solver
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

# [2]
# case study, a 3D Octahedral molecuar of symmetry Oh 
molecular = {"na":7, 
             "position":np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),
             "type": [1,0,0,0,0,0,0]
            }
Octahedral  = vibration_space(molecular)
Octahedral.RepOfGroup(Oh)
Multiplicity = Oh.decompose(Octahedral .G, chi_table)
print('Multiplicity = ',Multiplicity)
V_Ia = Oh.basis_function_break_multiplicity(Octahedral .G, chi_table, excluded_space=Octahedral .acoustic_translation_mode())
print('# of V_Ia (# of invariant subspace) = ',len(V_Ia)) 
n_Ia = len(V_Ia)

# [3] build the ijk -> x CG coefficient for symmetric tensor product for A1g 
from CGC_symmetricTP import symmetricTensorProduct
CGC_IJK = symmetricTensorProduct(Oh.D_IR)

n_parameter = 0
for i in range(n_Ia):
   for j in range(i+1):
      for k in range(j+1):
         #print(i,j,k)
         I = V_Ia[f'{i}-th subspace']['IR index']
         J = V_Ia[f'{j}-th subspace']['IR index']
         K = V_Ia[f'{k}-th subspace']['IR index']

         basis, repG = CGC_IJK.build_ijk_rep(I,J,K, index_subspace=[i,j,k])
         #print(I,J,K,i,j,k )
         CGC = Oh.basis_function_for_oneIR(
                              repG, 
                              Oh.D_IR[0]  
                              )
         if len(CGC)!=0:
            print(f'{i} {j} {k}-> {len(CGC)} x 0')
            n_parameter = n_parameter + len(CGC)
print(f'n_parameter = {n_parameter}')
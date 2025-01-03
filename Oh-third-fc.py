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

# [2] build the ijk -> x CG coefficient for symmetric tensor product for A1g 
from CGC_symmetricTP import symmetricTensorProduct
CGC_IJK = symmetricTensorProduct(Oh.D_IR)

#debug 
n3 = 0 
da1g3 = 0 
for i in range(9):
   for j in range(9):
      for k in range(9):
         basis, repG = CGC_IJK.build_ijk_rep(i,j,k)
         CGC = Oh.basis_function_for_oneIR(
                              repG, 
                              Oh.D_IR[0]  # this is the A1g rep 
                              )
         if i!=j and k==0 and len(CGC)!=0:
            raise ValueError(f'error for {i}!={j}')
         if i==j and k==0 and len(CGC)!=1:
            raise ValueError(f'error for {i}={j}')
         if len(CGC)!=0:
            print(f'{i} {j} {k}-> {len(CGC)} x 0')
            n3 += 1 
            da1g3 += len(CGC)

#debug 
n4 = 0 
da1g4 = 0 
for i in range(9):
   for j in range(9):
      for k in range(9):
         for l in range(9):
            basis, repG = CGC_IJK.build_ijkl_rep(i,j,k,l)
            CGC = Oh.basis_function_for_oneIR(
                                 repG, 
                                 Oh.D_IR[0]  # this is the A1g rep 
                                 )
            if len(CGC)!=0:
               print(f'{i} {j} {k} {l}-> {len(CGC)} x 0')
               n4 += 1 
               da1g4 += len(CGC)
print(f'# triplet = {n3}, # of quad = {n4}')
print(f'triplet, a1g dim = {da1g3}; quad, a1g dim = {da1g4}')




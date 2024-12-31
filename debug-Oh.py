# case study of Oh point group 
import numpy as np 
from group import MatrixGroup
from spherical import R_X, R_Y, R_Z 
from polynormial import polynomial_space

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
   raise ValueError('# of elements in C4v is inconsistent')
Oh.constructMultiplicationTable()
Oh.conjugacy_class()
if Oh.nClass!=10:
   raise ValueError('# of conjugacy classes in C4v is inconsistent')
print('C4v ConjClass = ',Oh.ConjClass)
#
from BDS import character_solver
ChiSolver = character_solver(Oh.Table, Oh.ConjClass)
chi_table = ChiSolver.solve()
chi_table[np.abs(chi_table) < 1e-4] = 0
print('Character table of Oh = ')
for i in range(Oh.nClass):
   print(" ".join(f"{x:10.2f}" for x in chi_table[:,i]))  # Format numbers to 2 decimal places
#apply C4v to the polynomial of (x,y)
# build the matrix representtaion of C4v on linear function of (x,y)
# Here, it is coincidently the same as the original C4v, for cubic or quadratic, it will be different. 
Oh.decompose(Oh.G, chi_table)
Oh.build_explicit_IRmatrix(chi_table)
Ls = [2] 
xl = [ polynomial_space( l = l) for l in Ls]
Multiplicity = []
for i in range(len(xl)):
   xl[i].RepOfGroup(Oh)
   #Multiplicity.append(Oh.decompose(xl[i].polyG, chi_table))
print('sigma_d = \n',Oh.G[4])
xl[0].print_poly(np.ones(6))
print('sigma_d = \n',xl[0].polyG[4])
print('sigma_d_ = \n',Oh.D_IR[6][4])
print('char(sigma_d) = \n',chi_table[Oh.class_index[4],6])
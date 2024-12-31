import numpy as np
#from group import MatrixGroup
class polynomial_space:

   def __init__(self, l):
      self.dim = 3 
      self.l = l 
      if self.dim ==3:
         self.nbasis = int((self.l+2) * (self.l+1) /2)
      print(f'nbasis = {self.nbasis}') 
      lxlylz = [] 
      for lz in range(self.l + 1):
        for ly in range(self.l-lz+1):
            lx = self.l - lz - ly 
            lxlylz.append([lx,ly,lz])
            #print(lx,ly,lz)
      self.lxlylz_set = np.asarray(lxlylz).astype(np.int32)

      return  
   def RepOfR(self, R):
      # R is an element of O(3)
      # compute the representation of R in the polynomial space 
      Dpoly = np.zeros([self.nbasis, self.nbasis])
      detR = abs(np.linalg.det(R))
      if abs(detR-1)>1e-6:
         raise ValueError('not O(3) matrix in func RepOfR of class polynomial_space')
      
      # check if every row only has one entry of 1 for R, if so it is very simple 
      if abs( np.sum(np.abs(R))-3 )<1e-10:
         Rinv = np.linalg.inv(R)
         mapping = np.zeros(3).astype(np.int32)
         for i in range(3):
            for j in range(3):
               if abs(R[i,j])>1e-10:
                  mapping[i] = j
                 
         for i_lxlylz, lxlylz in enumerate(self.lxlylz_set):
            Rlxlylz = np.zeros(3).astype(np.int32)
            factor = 1  
            for i in range(3):
               Rlxlylz[mapping[i]] = lxlylz[i] 
               factor = factor * (R[i,mapping[i]])**(lxlylz[i])
            i_Rlxlylz = np.argmin(np.einsum('ia->i',np.abs(self.lxlylz_set - Rlxlylz)))
            Dpoly[i_lxlylz,i_Rlxlylz] = factor 
      else:
         raise ValueError('Rotation matrix with elemement |R_{ij}| !=1, not surpported. ')
      return Dpoly 
   
   def RepOfGroup(self, group):
      
      G = np.zeros([group.nG, self.nbasis, self.nbasis])
      for ig, g in enumerate(group.G):
         G[ig] = self.RepOfR(g)
      
      self.polyG = G

      return 
   
   def print_poly(self, a):
      formula = ""
      for i,lxlylz in enumerate(self.lxlylz_set):
         if abs(a[i])>1e-5:
            
            if formula !="":
               if a[i]>0:
                  formula = formula+f' + {a[i]:2.4f} '
               else:
                  formula = formula+f' {a[i]:2.4f} '
            else:
               if a[i]>0:
                  formula = formula+f'{a[i]:2.4f} '
               else:
                  formula = formula+f'{a[i]:2.4f} '

            if lxlylz[0]!=0:
               if lxlylz[0] == 1:
                  formula = formula + f'x'
               else:
                  formula = formula + f'x^{lxlylz[0]}'
            if lxlylz[1]!=0:
               if lxlylz[1] == 1:
                  formula = formula + f'y'
               else:
                  formula = formula + f'y^{lxlylz[1]}'
            if lxlylz[2]!=0:
               if lxlylz[2] == 1:
                  formula = formula + f'z'
               else:
                  formula = formula + f'z^{lxlylz[2]}'
      print(formula)
      return 
if __name__ == '__main__':
   x1 = polynomial_space( l = 1)
   x2 = polynomial_space( l = 2)
   from spherical import R_X, R_Y, R_Z 

   Rx90 = R_X(np.pi/2); Rx90[np.abs(Rx90)<1e-10] = 0;
   print('Rx90 = \n', Rx90)
   D = x1.RepOfR(Rx90)
   print('Rx90 in span(x, y, z) = \n', D)

   Rx90 = R_Y(np.pi/2); Rx90[np.abs(Rx90)<1e-10] = 0;
   print('Rx90 = \n', Rx90)
   D = x1.RepOfR(Rx90)
   print('Rx90 in span(x, y, z) = \n', D)

   Rx90 = R_Z(np.pi/2); Rx90[np.abs(Rx90)<1e-10] = 0;
   print('Rx90 = \n', Rx90)
   D = x1.RepOfR(Rx90)
   print('Rx90 in span(x, y, z) = \n', D)

   # test x2 
   Rx90 = R_Z(np.pi/2); Rx90[np.abs(Rx90)<1e-10] = 0;
   print('Rx90 = \n', Rx90)
   D = x2.RepOfR(Rx90)
   
   print('Rx90 in span(x^2, xy, y^2, xz, yz, z^2) = \n ', D)
   #for i in range(x2.nbasis):
   #   print(D[:,i])








   

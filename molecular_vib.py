import numpy as np
import matplotlib.pyplot as plt

class vibration_space:

   def __init__(self, molecular):

      #molecular is a dict {'na','position', 'type'}
      self.molecular = molecular 
      return  
   
   def RepOfR(self, R):
      # R is an element of O(3)
      # compute the representation of R in the molecular vibration 
      detR = abs(np.linalg.det(R))
      if abs(detR-1)>1e-6:
         raise ValueError('not O(3) matrix in func RepOfR of class polynomial_space')
      Dg_mol = np.zeros([self.molecular['na'],self.molecular['na'],3,3])
      #pos = np.zeros_like(self.molecular['position'])
      for ia in range(self.molecular['na']):
         pos = R@self.molecular['position'][ia]
         residal = np.sum( (self.molecular['position'] - pos)**2, axis=-1 )
         iap = np.argmin(residal)
         if residal[iap]>1e-10:
            #print(pos)
            raise ValueError('rotated atom not in the molecular, check your symmetry or the molecular')
         if self.molecular['type'][ia]!=self.molecular['type'][iap]:
            raise ValueError('rotated atom type inconsistency in the molecular')
         Dg_mol[iap,ia] = R   
      Dg_mol = np.transpose(Dg_mol,(0,2,1,3)).reshape( self.molecular['na']*3, self.molecular['na']*3)
      return Dg_mol 
   
   def RepOfGroup(self, group):
      G = np.zeros([group.nG, self.molecular['na']*3,  self.molecular['na']*3])
      for ig, g in enumerate(group.G):
         G[ig] = self.RepOfR(g)
      self.G = G
      return 
   
   def acoustic_translation_mode(self):
      v = np.zeros([3,self.molecular['na'],3])
      for ix in range(3):
         for ia in range(self.molecular['na']):
            v[ix,ia,ix] = 1
         v[ix] = v[ix] / np.sum(v[ix]**2)**0.5
      return v.reshape(3,-1)
   
   def acoustic_translation_mode_2D(self):
      v = np.zeros([2+self.molecular['na'],self.molecular['na'],3])
      for ix in range(2):
         for ia in range(self.molecular['na']):
            v[ix,ia,ix] = 1
         v[ix] = v[ix] / np.sum(v[ix]**2)**0.5
      #z modes fixed 
      for ix in range(self.molecular['na']):
         v[ix+2,ix,2] = 1
      return v.reshape(2+self.molecular['na'], -1)
   
   def visualize_mode_2D(self, a):
      ph = a.reshape(-1,3)
      for ia in range(self.molecular['na']):
         pos = self.molecular['position'][ia]
         vec = ph[ia]
         # Plot the vector using plt.quiver
         plt.quiver(
             pos[0], pos[1],  # Origin x and y
             vec[0], vec[1],  # Vector x and y
             angles='xy', scale_units='xy', scale=1, color='r', label='Vector'
         )
         plt.scatter([pos[0]],[pos[1]],c='k')
      
      plt.axis('equal')
      plt.xlim([-2,2])
      plt.ylim([-2,2])
      plt.show()
      return 

   def visualize_mode_3D(self, a, clist=['r','g','b','k']):
      ph = a.reshape(-1,3)
      
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      for ia in range(self.molecular['na']):
         origin = self.molecular['position'][ia]
         vector = ph[ia]
         # Plot the vector using plt.quiver

         # Plot the vector using quiver
         ax.quiver(
             origin[0], origin[1], origin[2],  # Origin x, y, z
             vector[0], vector[1], vector[2],  # Vector x, y, z components
             color=clist[self.molecular['type'][ia]], label='Vector'
         )
         ax.scatter(
            origin[0], origin[1], origin[2],  # Coordinates of the point
            color=clist[self.molecular['type'][ia]], s=50, label='Point', marker='o'  # Customize color, size, and marker
         )

      plt.axis('equal')
      plt.xlim([-2,2])
      plt.ylim([-2,2])
      #plt.show()
      return ax 
   
if __name__ == '__main__':
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
   from BDS import character_solver
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

   basis = C4v.basis_function(square.G, chi_table, excluded_space=square.acoustic_translation_mode_2D())
   #basis = C4v.basis_function(square.G, chi_table)


   dim = 0 
   for ir in range(C4v.nClass):
      if len(basis[ir])==0:
         print(f'{ir+1}-IR, chi = {chi_table[:,ir]}, multiplicity = 0, space dim = {0}') 
      else:
         print(f'{ir+1}-IR, chi = {chi_table[:,ir]}, multiplicity = {int(len(basis[ir][0])/int(chi_table[0,ir]+1e-4))}, space dim = {len(basis[ir][0])}') 
      dim = dim + len(basis[ir][0])
   print('# of vibration modes = ', dim)

   #visualize 2D modes 
   square.visualize_mode_2D(basis[4][:,0])
   square.visualize_mode_2D(basis[4][:,1])

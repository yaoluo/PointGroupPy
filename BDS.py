# Burnside–Dixon–Schneider algorithm 
import numpy as np 

def common_eigenvector_solver(M):
   # M should be size of [n,n,n]
   # either two of M commute 
   # starting from \sum_{i} a_i M_i, a_i = random number 
   n,_,_ = M.shape 
   f = np.random.random(n)
   fM = np.einsum('i,iab->ab',f,M)
   evals, evec = np.linalg.eig(fM)
   # detect degenerate space 
   for i in range(n):
      e_i = evals[i]
      if np.sum( np.abs(evals - e_i)<1e-10 )>1:
         #found degeneracy 
         print('found degenercy')
   return evec

class character_solver:

   def __init__(self, multiplication_table, ConjClass ):
      '''
      This solver for character table only needs the abtract multiplication_table, independent of the representation 
      '''
      self.multiTable = multiplication_table
      self.ConjClass = ConjClass
      self.nClass = len(ConjClass)

      
      return 
   def build_Mi(self):

      nG = np.max(self.multiTable)+1
      index_Class = np.zeros(nG).astype(np.int32)
      for ic, c in enumerate(self.ConjClass):
         for ig in c:
            index_Class[ig] = ic
      print('index_Class = ', index_Class)
      nClass = self.nClass 

      Mi = np.zeros([nClass, nClass, nClass])
      for i, Ci in enumerate(self.ConjClass):
         for j, Cj in enumerate(self.ConjClass):
            for ig in Ci:
               for jg in Cj:
                  k_Class = index_Class[self.multiTable[ig,jg]]
                  Mi[i,j,k_Class] = Mi[i,j,k_Class] + 1.0 / len(self.ConjClass[k_Class])
         #print(f'{i} M = ',Mi[i])

      self.Mi = Mi

      return 
   
   def check_commutativity(self, Mi):
      for a in Mi:
         for b in Mi:
            if np.abs(a@b - b@a).max()> 1e-10:
               print(a)
               print(b)
               print(a@b - b@a)
               raise ValueError('Error: Mi Mj not commute!')
      return 
   
   def solve(self):
      nClass = self.nClass 
      nG = np.max(self.multiTable)+1
      self.build_Mi()
      self.check_commutativity(self.Mi)
      #find common eigen basis 
      chi = common_eigenvector_solver(self.Mi)
      
   
      sClass = np.array([len(self.ConjClass[i]) for i in range(nClass)])
      rank = np.zeros([nClass])
      for i in range(nClass):
         chi[:,i] = chi[:,i] / sClass
         chi[:,i] = chi[:,i] / np.sqrt(np.sum(chi[:,i]**2 * sClass )) * np.sqrt(nG)
         if (chi[0,i]<0):
            chi[:,i] = -chi[:,i]
         rank[i] = chi[0,i] * nClass * 2 + nClass - np.sum(chi[:,i])
      
      chi = chi[:,np.argsort(rank)]

      chi[np.abs(chi) < 1e-4] = 0
      return chi 
   
   

   

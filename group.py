import numpy as np

class MatrixRepresentation:
    def __init__(self, generator = []):
        """
        Initialize the matrix representation of a finite group.

        Args:
            group_elements (list): A list of group elements (e.g., strings or integers).
            matrices (dict): A dictionary mapping group elements to their corresponding matrices.
                             Example: { "e": np.eye(2), "g": np.array([[0, 1], [1, 0]]) }
            multiplication_rule (dict): A dictionary representing the group's multiplication table.
                                        Example: {("e", "g"): "g", ("g", "g"): "e"}.
        """
   
        if len(generator)!=0:
            self.generate_G(generator)
        return 

    def constructMultiplicationTable(self):
         nG = self.nG
         Table = np.zeros([nG, nG]).astype(np.int32)
         for i, gi in enumerate(self.G):
            G_prime = np.einsum('iab,bc->iac',self.G, gi)
            for j, gigj in enumerate(G_prime):
               #G[i] * G[j] = G[k]
               k = np.argmin(np.einsum('iab->i',(self.G - gigj)**2))
               if( np.sum((self.G[k] - gigj)**2) >1e-10):
                   raise ValueError('gi*G not equal G')
               Table[i,j] = k
         self.Table = Table  
         print(Table)
         # identify identity 
         for i in range(nG):
            if np.max(np.abs(Table[i]-np.arange(nG)))==0:
                self.E = i
         print(f'E = {self.E}')
         # construct invG 
         index_invG = np.zeros(nG).astype(np.int32)
         for i in range(nG):
            for j in range(nG):
                if Table[i,j] == self.E:
                    index_invG[i] = j
                    index_invG[j] = i
         self.index_invG = index_invG
         print(f'invG = {index_invG}')
    
    def generate_G(self, generators):
        self.G = np.asanyarray(generators)
        shape = self.G.shape 
        if (len(shape)!=3):
            raise ValueError('shape error, self.G.shape  not 3D')
        if (shape[1]!=shape[2]):
            raise ValueError('shape error, not square matrix rep')
        #print(self.G.shape )
        i = 0 
        while True:
            self.G, find_new = self.GxG() 
            i+=1
            #print(f'{i}-itr, |G| = {len(self.G)}, {find_new}')
            if (len(self.G)>100):
               raise ValueError('num of group elements > 100, indicates the generators are not closed')
            if find_new==False:
               break 
        #for i,g in enumerate(self.G):
            
        #   print(f'G[{i}] =  \n', g)
        self.nG, self.dim,_ = self.G.shape
          
    def GxG(self):
         Find_new = False 
         nG = len(self.G)
         G = [x for x in self.G]
         for i, gi in enumerate(self.G):
            G_prime = np.tensordot(self.G, gi,axes=([-1],[0]))
            for k, gk in enumerate(G_prime):
                collect = True  
                for j, gj in enumerate(G):
                    
                    if( np.sum((gj - gk)**2) <1e-10):
                        collect = False 
                if collect:
                    G.append(gk)
                    Find_new = True 
         return np.asarray(G), Find_new
    
    def conjugacy_class(self):
      nG = len(self.G)
      class_index = -np.ones(nG).astype(np.int32)
      ith_conjugacy = 0 
      for i in range(nG):
            if class_index[i]==-1:
               gi = self.G[i] 
               for j in range(nG):
                  #find G[k] = gi*gj 
                  k_gigj = self.Table[i,j]
                  i_invgj_gi_gj = self.Table[self.index_invG[j],k_gigj]
                  class_index[i_invgj_gi_gj] = ith_conjugacy
               ith_conjugacy += 1
      nClass = np.max(class_index)+1
      ConjClass = [[] for i in range(nClass)] 
      for i in range(nG):
          ConjClass[class_index[i]].append(i)
      self.ConjClass = ConjClass
      self.nClass = nClass
      print('class_index = ',class_index)
      return 
    
if __name__ == '__main__':

   for n in [4]:
      th = 2*np.pi/n
      Id = np.array([[1,0],[0,1]])
      R90 = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])
      Cn = MatrixRepresentation(generator = [Id,R90])
      print(f'|C{n}| = {Cn.nG}')
      if Cn.nG!=n:
          raise ValueError('# of elements in Cn is inconsistent')
      Cn.constructMultiplicationTable()
   #C4v 
   th = np.pi/2
   R90 = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])
   sigma_v = np.array([[-1,0],[0,1]])
   sigma_d = np.array([[0,1],[1,0]])
   C4v = MatrixRepresentation(generator = [Id,R90,sigma_v,sigma_d])
   print(f'|C4v| = {C4v.nG}')
   if C4v.nG!=8:
      raise ValueError('# of elements in C4v is inconsistent')
   C4v.constructMultiplicationTable()
   C4v.conjugacy_class()
   if C4v.nClass!=5:
      raise ValueError('# of conjugacy classes in C4v is inconsistent')
   print('C4v ConjClass = ',C4v.ConjClass)


   from BDS import character_solver
   BDSsolver = character_solver(C4v.Table, C4v.ConjClass)
   chi = BDSsolver.solve()
   chi[np.abs(chi) < 1e-4] = 0

   print('Character table of C4v = ')
   for i in range(C4v.nClass):
      print(" ".join(f"{x:10.2f}" for x in chi[:,i]))  # Format numbers to 2 decimal places

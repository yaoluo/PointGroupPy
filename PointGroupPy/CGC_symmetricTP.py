# compute CG coefficient for symmetric tensor product of IRs 
import numpy as np 
import itertools 

class symmetricTensorProduct:

   def __init__(self, D_IR):
      self.D_IR = D_IR
      self.nG = len(D_IR[0])
      self.nIR = len(D_IR)
      self.dimIR = [len(D_IR[i]) for i in range(self.nIR)] 
      return 
   
   def build_ijk_rep(self, I, J, K, index_subspace):
      # the basis of Di, Dj, Dk are orthgonal 
      # the matrix rep is the same as the simple tp
      # for any renormalization, the matrix rep is the same 
      nG = self.nG
      d1 = len(self.D_IR[I][0])
      d2 = len(self.D_IR[J][0])
      d3 = len(self.D_IR[K][0])

      if index_subspace[0] < index_subspace[1]:
         raise ValueError('Error : index_subspace should be in descending order')
      if index_subspace[0] < index_subspace[2]:
         raise ValueError('Error : index_subspace should be in descending order')
      if index_subspace[1] < index_subspace[2]:
         raise ValueError('Error : index_subspace should be in descending order')
      if index_subspace[0] == index_subspace[1] and I!=J:
         raise ValueError('Error : I!=J while they are the same space')
      if index_subspace[1] == index_subspace[2] and J!=K:
         raise ValueError('Error : I!=J while they are the same space')
      
      basis = []
      
      for i in range(d1):
         if index_subspace[0]==index_subspace[1]:
            for j in range(i+1):
               if index_subspace[1]==index_subspace[2]:
                  for k in range(j+1):
                     basis.append([[index_subspace[0],I,i],[index_subspace[1],J,j],[index_subspace[2],K,k]])
               else:
                  for k in range(d3):
                     basis.append([[index_subspace[0],I,i],[index_subspace[1],J,j],[index_subspace[2],K,k]])
         else:
            for j in range(d2):
               if index_subspace[1]==index_subspace[2]:
                  for k in range(j+1):
                     basis.append([[index_subspace[0],I,i],[index_subspace[1],J,j],[index_subspace[2],K,k]])
               else:
                  for k in range(d3):
                     basis.append([[index_subspace[0],I,i],[index_subspace[1],J,j],[index_subspace[2],K,k]])
      
      Z = []
      for b in basis:
         if b[0][0] == b[1][0] and b[0][0] == b[2][0]:
            uvs = np.asarray([b[0][2],b[1][2],b[2][2]]).astype(np.int32)
            n_diff = len(np.unique(uvs))
            #print(n_diff)
            if n_diff==1:
               Z.append(1.0) 
            elif n_diff==2:
               Z.append(1.0/3) 
            else:
               Z.append(1.0/6.0) 
         elif b[0][0] == b[1][0] and b[0][0] != b[2][0]:
            if b[0][2] == b[1][2]:
               Z.append(1.0/3)
            else:
               Z.append(1.0/6)
         elif b[0][0] != b[1][0] and b[0][0] == b[2][0]:
            if b[0][2] == b[2][2]:
               Z.append(1.0/3)
            else:
               Z.append(1.0/6)
         else:
            Z.append(1.0/6)
            
      basis = np.asarray(basis).astype(np.int32)
      Z = np.asarray(Z)
      nbasis = len(basis)

      repG = np.zeros([nG, nbasis, nbasis])
      for ig in range(nG):
         G1 = np.zeros([nbasis,nbasis])
         for i in range(nbasis):
            p1 = basis[i]
            for j in range(nbasis):
               for p2 in itertools.permutations(basis[j]):   
                  if p1[0][0] == p2[0][0] and p1[1][0] == p2[1][0] and p1[2][0] == p2[2][0] :
                     G1[i,j] += self.D_IR[p1[0][1]][ig,p1[0][2],p2[0][2]] * self.D_IR[p1[1][1]][ig,p1[1][2],p2[1][2]] * self.D_IR[p1[2][1]][ig,p1[2][2],p2[2][2]]
               # transforms to normalized basis 
               G1[i,j]  = G1[i,j]  / np.sqrt(Z[i]*Z[j])
            G1[i] = G1[i] / (6.0)
         repG[ig] = G1 + 0
         
         
         
      return basis, repG
 
   def build_ijkl_rep(self, I, J, K, L):
      # the basis of Di, Dj, Dk are orthgonal 
      # the matrix rep is the same as the simple tp
      nG = self.nG
      d1 = len(self.D_IR[I][0])
      d2 = len(self.D_IR[J][0])
      d3 = len(self.D_IR[K][0])
      d4 = len(self.D_IR[L][0])

      d = d1*d2*d3*d4  
      #print(f'dims = {d1} {d2} {d3}')
      # 
      basis = []
      for i in range(d1):
         for j in range(d2):
            for k in range(d3):
               for l in range(d4):
                  f = np.zeros([d1,d2,d3,d4])
                  f[i,j,k,l] = 1
                  #f = f / np.sum(f**2)**0.5 
                  basis.append(f)
      basis = np.asarray(basis)
      nbasis = len(basis)

      repG = np.zeros([nG, d, d])
      for ig in range(nG):
         g_t = np.einsum('ar,bs,ct,dq->abcdrstq',self.D_IR[I][ig],self.D_IR[J][ig],self.D_IR[K][ig],self.D_IR[L][ig])
         repG[ig] = g_t.reshape(d,d)
         #G1 = np.zeros([nbasis,nbasis])
         #for i in range(nbasis):
         #   gv = np.einsum('ar,bs,ct,rst->abc',self.D_IR[I][ig],self.D_IR[J][ig],self.D_IR[K][ig], basis[i])
         #   G1[:,i] = np.einsum("jrst,rst->j",basis,gv)
         #print(np.max(np.abs(G1 - repG[ig])))
      return basis, repG
   

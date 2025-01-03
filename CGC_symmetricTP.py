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
   
   def build_ijk_rep(self, I, J, K):
      # the basis of Di, Dj, Dk are orthgonal 
      # the matrix rep is the same as the simple tp
      nG = self.nG
      d1 = len(self.D_IR[I][0])
      d2 = len(self.D_IR[J][0])
      d3 = len(self.D_IR[K][0])


      basis = []
      for i in range(d1):
         for j in range(d2):
            for k in range(d3):
               f = np.zeros([d1,d2,d3])
               f[i,j,k] = 1
               #f = f / np.sum(f**2)**0.5 
               basis.append(f)
      basis = np.asarray(basis)
      nbasis = len(basis)
      #print(f'nbasis = {nbasis}, basis shape ={basis.shape}')

      repG = np.zeros([nG, nbasis, nbasis])
      for ig in range(nG):
         g_t = np.einsum('ar,bs,ct->abcrst',self.D_IR[I][ig],self.D_IR[J][ig],self.D_IR[K][ig])
         repG[ig] = g_t.reshape(nbasis,nbasis)
         #G1 = np.zeros([nbasis,nbasis])
         #for i in range(nbasis):
         #   gv = np.einsum('ar,bs,ct,rst->abc',self.D_IR[I][ig],self.D_IR[J][ig],self.D_IR[K][ig], basis[i])
         #   G1[:,i] = np.einsum("jrst,rst->j",basis,gv)
         #print(np.max(np.abs(G1 - repG[ig])))
      return basis, repG
   

   def build_iik_rep(self, I, J, K):
      # the basis of Di, Dj, Dk are orthgonal 
      # the matrix rep is the same as the simple tp
      nG = self.nG
      d1 = len(self.D_IR[I][0])
      d2 = len(self.D_IR[J][0])
      d3 = len(self.D_IR[K][0])

      if d1!=d2:
         raise ValueError('dimension Error')

      basis = []
      for i in range(d1):
         for j in range(i):
            for k in range(d3):
               f = np.zeros([d1,d2,d3])
               f[i,j,k] = 1
               f[j,i,k] = 1
               f = f / np.sum(f**2)**0.5 
               basis.append(f)
      basis = np.asarray(basis)
      nbasis = len(basis)

      repG = np.zeros([nG, nbasis, nbasis])
      for ig in range(nG):
         G1 = np.zeros([nbasis,nbasis])
         for i in range(nbasis):
            gv = np.einsum('ar,bs,ct,rst->abc',self.D_IR[I][ig],self.D_IR[J][ig],self.D_IR[K][ig], basis[i])
            G1[:,i] = np.einsum("jrst,rst->j",basis,gv)
         repG[ig] = G1
      return basis, repG
   

   def build_iii_rep(self, I, J, K):
      # the basis of Di, Dj, Dk are orthgonal 
      # the matrix rep is the same as the simple tp
      nG = self.nG
      d1 = len(self.D_IR[I][0])
      d2 = len(self.D_IR[J][0])
      d3 = len(self.D_IR[K][0])

      if d1!=d2 or d1!=d2:
         raise ValueError('dimension Error')

      basis = []
      for i in range(d1):
         for j in range(i):
            for k in range(j):
               f = np.zeros([d1,d2,d3])
               S3 = [i,j,k]
               for p in itertools.permutations(S3):
                  f[p[0],p[1],p[2]] = 1.0 
               f = f / np.sum(f**2)**0.5 
               basis.append(f)
      basis = np.asarray(basis)
      nbasis = len(basis)

      repG = np.zeros([nG, nbasis, nbasis])
      for ig in range(nG):
         G1 = np.zeros([nbasis,nbasis])
         for i in range(nbasis):
            gv = np.einsum('ar,bs,ct,rst->abc',self.D_IR[I][ig],self.D_IR[J][ig],self.D_IR[K][ig], basis[i])
            G1[:,i] = np.einsum("jrst,rst->j",basis,gv)
         repG[ig] = G1
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
   

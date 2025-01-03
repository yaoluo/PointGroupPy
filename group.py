import numpy as np
from polynormial import polynomial_space
from mpl_toolkits.mplot3d import Axes3D
class MatrixGroup:
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
         MultilicationTable = np.zeros([nG, nG]).astype(np.int32)
         for i, gi in enumerate(self.G):
            G_prime = np.einsum('iab,bc->iac',self.G, gi)
            for j, gigj in enumerate(G_prime):
               #G[i] * G[j] = G[k]
               k = np.argmin(np.einsum('iab->i',(self.G - gigj)**2))
               if( np.sum((self.G[k] - gigj)**2) >1e-10):
                   raise ValueError('gi*G not equal G')
               MultilicationTable[i,j] = k
         self.MultilicationTable = MultilicationTable  
         print(MultilicationTable)
         # identify identity 
         for i in range(nG):
            if np.max(np.abs(MultilicationTable[i]-np.arange(nG)))==0:
                self.E = i
         print(f'E = {self.E}')
         # construct invG 
         index_invG = np.zeros(nG).astype(np.int32)
         for i in range(nG):
            for j in range(nG):
                if MultilicationTable[i,j] == self.E:
                    index_invG[i] = j
                    index_invG[j] = i
         self.index_invG = index_invG
         #print(f'invG = {index_invG}')
    
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
                  k_gigj = self.MultilicationTable[i,j]
                  i_invgj_gi_gj = self.MultilicationTable[self.index_invG[j],k_gigj]
                  class_index[i_invgj_gi_gj] = ith_conjugacy
               ith_conjugacy += 1
      nClass = np.max(class_index)+1
      ConjClass = [[] for i in range(nClass)] 
      for i in range(nG):
          ConjClass[class_index[i]].append(i)

      self.class_index = class_index
      self.ConjClass = ConjClass
      self.nClass = nClass
      self.sClass = np.array([len(self.ConjClass[i]) for i in range(nClass)])
      #print('class_index = ',class_index)
      return 
    
    def check_rep(self, Dg):
        for c in self.ConjClass:
            for a in c: 
                if abs(np.trace(Dg[a])- np.trace(Dg[c[0]])) > 1e-10 :
                    raise ValueError('This rep is inconsistent to the conjugacy class')
        return 
    #decompostion 
    def decompose(self, Dg, chi_table):
        # Dg, rep of G
        self.chi_table = chi_table
        nClass = self.nClass 
        chi_rep = np.asarray([np.trace(Dg[self.ConjClass[i][0]]) for i in range(nClass)])

        # check conjclass 
        self.check_rep(Dg)

        #print('chi rep = ',chi_rep)
        Multiplicity = np.asarray([ np.dot(chi_rep*self.sClass, np.conj(self.chi_table[:,i])) for i in range(nClass)])/len(self.G)
        Multiplicity[np.abs(Multiplicity)<1e-10] = 0
        #print('Multiplicity = ',Multiplicity)
        if abs(np.sum(Multiplicity)-1)<1e-10:
            print('This representation is already irreducible!')
        return Multiplicity

    def projector_Iu(self, Dg, Dg_IR):
        # project phi to IR basis |Iu>
        Proj_Iu = np.einsum('iab,icd->acd',Dg_IR,Dg) / self.nG
        return Proj_Iu
    
    def projector_I(self, Dg, Dg_IR):
        nG,dim,_ = Dg_IR.shape 
        chi = np.trace(Dg_IR, axis1=1,axis2=2)
        Proj_I = np.einsum('i,iab->ab',chi*chi[0], Dg) / self.nG
        return Proj_I

    def basis_function(self, Dg, chi_table, excluded_space = []):
        #excluded_space = size of [*, nbasis]
        #
        nG, ndim, _ = Dg.shape 
        #Proj = self.projector_I()
        Proj = np.zeros([self.nClass,ndim, ndim])
        for ic in range(self.nClass):
            for ig in range(nG):
                Proj[ic] = Proj[ic] + chi_table[0, ic] * chi_table[self.class_index[ig], ic] * Dg[ig]
        Proj = Proj / nG 


        if len(excluded_space)!=0:
            # make sure that excluded_space is orth-normal basis 
            I = excluded_space@excluded_space.T 
            e = np.max(np.abs(I - np.eye(len(I))))
            if e > 1e-10:
                raise ValueError('excluded_space is not orth-norm basis! Please do your own homework.')
            proj_out =  excluded_space.T@excluded_space
            for ic in range(self.nClass): 
                Proj[ic] = Proj[ic] - Proj[ic]@proj_out

        Si = []
        for ic in range(self.nClass): 
            evals, evec = np.linalg.eigh(Proj[ic])
            evec[np.abs(evec)<1e-10] = 0
            Si.append(evec[:,np.abs(evals)>1e-5])
        return Si
    
    def basis_function_break_multiplicity(self, Dg, chi_table, excluded_space = []):
        #excluded_space = size of [*, nbasis]
        #
        nG, ndim, _ = Dg.shape 
        Proj = np.zeros([self.nClass,ndim, ndim])
        for ic in range(self.nClass):
            Proj[ic] = self.projector_I(Dg, self.D_IR[ic])

        if len(excluded_space)!=0:
            # make sure that excluded_space is orth-normal basis 
            I = excluded_space@excluded_space.T 
            e = np.max(np.abs(I - np.eye(len(I))))
            if e > 1e-10:
                raise ValueError('excluded_space is not orth-norm basis! Please do your own homework.')
            proj_out =  excluded_space.T@excluded_space
            for ic in range(self.nClass): 
                Proj[ic] = Proj[ic] - Proj[ic]@proj_out
        basis_dict = {} 
        i_sub = 0
        for ic in range(self.nClass): 
            evals, evec = np.linalg.eigh(Proj[ic])
            evec[np.abs(evec)<1e-10] = 0
            #
            dim_chi = int(chi_table[0,ic]+1e-5)
            # subspace of shape [dim, nbasis]
            dim = int(sum(np.abs(evals)>1e-10)+1e-5)
            multiplicity = int(dim / dim_chi+1e-5)
            subspace = evec[:,np.abs(evals)>1e-5]
            subspace[np.abs(subspace)<1e-10] = 0
            if multiplicity==1:
                basis_dict[f'{i_sub}-th subspace'] = {'IR index': ic, 'dimension': dim_chi ,'basis': np.transpose(subspace)}
                i_sub += 1 
            else:
                #Proj_Iu of shape [dim_chi, nbasis, nbasis] 
                
                Proj_Iu = self.projector_Iu(Dg, self.D_IR[ic])
                Projout_included = np.eye(len(Dg[0]))
                for alpha in range(multiplicity):
                    print('subspace.shape = ',subspace.shape, np.random.random(dim).shape)
                    phi = subspace@(np.random.random(dim))  
                    # project the states that are already tabulated   
                    phi = Projout_included@phi
                    subspace_seperate = []
                    for mu in range(dim_chi):
                        phi_u = Proj_Iu[mu]@phi
                        phi_u = phi_u/np.linalg.norm(phi_u)
                        phi_u[np.abs(phi_u)<1e-10] = 0
                        subspace_seperate.append(phi_u.reshape(-1))
                        Projout_included = Projout_included - np.einsum('i,j->ij', phi_u,phi_u)
                    basis_dict[f'{i_sub}-th subspace'] = {'IR index': ic, 'dimension': dim_chi ,'basis': np.asarray(subspace_seperate)}
                    i_sub += 1
        return basis_dict
    
    def basis_function_for_oneIR(self, Dg, D_IR, excluded_space = []):
        #excluded_space = size of [*, nbasis]
        # project Dg to
        
        nG, ndim, _ = Dg.shape 
        chi_IR = np.array([np.trace(D_IR[ig]) for ig in range(nG)])
        chi_G = np.array([np.trace(Dg[ig]) for ig in range(nG)])
        

        
        Proj = np.zeros([ndim, ndim])
        Proj = self.projector_I(Dg, D_IR)

        if len(excluded_space)!=0:
            # make sure that excluded_space is orth-normal basis 
            I = excluded_space@excluded_space.T 
            e = np.max(np.abs(I - np.eye(len(I))))
            if e > 1e-10:
                raise ValueError('excluded_space is not orth-norm basis! Please do your own homework.')
            proj_out =  excluded_space.T@excluded_space
           
            Proj = Proj - Proj@proj_out
        basis_dict = {} 
        i_sub = 0
 
        evals, evec = np.linalg.eigh(Proj)
        evec[np.abs(evec)<1e-10] = 0
        #
        
        _,dim_chi,_ = D_IR.shape  
        # subspace of shape [dim, nbasis]
        dim = int(sum(np.abs(evals)>1e-10)+1e-5)
        multiplicity = int(np.sum(chi_IR * chi_G) / nG+1e-3)
       
        if multiplicity != int(dim / dim_chi+1e-5):
            raise ValueError('Error on multiplicity') 

        if multiplicity == 0 :
            return []
        
        subspace = evec[:,np.abs(evals)>1e-5]
        subspace[np.abs(subspace)<1e-10] = 0
        if multiplicity==1:
            basis_dict[f'{i_sub}-th subspace'] = {'chi': chi_IR, 'dimension': dim_chi ,'basis': np.transpose(subspace)}
            i_sub += 1 
        else:
            #Proj_Iu of shape [dim_chi, nbasis, nbasis]
            Proj_Iu = self.projector_Iu(Dg, D_IR)
            Projout_included = np.eye(len(Dg[0]))
            for alpha in range(multiplicity):
                #print('subspace.shape = ',subspace.shape, np.random.random(dim).shape)
                phi = subspace@(np.random.random(dim))  
                # project the states that are already tabulated   
                phi = Projout_included@phi
                subspace_seperate = []
                for mu in range(dim_chi):
                    phi_u = Proj_Iu[mu]@phi
                    phi_u = phi_u/np.linalg.norm(phi_u)
                    phi_u[np.abs(phi_u)<1e-10] = 0
                    subspace_seperate.append(phi_u.reshape(-1))
                    Projout_included = Projout_included - np.einsum('i,j->ij', phi_u,phi_u)
                basis_dict[f'{i_sub}-th subspace'] = {'chi': chi_IR, 'dimension': dim_chi ,'basis': np.asarray(subspace_seperate)}
                i_sub += 1
        return basis_dict
    
    def build_explicit_IRmatrix(self, chi_table):
        self.chi_table = chi_table
        names = self.name_IR()
        Ls = [0,1,2,3,4,5,6,7,8,9] 
        xl = [ polynomial_space( l = l) for l in Ls]
        Multiplicity = []
        for i in range(len(xl)):
           xl[i].RepOfGroup(self)
           Multiplicity.append(self.decompose(xl[i].polyG, chi_table))

        print('Decompostion of Cartesian Harmonics in the IRs = ')
        for i in range(len(Multiplicity)):
           print(f' l = {Ls[i]} = ',Multiplicity[i])

        # identify the lowest basis function for one IR
        reps_L = [] 
        print(self.nClass)
        for ir in range(self.nClass):
           for i in range(len(xl)):
              if abs(Multiplicity[i][ir]-1)<1e-5:
                 reps_L.append(i)
                 break 
        if len(reps_L)!=self.nClass:
            raise ValueError('Cartesian Harmonic up to l = 9 cannot cover the full IRs!')
        print('reps_L = ',reps_L)

        # build the invaraint subspace of each irredicuble representation 
        # collect the basis function and matrix rep for each IR 
        # for ir-th IR, we look at the rep in irreps_L[ir]
        D_IR = [ np.zeros([self.nG, int(chi_table[0,i]+0.1), int(chi_table[0,i]+0.1)]) for i in range(self.nClass)]
        for ir in range(self.nClass):
            if( int(chi_table[0,ir]+0.1)==1 ):
                for ig in range(self.nG):
                    D_IR[ir][ig,0,0] = chi_table[self.class_index[ig],ir] 
            else:
                sym_f = self.basis_function(xl[reps_L[ir]].polyG, chi_table)
                #f_ir transforms as the ir-th irreducible rep 
                f_ir = sym_f[ir]
                #polarize states in the z direction 
                V = np.zeros([xl[reps_L[ir]].nbasis,xl[reps_L[ir]].nbasis])
                #pick the maxium amplitude ones and add a potential to lift it.
                amplitude = np.sum(f_ir**2, axis=1)
                i_max = np.argmax(amplitude)
                if abs(amplitude[-1]-amplitude[i_max])<1e-5:
                    i_max = -1 
                V[i_max, i_max] = 1
        
                eval, evc = np.linalg.eigh(f_ir.T@V@f_ir) 
                f_ir = f_ir@evc
                for i in range(len(f_ir[0])):
                    # choose a positive convention 
                    f_ir[:,i] = np.sign( f_ir[np.argmax(np.abs(f_ir[:,i])), i] ) * f_ir[:,i]
                print(f'----------------------------------')
                print(f'{ir}-th IR, basis = {names[ir]}')
                for i in range(int(chi_table[0,ir]+0.1)):
                   xl[reps_L[ir]].print_poly(f_ir[:,i])
                
                for ig in range(self.nG):
                   D_IR[ir][ig] = f_ir.T@xl[reps_L[ir]].polyG[ig]@f_ir 
        self.D_IR = D_IR

        #check the trace 
        for ig in range(self.nG):
            for ic in range(self.nClass):
                e = np.max(np.abs(np.trace(D_IR[ic][ig]) - chi_table[self.class_index[ig],ic]))
                #print(e)
                if e> 1e-5:
                    raise ValueError('trace error')
    
    def name_IR(self):
        Names = [] 
        for ir in range(self.nClass):
            s = ""
            if abs(self.chi_table[0,ir]-1)<1e-5:
                s = s+'A/B'
            if abs(self.chi_table[0,ir]-2)<1e-5:
                s = s+'E'
            if abs(self.chi_table[0,ir]-3)<1e-5:
                s = s+'T'
            sigma_d = np.array([[0,1,0],[1,0,0],[0,0,1]])
            for ig, g in enumerate(self.G):
                if np.abs(g - sigma_d).max()<1e-5:
                    if self.chi_table[self.class_index[ig],ir]>0:
                        s = s+'_d=+'
                    else:
                        s = s+'_d=-'
                    break 
            sigma_v = np.array([[-1,0,0],[0,1,0],[0,0,1]])
            for ig, g in enumerate(self.G):
                if np.abs(g - sigma_v).max()<1e-5:
                    if self.chi_table[self.class_index[ig],ir]>0:
                        s = s+'_v=+'
                    else:
                        s = s+'_v=-'
                    break 
            sigma_h = np.array([[1,0,0],[0,1,0],[0,0,-1]])
            for ig, g in enumerate(self.G):
                if np.abs(g - sigma_h).max()<1e-5:
                    if self.chi_table[self.class_index[ig],ir]>0:
                        s = s+'_h=+'
                    else:
                        s = s+'_h=-'
                    break 
            inv = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
            for ig, g in enumerate(self.G):
                if np.abs(g - inv).max()<1e-5:
                    if self.chi_table[self.class_index[ig],ir]>0:
                        s = s+'_g'
                    else:
                        s = s+'_u'
                    break 
            Names.append(s)
        return Names


if __name__ == '__main__':
   for n in [4]:
      th = 2*np.pi/n
      Id = np.array([[1,0],[0,1]])
      R90 = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])
      Cn = MatrixGroup(generator = [Id,R90])
      print(f'|C{n}| = {Cn.nG}')
      if Cn.nG!=n:
          raise ValueError('# of elements in Cn is inconsistent')
      Cn.constructMultiplicationTable()

   #C4v 
   Id = np.array([[1,0],[0,1]])
   th = np.pi/2
   R90 = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])
   sigma_v = np.array([[-1,0],[0,1]])
   sigma_d = np.array([[0,1],[1,0]])
   C4v = MatrixGroup(generator = [Id,R90,sigma_v,sigma_d])
   print(f'|C4v| = {C4v.nG}')
   if C4v.nG!=8:
      raise ValueError('# of elements in C4v is inconsistent')
   C4v.constructMultiplicationTable()
   C4v.conjugacy_class()
   if C4v.nClass!=5:
      raise ValueError('# of conjugacy classes in C4v is inconsistent')
   print('C4v ConjClass = ',C4v.ConjClass)

   #
   from BDS_solver import character_solver
   ChiSolver = character_solver(C4v.Table, C4v.ConjClass)
   chi_table = ChiSolver.solve()
   chi_table[np.abs(chi_table) < 1e-4] = 0
   print('Character table of C4v = ')
   for i in range(C4v.nClass):
      print(" ".join(f"{x:10.2f}" for x in chi_table[:,i]))  # Format numbers to 2 decimal places
   #apply C4v to the polynomial of (x,y)

   # build the matrix representtaion of C4v on linear function of (x,y)
   # Here, it is coincidently the same as the original C4v, for cubic or quadratic, it will be different. 
   C4v.decompose(chi_table)
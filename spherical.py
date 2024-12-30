import numpy as np 

def R_from_abc(alpha, beta, gamma):
    """
    Generate the rotation matrix using the ZXZ proper Euler convention.

    Parameters
    ----------
    alpha : float
        The first Euler angle (rotation about the x-axis).
    beta : float
        The second Euler angle (rotation about the z-axis).
    gamma : float
        The third Euler angle (rotation about the x-axis).

    Returns
    -------
    numpy.ndarray
        The 3x3 rotation matrix corresponding to the given Euler angles.
    """
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    c_beta = np.cos(beta)
    s_beta = np.sin(beta)
    c_gamma = np.cos(gamma)
    s_gamma = np.sin(gamma)
    
    # Construct the matrix
    matrix = np.array([
        [c_alpha * c_gamma - c_beta * s_alpha * s_gamma, -c_alpha * s_gamma - c_beta * c_gamma * s_alpha, s_alpha * s_beta],
        [c_gamma * s_alpha + c_alpha * c_beta * s_gamma, c_alpha * c_beta * c_gamma - s_alpha * s_gamma, -c_alpha * s_beta],
        [s_beta * s_gamma, c_gamma * s_beta, c_beta]
    ])
    return matrix  # Return the rotation matrix

def abc_from_R(R):
    """
    Compute the proper Euler angles in the XZX convention given a rotation matrix.

    Parameters
    ----------
    R : numpy.ndarray
        A 3x3 rotation matrix.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the Euler angles [alpha, beta, gamma].
    
    Raises
    ------
    ValueError
        If the input rotation matrix is not unitary.
    """
    '''
    compute the proper Euler angle in xzx convention given R \in O(3)
    '''
    I = np.matmul(np.transpose(R), R)  # Compute the identity matrix
    #c = np.trace(cI) / 3.0 
    err = np.max(np.abs(I - np.eye(3)))  # Check for unitarity
    if err > 1e-10:
        raise ValueError('R matrix is not unitary!')  # Raise error if R is not unitary

    # map R = {E, i} * Rot, where i = inversion 
    c = np.linalg.det(R)  # Compute the determinant of R
    Rot = R * np.sign(c) # Normalize the rotation matrix
    if (abs(Rot[2, 2])>1.0 and abs(Rot[2, 2])<1.0 + 1e-10):
        Rot[2, 2] = 1.0 - 1e-12

    beta = np.arccos(Rot[2, 2])  # Calculate beta
    if Rot[2,2]>1.0 - 1e-6:
        alpha = np.arccos(Rot[0, 0] )  
        gamma = 0.0
        return np.array([alpha, beta, gamma])  # Return the Euler angles
    if Rot[2,2]<-1.0 + 1e-6:
        alpha = np.arccos(Rot[0, 0] )  
        gamma = 0.0
        return np.array([alpha, beta, gamma])  # Return the Euler angles
    alpha = np.arctan(-Rot[0, 2] / Rot[1, 2])  # Calculate alpha
    beta = np.arccos(Rot[2, 2])  # Calculate beta
    gamma = np.arctan(Rot[2, 0] / Rot[2, 1])  # Calculate gamma
    return np.array([alpha, beta, gamma])  # Return the Euler angles
    
from scipy.special import sph_harm
class sphericalHarmon:
    
    def __init__(self, l, convention = 'original'):
        self.l = l 
        self.m = np.arange(-l,l+1)
        self.S = np.eye(2*l+1)
        self.iS = self.S + 0 
        if convention=='real':
            self.convenstion_real()
        elif convention=='original':
            self.real = False 
        else:
            raise ValueError('un-surpport convention encountered@sphericalHarmon')
        return 
    
    def Yl(self, theta, phi):
        ylm = np.array([sph_harm(m, l, phi, theta) for m in range(-l, l + 1)])
        ylm_conv = self.S@ylm
        if self.real: 
            return  ylm_conv.real 
        return ylm_conv
    
    def __call__(self, rvec):
        rvec = rvec / np.linalg.norm(rvec)
        x,y,z = rvec 
        theta = np.arccos(z)
        if abs(z-1) < 1e-6:
            phi = 0.0
            return self.Yl(theta, phi)
        phi = np.sign(y) * np.arccos(x / (x**2+y**2)**0.5)
        return self.Yl(theta, phi)
    
    def convenstion_real(self):
        #m<0, Real convention of Ylm
        l = self.l
        S = np.complex128(np.zeros([l*2+1,l*2+1]))
        i = 0 
        for m in self.m:
            if m<0:
                S[i,m+l] = 1j/(2**0.5) 
                S[i,-m+l] = -1j/(2**0.5)  * (-1)**(2*l+m) 
            if m>0:
                S[i,m+l] = 1/(2**0.5) 
                S[i,-m+l] = 1/(2**0.5)  * (-1)**(2*l+m) 
            if m ==0:
                S[i,m+l] = 1
            i += 1
        self.S = S
        self.iS = np.linalg.inv(S)
        self.real = True 
        return
    
    def WignerD(self, alpha, beta, gamma):
        D = wignerD_FromEuler(self, a, b, c)
        D_S = np.conj(self.S@(np.conj(D)@self.iS))
        if self.real:
            return D_S.real 
        return D_S
    
    def WignerD_fromR(self, R):
        a,b,c = abc_from_R(R)
        #print('alpha, beta, gamma = ',a,b,c)
        D = wignerD_FromEuler(self, a, b, c)
        D_S = np.conj(self.S@(np.conj(D)@self.iS))
        if self.real:
            return D_S.real 
        return D_S
    
#drdrqC$v93HU6*w， password for GEM
from math import factorial

def wigner_small_d(j, m_prime, m, beta):
    """
    Computes the Wigner small d-matrix element d^j_{m'm}(beta).
    
    Parameters:
        j (int): Total angular momentum quantum number.
        m_prime (int): Magnetic quantum number after rotation.
        m (int): Magnetic quantum number before rotation.
        beta (float): The Euler angle beta (in radians).
    
    Returns:
        float: The value of d^j_{m'm}(beta).
    """
    # Factorial prefactor
    prefactor = np.sqrt(
        factorial(j + m_prime) * factorial(j - m_prime) *
        factorial(j + m) * factorial(j - m)
    )
    
    # Define summation limits
    s_min = max(0, m - m_prime)
    s_max = min(j + m, j - m_prime)
    
    # Summation over s
    c_halfb = np.cos(beta / 2)
    s_halfb = np.sin(beta / 2)

    if abs(s_halfb)<1e-6:
        summation = 0
        for s in range(s_min, s_max + 1):
            if (m_prime-m+2*s==0):
                term = (
                    (-1)**(s) *
                    (c_halfb**(2 * j + m - m_prime - 2 * s)) /
                    (factorial(j + m - s) * factorial(s) *
                     factorial(m_prime - m + s) * factorial(j - m_prime - s))
                )
                summation += term
        # Combine prefactor and summation
        d_element = prefactor * summation * (1j)**(m-m_prime) 
        return d_element
    
    if abs(c_halfb)<1e-6:
        summation = 0
        for s in range(s_min, s_max + 1):
            if (2*l+m-m_prime-2*s==0):
                term = (
                    (-1)**(s) * (s_halfb**(m_prime - m + 2 * s)) /
                    (factorial(j + m - s) * factorial(s) *
                     factorial(m_prime - m + s) * factorial(j - m_prime - s))
                )
                summation += term
        # Combine prefactor and summation
        d_element = prefactor * summation * (1j)**(m-m_prime) 
        return d_element
    
    summation = 0
    for s in range(s_min, s_max + 1):
        term = (
            (-1)**(s) *
            (c_halfb**(2 * j + m - m_prime - 2 * s)) *
            (s_halfb**(m_prime - m + 2 * s)) /
            (factorial(j + m - s) * factorial(s) *
             factorial(m_prime - m + s) * factorial(j - m_prime - s))
        )
        summation += term
    
    # Combine prefactor and summation
    d_element = prefactor * summation * (1j)**(m-m_prime) 
    return d_element

def wignerD_FromEuler(SPH : sphericalHarmon, alpha, beta, gamma):
    dim = 2*SPH.l+1
    D_l = np.complex128(np.zeros([dim,dim]))
    d_l = np.complex128(np.zeros([dim,dim]))
    for i in range(dim):
        m_prime = SPH.m[i]
        for j in range(dim):
            m = SPH.m[j]
            #d_l[i,j] =  wigner_small_d(SPH.l, m_prime, m, beta)
            D_l[i,j] = wigner_small_d(SPH.l, m_prime, m, beta) * np.exp(-1j*(alpha*m_prime+gamma*m))
    #print('d_l = ',d_l)
    return D_l

def wignerD_FromR(SPH:sphericalHarmon, R):
    c = np.linalg.det(R)
    if abs(np.abs(c)-1)>1e-6:
        raise ValueError('R is not unitary in wignerD_FromR')
    alpha, beta, gamma = abc_from_R(R)
    print('det R = ',c)
    D_l = (c)**(SPH.l) * wignerD_FromEuler(SPH, alpha, beta, gamma)
    return D_l

if __name__ == '__main__':
    #test 
    a, b ,c = np.random.random(3)
    R = R_from_abc(a,b,c)
    print(f'R(alpha, beta, gamma) = \n ',R)
    # [1] check Identity 
    I = R@ R.T
    print("\n".join([" ".join([f"{element:.5f}" for element in row]) for row in I]))
    if np.abs(I - np.eye(3)).max()>1e-10:
        raise ValueError('R@ R.T != identity')
    
    # [2] check consistency between abc_from_R and rotMat_abc
    ap, bp, cp = abc_from_R(R)
    Rp = R_from_abc(ap, bp, cp)
    print('|R - Rp| = ',np.abs(R - Rp).max())
    if np.abs(R - Rp).max()>1e-10:
        raise ValueError('abc_from_R not consistent with R_from_abc')
    
    # [3] test l 
    print(f'test rotating Ylm with wigner D matrix, using original convention')
    for l in [1,2,3,4,5]:
        a, b ,c = np.random.random(3)
        R = R_from_abc(a,b,c)
        Y2 = sphericalHarmon(l, convention = 'original')
        #theta, phi = np.random.random(2)
        rvec = np.random.random(3)
        rvec = np.array([0,0,1])
        rvec = rvec/np.linalg.norm(rvec)
        rvec_rot =R@rvec
        Yl_unrot = Y2( rvec )
        Yl_rot = Y2( rvec_rot )
        Dl_R =   Y2.WignerD(a, b, c)
        error = Yl_rot - np.conj(Dl_R)@Yl_unrot
        print(f'L{l}: error = {np.abs(error).max()}')
        if np.abs(error).max()>1e-10:
            raise ValueError('Wigner D matrix error')
    #test new convention 
    # [3] test l 
    print(f'test rotating Ylm with wigner D matrix, using real convention')
    for l in [1,2,3,4,5]:
        a, b ,c = np.random.random(3)
        R = R_from_abc(a,b,c)
        Y2 = sphericalHarmon(l)
        Y2.convenstion_real()
        #theta, phi = np.random.random(2)
        rvec = np.random.random(3)
        rvec = np.array([0,0,1])
        rvec = rvec/np.linalg.norm(rvec)
        rvec_rot = R@rvec
        Yl_unrot = Y2( rvec )
        Yl_rot = Y2( rvec_rot )
        #Dl_R =  Y2.WignerD(a, b, c)
        Dl_R =  Y2.WignerD_fromR(R)
        error = Yl_rot - np.conj(Dl_R)@Yl_unrot
        print(f'L{l}: error = {np.abs(error).max()}')
        if np.abs(error).max()>1e-10:
            raise ValueError('Wigner D matrix error')


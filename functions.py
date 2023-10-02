"""
``wavegenerator.functions``
========================

The WaveGenerator functions are based on Schaffer (1996).

Functions present in wavegenerator.functions are listed below.

Functions
---------

    Name    Eq # (Schaffer (1996))
    ----    ----------------------
    kj      Equation (15)
    Dj      Equation (21)
    cj      Equation (20)
    K0      Equation (33)
    H       Equation (25d)
    M1      Equation (41c)
    delta   Equation (25b)
    M_2     Equation (36b)
    F       Equation (41a)
    E       Equation (41b)

"""

import numpy as np
from scipy.optimize import fsolve

# parameters based in Fig. 1 from Schaffer (1996)
h = 1.0 # still water depth in meters

# if the centre of rotation is over the bottom: -h<l<inf AND d>=0.
l = -0.3
d = -l # d>=0 is the elevation of the hinge over the bottom, i.e., d=-l.

# if the centre of rotation is at OR below the bottom: 0=<l<=inf AND d=0.
#l = 100.0 #0.3
#d = 0.0

# other parameters
g = 9.81 # acceleration due to gravity in m/s^2

# small ordering parameter relating to the wave steepness, and the subscripts (1) and (2) denote the order of the quantity
epsilon_1 = 1e-15
epsilon_2 = 1e-15

modes = 10 # number of modes

def kj(w):
    """
    Function k_j based in Equation (15) from Schaffer (1996).
    """

    kjlist=[]
    
    for i in range(modes+1):

        if i==0:
            def dispersion(k, g, w, h):
                return np.power(w, 2) - k*g*np.tanh(k*h) #(1.0/k)-(g/w**2.0)*np.tanh(k*h)
            wavenumber = fsolve(dispersion, [0.1], args=(g, w, h))[0]
            kjlist.append(wavenumber)

        else:
            def imag_dispersion(k, g, w, h, i):
                return np.arctan(-w**2.0/(g*k))-k*h-(np.pi*i/h)
            wavenumber = fsolve(imag_dispersion, [0.1], args=(g, w, h, i))[0]
            kjlist.append(wavenumber*complex(0 + 1j))

    return np.asarray(kjlist, dtype = 'complex_')

def Dj(k_j):
    """
    Function D_j based in Equation (21) from Schaffer (1996).
    """
    return (k_j*h/2.0) * (((k_j*h) / (np.sinh(k_j*h) * np.cosh(k_j*h))) + 1.0)

def cj(w):
    """
    Function c_j based in Equation (20) from Schaffer (1996).
    """
    k_j = kj(w)
    D_j = Dj(k_j)

    return ((w**2*h/g) - (h/(h+l))) * (1.0/D_j) + (h/(h+l)) * (1.0/D_j) * (np.cosh(k_j*d) / np.cosh(k_j*h))

def K0(w_n, w_m, sign_value):
    """
    Function K^{+-}_{p} based in Equation (33) from Schaffer (1996).
    """
    return kj(w_n + sign_value*w_m)[0]

def H(w_n, w_m, j, l, sign_value):
    """
    Function H^{+-}_{jnlm} based in Equation (25d) from Schaffer (1996).
    """
    k_j_w_n = kj(w_n)
    k_j_w_m = kj(w_m)

    return (w_n + sign_value*w_m)*(sign_value*w_n*w_m - (g**2*k_j_w_n[j]*k_j_w_m[l])/(w_n*w_m))\
                                    + ((w_n**3 + sign_value*w_m**3)/2.0) - ((g**2)/2.0) * ((k_j_w_n[j]**2/w_n) + sign_value*(k_j_w_m[l]**2/w_m))

#Esses H estão dando valores muito altos, mas enfim, continuamos.

def M1(w_n, w_m, K_0, sign_value):
    """
    Function M_1 based in Equation (41c) from Schaffer (1996).
    """
    #K_0 = K0(w_n, w_m, sign_value)

    return ((1.0/(h+l)) * (g/(np.power(w_n+sign_value*w_m, 2))) \
            * ((np.cosh(K_0*d) / np.cosh(K_0*h)) - 1.0))

def delta(w_n, w_m):
    """
    Function delta based in Equation (25b) from Schaffer (1996).
    """

    if w_n == w_m:
        return 0.5
    else:
        return 1.0

def M2(w_n, w_m, sign_value):
    """
    Function M_2 based in Equation (36b) from Schaffer (1996).
    """

    K_0 = K0(w_n, w_m, sign_value)
    k_j = kj(w_n)
    
    Ch1p = 2.0*k_j*K_0 * (1.0 - (np.cosh(k_j*d)*np.cosh(K_0*d)) / (np.cosh(k_j*h)*np.cosh(K_0*h)))
    
    Ch2p = -(np.power(k_j, 2) + np.power(K_0, 2)) * (((np.power(w_n, 2)*np.power(w_n+sign_value*w_m, 2)) / (g*g*k_j*K_0)) - (np.sinh(k_j*d)*np.sinh(K_0*d)) / (np.cosh(k_j*h)*np.cosh(K_0*h)))
    
    return -(g/(h+l)) * ((K_0/k_j)/(np.power(k_j, 2) - np.power(K_0, 2))) * (Ch1p + Ch2p)

@np.vectorize
def F(w_n, w_m, sign='+'):
    """
    Function F based in Equation (41a) from Schaffer (1996).

    Parameters
    ----------
    w_n : float
        first wave component angular frequency.

    w_m : float
        second wave component angular frequency.

    signal : str
        sign of the function '+' or '-'.

    Returns
    -------
    F : numpy.array
        array containing the ?.
    """

    sign_value = None
    if sign=='+': sign_value = 1.0
    elif sign=='-': sign_value = -1.0
    
    def E(w_n, w_m, sign_value, K_0, c_j_w_n, c_j_w_m):
        """
        Function E^{+-} based in Equation (41b) from Schaffer (1996).
        """
        #K_0 = K0(w_n, w_m, sign_value)
        #c_j_w_n = cj(w_n)
        #c_j_w_m = cj(w_m)

        return ((delta(w_n, w_m) * np.power(K_0, 2)*h) / (c_j_w_n[0] * c_j_w_m[0] * np.power(w_n+sign_value*w_m, 3)*(1.0+M1(w_n, w_m, K_0, sign_value))))

    def SUM1(w_n, w_m, c_j_w_n, K_0_, k_j_w_n):
        sum_ = 0.0

        c_j = c_j_w_n #cj(w_n)
        k_j = k_j_w_n #kj(w_n)
        K_0 = np.power(K_0_, 2)
        M_2 = M2(w_n, w_m, sign_value)
        c_00 = np.power(w_n, 2) - np.power(w_n+w_m, 2)

        for i in range(modes+1):
            a = c_j[i]
            b = np.power(k_j[i], 2) / (np.power(k_j[i],2) - K_0)
            c = c_00 + M_2[i]
            sum_ += a*b*c

        return sum_
    
    def SUM2(w_n, w_m, c_j_w_m, k_j_w_m, K_0_):
        sum_ = 0.0

        c_j = c_j_w_m #cj(w_m)
        k_j = k_j_w_m #kj(w_m)
        K_0 = np.power(K_0_, 2)
        M_2 = M2(w_m, w_n, sign_value)
        c_00 = np.power(w_m, 2) - np.power(w_n+w_m, 2)

        for i in range(modes+1):
            a = c_j[i]
            b = np.power(k_j[i], 2) / (np.power(k_j[i],2) - K_0)
            c = c_00 + M_2[i]
            sum_ += a*b*c

        return sum_

    def SUM3(w_n, w_m, c_j_w_n, c_j_w_m, k_j_w_n, k_j_w_m, K_0_):
        sum_ = 0

        #c_j_w_n = cj(w_n)
        #c_j_w_m = cj(w_m)
        #k_j_w_n = kj(w_n)
        #k_j_w_m = kj(w_m)
        K_0 = np.power(K_0_, 2)

        for i in range(modes+1):
            for l in range (0,modes):
                a = c_j_w_n[i] * c_j_w_m[l]
                b = (k_j_w_n[i] + k_j_w_m[l]) / (np.power(k_j_w_n[i] + k_j_w_m[l], 2) - K_0)
                c = H(w_n, w_m, i, l, sign_value)
                sum_ += a*b*c

        return sum_
    
    c_j_w_n = cj(w_n)
    c_j_w_m = cj(w_m)
    k_j_w_n = kj(w_n)
    k_j_w_m = kj(w_m)
    K_0 = K0(w_n, w_m, sign_value)
    
    result = E(w_n, w_m, sign_value, K_0, c_j_w_n, c_j_w_m) * (-sign_value*(g/(2.0*w_n)) * SUM1(w_n, w_m, c_j_w_n, K_0, k_j_w_n) \
                                    -sign_value*(g/(2.0*w_m)) * SUM2(w_n, w_m, c_j_w_m, k_j_w_m, K_0) \
                                    + SUM3(w_n, w_m, c_j_w_n, c_j_w_m, k_j_w_n, k_j_w_m, K_0))
    result = result/delta(w_n, w_m) # similar to Schaffer (1996)

    return result

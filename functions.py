"""
``wavegenerator.functions``
========================

The WaveGenerator functions are based on Schaffer (1996).

Functions present in wavegenerator.functions are listed below.

Functions
---------

    kj
    Dj

"""

from itertools import combinations
#from mpmath import *

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

def Dj(w):
    return(((kj(w)*h)/2)*(((kj(w)*h)/(np.sinh(kj(w)*h)*np.cosh(kj(w)*h)))+1))

def cj(w):
    return((w**2*h/g)-(h/(h+l)))*(1/Dj(w))+(h/(h+l))*(1/Dj(w))*(np.cosh(kj(w)*d)/np.cosh(kj(w)*h))

def K0(w_n, w_m, sign_value):
    return kj(w_n + sign_value*w_m)[0]

def Hp(w1,w2,j,l):
    return (w1+w2)*(w1*w2-g**2*kj(w1)[j]*kj(w2)[l]/(w1*w2))+(w1**3+w2**3)/2-((g**2)/2)*(kj(w1)[j]**2/w1+kj(w2)[l]**2/w2)

def Hl(w1,w2,j,l):
    return ((w1-w2)*(-w1*w2-g**2*kj(w1)[j]*np.conjugate(kj(w2)[l])/(w1*w2))+(w1**3-w2**3)/2-((g**2)/2)*(kj(w1)[j]**2/w1-kj(w2)[l]**2/w2))   

#Esses H estão dando valores muito altos, mas enfim, continuamos.

def M1(w_n, w_m, sign_value):
    return ((1.0/(h+l)) * (g/(np.power(w_n+sign_value*w_m, 2))) \
            * ((np.cosh(K0(w_n, w_m, sign_value)*d) / np.cosh(K0(w_n, w_m, sign_value)*h)) - 1.0))

def delta(w_n, w_m):
    """
    Function delta based in Equation (25b) from Schaffer (1996).
    """

    if w_n == w_m:
        return 0.5
    else:
        return 1.0

def M2(w_n, w_m, sign_value):

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
    
    def E(w_n, w_m, sign_value):
        return ((delta(w_n, w_m) * np.power(K0(w_n, w_m, sign_value), 2)*h) / (cj(w_n)[0] * cj(w_m)[0] * np.power(w_n+sign_value*w_m, 3)*(1.0+M1(w_n, w_m, sign_value))))

    def SUM1(w_n, w_m):
        sum_ = 0.0

        c_j = cj(w_n)
        k_j = kj(w_n)
        K_0 = np.power(K0(w_n, w_m, sign_value), 2)
        M_2 = M2(w_n, w_m, sign_value)
        c_00 = np.power(w_n, 2) - np.power(w_n+w_m, 2)

        for i in range(modes+1):
            a = c_j[i]
            b = np.power(k_j[i], 2) / (np.power(k_j[i],2) - K_0)
            c = c_00 + M_2[i]
            sum_ += a*b*c

        return sum_
    
    def SUM2(w_n, w_m):
        sum_ = 0.0

        c_j = cj(w_m)
        k_j = kj(w_m)
        K_0 = np.power(K0(w_m, w_n, sign_value), 2)
        M_2 = M2(w_m, w_n, sign_value)
        c_00 = np.power(w_m, 2) - np.power(w_n+w_m, 2)

        for i in range(modes+1):
            a = c_j[i]
            b = np.power(k_j[i], 2) / (np.power(k_j[i],2) - K_0)
            c = c_00 + M_2[i]
            sum_ += a*b*c

        return sum_

    def SUM3(w_n, w_m):
        sum_ = 0

        c_j_w_n = cj(w_n)
        c_j_w_m = cj(w_m)
        k_j_w_n = kj(w_n)
        k_j_w_m = kj(w_m)
        K_0 = np.power(K0(w_n, w_m, sign_value), 2)

        for i in range(modes+1):
            for l in range (0,modes):
                a = c_j_w_n[i] * c_j_w_m[l]
                b = (k_j_w_n[i] + k_j_w_m[l]) / (np.power(k_j_w_n[i] + k_j_w_m[l], 2) - K_0)
                c = Hp(w_n, w_m, i, l)
                sum_ += a*b*c

        return sum_
    
    result = E(w_n, w_m, sign_value) * (-sign_value*(g/(2.0*w_n)) * SUM1(w_n, w_m) \
                                    -sign_value*(g/(2.0*w_m)) * SUM2(w_n, w_m) \
                                    + SUM3(w_n, w_m))
    result = result/delta(w_n, w_m) # similar to Schaffer (1996)

    return result

#DETERMINAÇÃO DOS PARES (bastante incompleto)
    
w=np.linspace(1,2,11) 
res = list(combinations(w, 2))
pares=np.array(res, dtype = 'complex_')
for i in w:
    pares=np.append(pares,[[i,i]],axis=0)
    

for i in pares:
    wn=i[0]
    wm=i[1]


#CÁLCULO PARA COMPUTAR ELEVAÇÃO DE PRIMEIRA E SEGUNDA ORDEM:

    
def Lp(w1,w2,j,l):
    return 0.5*((g**2)*kj(w1)[j]*kj(w2)[l]/(w1*w2)-w1*w2-(w1**2+w2**2))

def Ll(w1,w2,j,l):
    return 0.5*((g**2)*kj(w1)[j]*np.conjugate(kj(w2)[l])/(w1*w2)+w1*w2-(w1**2+w2**2))

def Dp(w1,w2,j,l):
    d=g*(kj(w1)[j]+kj(w2)[l])*np.tanh((kj(w1)[j]+kj(w2)[l])*h)-(w1+w2)**2
    return d

def Dl(w1,w2,j,l):
    d=g*(kj(w1)[j]-np.conjugate(kj(w2)[l]))*np.tanh((kj(w1)[j]-np.conjugate(kj(w2)[l]))*h)-(w1-w2)**2
    return d

def Gp(w1,w2,j,l):
    chaves=(w1+w2)*Hp(w1,w2,j,l)/Dp(w1,w2,j,l)-Lp(w1,w2,j,l)
    return delta(w1,w2)*chaves/g

def Gl(w1,w2,j,l):
    chaves=(w1-w2)*Hl(w1,w2,j,l)/Dl(w1,w2,j,l)-Ll(w1,w2,j,l)
    return delta(w1,w2)*chaves/g






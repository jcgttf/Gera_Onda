import math
import numpy as np
from itertools import combinations
from mpmath import *
import cmath

g = 9.81 # acceleration due to gravity in m/s^2
h = 1 # water depth in m
modes=20
d=0.3
l=-0.3
epsilon=0.000000000000000001



def kj(w):
    kjlist=[]
    def solving_for_wavenumber(w):
        def func(k):
            return (1/k)-(g/w**2)*math.tanh(k*h)
        
        def deriv(k):
            return -(1/k**2)-((g/w**2)*h*(1-math.tanh(k*h)**2))
        
        def newton_raphson1(func, deriv, k0, accuracy, max_iter):
            x = k0
            for i in range(max_iter):
                x_new = x - func(x) / deriv(x)
                if abs(x_new - x) < accuracy:
                    return x_new
                x = x_new
            raise ValueError(f'The equation failed to converge after {max_iter} iterations')
        
        accuracy = 0.00001 # desired level of accuracy
        max_iter = 5000 # maximum number of iterations
        k0 = 0.10 # initial guess for the root
        
        return newton_raphson1(func, deriv, k0, accuracy, max_iter)
    wavenumber = solving_for_wavenumber(w)
    kjlist.append(wavenumber)
    
    for i in range(1,modes):
        def solving_for_imaginarywavenumber(w):
                def func(k):
                    return math.atan(-w**2/(g*k))-k*h-(math.pi*i/h)
                
                def deriv(k):
                    return ((w**2/(g*k**2))/(1+(w**2/(g*k))**2))-h
                
                def newton_raphson1(func, deriv, k0, accuracy, max_iter):
                    x = k0
                    for i in range(max_iter):
                        x_new = x - func(x) / deriv(x)
                        if abs(x_new - x) < accuracy:
                            return x_new
                        x = x_new
                    raise ValueError(f'The equation failed to converge after {max_iter} iterations')
                
                accuracy = 0.00001 # desired level of accuracy
                max_iter = 5000 # maximum number of iterations
                k0 = 2 # initial guess for the root
                
                return newton_raphson1(func, deriv, k0, accuracy, max_iter)
            
        wavenumber = solving_for_imaginarywavenumber(w)
        kjlist.append(complex(0,wavenumber))
        kj=np.array(kjlist)

    return(kj)

def Dj(w):
    
    return(((kj(w)*h)/2)*(((kj(w)*h)/(np.sinh(kj(w)*h)*np.cosh(kj(w)*h)))+1))

def cj(w):

    return((w**2*h/g)-(h/(h+l)))*(1/Dj(w))+(h/(h+l))*(1/Dj(w))*(np.cosh(kj(w)*d)/np.cosh(kj(w)*h))

    
def K0p(w1,w2):
    return kj(w1+w2)[0]

def K0l(w1,w2):
    return kj(w1-w2)[0]


def Hp(w1,w2,j,l):
    return (w1+w2)*(w1*w2-g**2*kj(w1)[j]*kj(w2)[l]/(w1*w2))+(w1**3+w2**3)/2-((g**2)/2)*(kj(w1)[j]**2/w1+kj(w2)[l]**2/w2)

def Hl(w1,w2,j,l):
    return ((w1-w2)*(-w1*w2-g**2*kj(w1)[j]*np.conjugate(kj(w2)[l])/(w1*w2))+(w1**3-w2**3)/2-((g**2)/2)*(kj(w1)[j]**2/w1-kj(w2)[l]**2/w2))   

#Esses H estão dando valores muito altos, mas enfim, continuamos.

def M1p(w1,w2):
    
    return ((1/(h+l))*(g/((w1+w2)**2))*((cmath.cosh(K0p(w1,w2)*d)/cmath.cosh(K0p(w1,w2)*h))-1))

def M1l(w1,w2):
    return ((1/(h+l))*(g/(epsilon+((w1-w2)**2)))*((cmath.cosh(K0l(w1,w2)*d)/cmath.cosh(K0l(w1,w2)*h))-1))


def delta(w1,w2):
    if w1==w2:
        return 0.5
    else:
        return 1
    
def Ep(w1,w2):
    return ((delta(w1,w2)*K0p(w1,w2)**2*h)/(cj(w1)[0]*cj(w2)[0]*(w1+w2)**3*(1+M1p(w1,w2))))

def El(w1,w2):
    return ((delta(w1,w2)*K0l(w1,w2)**2*h)/(cj(w1)[0]*cj(w2)[0]*(w1-w2)**3*(1+M1l(w1,w2))))

def M2p(w1,w2):
    
    Ch1p=2*kj(w1)*K0p(w1,w2)*(1-(np.cosh(kj(w1)*d)*np.cosh(K0p(w1,w2)*d))/(np.cosh(kj(w1)*h)*np.cosh(K0p(w1,w2)*h)))
    Ch2p=-(kj(w1)**2+K0p(w1,w2)**2)*(((w1**2*(w1+w2)**2)/(g**2*kj(w1)*K0p(w1,w2)))-(np.sinh(kj(w1)*d)*np.sinh(K0p(w1,w2)*d))/(np.cosh(kj(w1)*h)*np.cosh(K0p(w1,w2)*h)))
    return (-(g/(h+l))*((K0p(w1,w2)/kj(w1))/(kj(w1)**2-K0p(w1,w2)**2))*(Ch1p+Ch2p))

def M2l(w1,w2):

    Ch1l=2*kj(w1)*K0l(w1,w2)*(1-(np.cosh(kj(w1)*d)*np.cosh(K0l(w1,w2)*d))/(np.cosh(kj(w1)*h)*np.cosh(K0l(w1,w2)*h)))
    Ch2l=-(kj(w1)**2+K0l(w1,w2)**2)*(((w1**2*(w1-w2)**2)/(g**2*kj(w1)*K0l(w1,w2)))-(np.sinh(kj(w1)*d)*np.sinh(K0l(w1,w2)*d))/(np.cosh(kj(w1)*h)*np.cosh(K0l(w1,w2)*h)))
    return (-(g/(h+l))*((K0l(w1,w2)/kj(w1))/(kj(w1)**2-K0l(w1,w2)**2))*(Ch1l+Ch2l))




def somatorio1(w1,w2):
        soma=0
        for i in range (0,modes):
            a=cj(w1)[i]
            b=kj(w1)[i]**2/(kj(w1)[i]**2-K0l(w1,w2)**2)
            c=w1**2-(w1-w2)**2+M2l(w1,w2)[i]
            d=cj(w2)[i]
            e=kj(w2)[i]**2/(kj(w2)[i]**2-K0l(w2,w1)**2)
            f=w2**2-(w2-w1)**2+M2l(w2,w1)[i]
            soma=a*b*c+d*e*f+soma
        return soma
def parte1(w1,w2):
        return (0.5*g/w1)*somatorio1(w1,w2)
    
def parte2(w1,w2):
        return parte1(w2,w1)

def somatorioduplo(w1,w2):
        soma=0
        for i in range(0,modes):
            for l in range (0,modes):
                a=cj(w1)[i]*np.conjugate(cj(w2)[l])
                b=(kj(w1)[i]-np.conjugate(kj(w2)[l]))/((kj(w1)[i]-np.conjugate(kj(w2)[l]))**2-K0l(w1,w2)**2)
                c=Hl(w1,w2,i,l)
                soma=soma+a*b*c
        return soma

    
def Fp(w1,w2):
    def somatorio1(w1,w2):
        soma=0
        for i in range (0,modes):
            a=cj(w1)[i]
            b=kj(w1)[i]**2/(kj(w1)[i]**2-K0p(w1,w2)**2)
            c=w1**2-(w1+w2)**2+M2p(w1,w2)[i]
            soma=a*b*c+soma
        return soma
    def parte1(w1,w2):
        return (-0.5*g/w1)*somatorio1(w1,w2)
    
    def parte2(w1,w2):
        return parte1(w2,w1)

    def somatorioduplo(w1,w2):
        soma=0
        for i in range(0,modes):
            for l in range (0,modes):
                a=cj(w1)[i]*cj(w2)[l]
                b=(kj(w1)[i]+kj(w2)[l])/((kj(w1)[i]+kj(w2)[l])**2-K0p(w1,w2)**2)
                c=Hp(w1,w2,i,l)
                soma=soma+a*b*c
        return soma
    
    return Ep(w1,w2)*(parte1(w1,w2)+parte2(w1,w2)+somatorioduplo(w1,w2))

def Fl(w1,w2):
    def somatorio1(w1,w2):
        soma=0
        for i in range (0,modes):
            a=cj(w1)[i]
            b=kj(w1)[i]**2/(kj(w1)[i]**2-K0l(w1,w2)**2)
            c=w1**2-(w1-w2)**2+M2l(w1,w2)[i]
            soma=a*b*c+soma
        return soma
    def parte1(w1,w2):
        return (0.5*g/w1)*somatorio1(w1,w2)
    
    def parte2(w1,w2):
        return parte1(w2,w1)

    def somatorioduplo(w1,w2):
        soma=0
        for i in range(0,modes):
            for l in range (0,modes):
                a=cj(w1)[i]*np.conjugate(cj(w2)[l])
                b=(kj(w1)[i]-np.conjugate(kj(w2)[l]))/((kj(w1)[i]-np.conjugate(kj(w2)[l]))**2-K0l(w1,w2)**2)
                c=Hl(w1,w2,i,l)
                soma=soma+a*b*c
        return soma
    
    return El(w1,w2)*(parte1(w1,w2)+np.conjugate(parte2(w1,w2))+somatorioduplo(w1,w2))


#DETERMINAÇÃO DOS PARES (bastante incompleto)
    
w=np.linspace(1,2,11) 
res = list(combinations(w, 2))
pares=np.array(res)
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






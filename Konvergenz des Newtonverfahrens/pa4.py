#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

"""
Created on Wed Jan 20 17:40:02 2021

@author: jamey
"""
PREC = 10**(-4)

def norm(x): #eukl. norm R^n
    if isinstance(x, (int, float,complex)):
        return abs(x)
    return np.sqrt(sum([(x[i])**2 for i in range(len(x))]))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def newton(F, dF, x0, delta = PREC, epsilon = PREC, maxIter = 100):
    x = x0
    
    for i in range(maxIter):
        y = np.copy(x) #vorheriges folgenglied

        dx = solve(dF(x), -F(x))
        if len(dx) == 1:
            dx = dx[0]
        
        x += dx

        if norm(x-y) < delta or norm(F(x)) < epsilon:
            return x
    return x

def exercise2():
    prec = 10*(-10)
    miter = 50

    f  = lambda x: x**3 - 2*x
    df = lambda x: 3**(x**2) - 2 

    roots = []
    for startwert in [0.1, 2, -2]:
        roots.append(newton(f, df, startwert, prec, prec, miter))

    t = np.arange(-2, 2, 0.1)
    plt.plot(t, f(t))
    plt.scatter(roots, [0, 0, 0])
    plt.show()
    return

def exercise3():
    g  = lambda x: np.array([x[0]**2+x[1]**2-6*x[0], (3/4)*np.exp(-x[0])-x[1]])
    dg = lambda x: np.array([[2*x[0]-6, 2*x[1]], [(-3/4)*np.exp(-x[0]), -1]])

    root = newton(g, dg, np.array([0.08, 0.7]))
    print(g(root))
    return

#def means512():
#    means = np.array([i*(/512+j/ for i in range(512)])
    #print(means)
    
    
    #return

def exercise4():
    h  = lambda x: x**3 -1
    dh = lambda x: 3*x**2

def toroot(x,roots,epsilon):
    #Es sollte darauf geachtet werden, dass epsilon kleiner als
    #Hälfte des Abstandes der Nullstellen ist, die Bindungung
    #nie für mehrere Nullstellen erfüllt werden kann.
    for i in range(len(roots)):
        if abs(x.real-roots[i].real)<epsilon and abs(x.imag-roots[i].imag)<epsilon:
            return i
    return len(roots)+1


def aufgabe4():
    #Bestimmen des diskreten Balles mit Radius 1 um die Null (bzg der unendlichnorm)
    H = np.linspace(start=-1, stop=1, num=512, endpoint=True)
    B = np.array([[complex(re, im) for re in H] for im in H], dtype=complex)

    f  = lambda x: x**3-1
    df = lambda x: 3*x**2

    #Die Nullstellen mit Hand ausgerechnet
    roots = [complex(-1/2,(np.sqrt(3))/2),complex(-1/2,-(np.sqrt(3))/2), complex(1,0)]

    #Anwenden des Newton verfahrens für alle Punkte im Ball
    V = np.empty_like(B)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i][j]=newton(f,df,B[i][j],delta=10**(-5),epsilon=10**(-5), maxIter=100)

    #Erstellen einer Farbkarte jenachdem ob und gegen welch Nullstelle das Verfahren
    #konvergiert ist. toroot gibt diese konvergenz an dabei gilt der wert als konvergiert
    #wenn der abstand zur einer Nullstelle kleiner als der dritte Übergabeparameter ist.
    C = np.ndarray(shape=V.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i][j] = toroot(V[i][j],roots,10**(-3))

    plt.imshow(C)
    plt.show()


def phase(z):
    x,y = z.real,z.imag
    if x>0:
        phi = np.arctan(y/x)%np.pi
    if x<0:
        phi = -np.arctan(y/x)%np.pi
    if x==0:
        if y>0:
            phi = np.pi/2
        if y<0:
            phi = -np.pi/2
    print(phi)
    if not (-np.pi<=phi and phi<=np.pi):
        print('damn fuck {}'.format(phi))
    return phi

def aufgabe5():
    H = np.linspace(start=-1, stop=1, num=512, endpoint=True)
    B = np.array([[complex(re, im) for re in H] for im in H], dtype=complex)

    f = lambda x: x ** 5 - 1
    df = lambda x: 5 * x ** 4

    roots = [complex(-0.809017,-0.587785), complex(-0.809017,0.587785),
             complex(0.309017,-0.951057),complex(0.309017, 0.981057),complex(1,0)]

    V = np.empty_like(B)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i][j] = newton(f, df, B[i][j],delta=10**(-14),epsilon=10**(-14),maxIter=5)

    Phase = np.empty(shape=V.shape)
    for i in range(Phase.shape[0]):
        for j in range(Phase.shape[1]):
            #Phase[i][j] = phase(V[i][j])
            #Phase[i][j] = np.arctan2(V[i][j].imag,V[i][j].real)
            phi = np.angle(V[i][j])
            if phi == np.pi:
                Phase[i][j]=-phi
            else:
                Phase[i][j]=phi
    plt.imshow(Phase)
    plt.show()

    #C ist die colourmap zur Visualisierung
    C = np.ndarray(shape=V.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i][j] = toroot(V[i][j], roots, 10 ** (-1))

    #plt.imshow(C)
    #plt.show()


#means512()
#exercise2()
#aufgabe4()
aufgabe5()





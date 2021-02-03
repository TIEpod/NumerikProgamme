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
    print("Die Nullstellen nach dem Newton-Verfahren sind %f und %f" % (g(root)[0], g(root)[1]))
    return


def toroot(x,roots,epsilon):
    '''Überprüft für einen Wert x ob dieser eine Nähe epsilon zu einer der
     Nullstellen aus roots hat. Theoretisch sagt das natürlich nicht viel über
     die Konvergenz. Aber wir wisses das in einer kleinen Umgebung um eine
     Nullstelle (hier) die Werte auch gegen dies konvergieren.'''
    #Es sollte darauf geachtet werden, dass epsilon kleiner als
    #Hälfte des Abstandes der Nullstellen ist, damit die Bedingung
    #nie für mehrere Nullstellen erfüllt werden kann.
    for i in range(len(roots)):
        if abs(x.real-roots[i].real)<epsilon and abs(x.imag-roots[i].imag)<epsilon:
            return i
    return len(roots)+1


def aufgabe4():
    #Bestimmen des diskreten Ball mit Radius 1 um die Null (bzg der unendlichnorm alse quadratisch)
    #Aquidistante Zerlegung in x bwz y Richtung
    H = np.linspace(start=-1, stop=1, num=512, endpoint=True)
    B = np.array([[complex(re, im) for re in H] for im in H], dtype=complex)

    #Funktion und ihre Ableitung
    f  = lambda x: x**3-1
    df = lambda x: 3*x**2

    #Die Nullstellen mit Hand ausgerechnet.
    roots = [complex(-1/2,(np.sqrt(3))/2),complex(-1/2,-(np.sqrt(3))/2), complex(1,0)]

    #Anwenden des Newton-Verfahrens für alle Punkte im Ball
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
    plt.title("Aufgabe 4")
    plt.show()


def phase(z):
    '''Bestimmt für eine komplexe Zahl den Winkel Phi für x+iy=re^(i*phi) und r=|z|
    Dabei ist phi aus [-pi,pi) eindeutig.'''
    x,y = z.real,z.imag
    if x>0:
        phi = np.arctan(y/x)
    if x<0 and y>0:
        phi = np.arctan(y/x)+np.pi
    if x<0 and y<=0:
        phi = np.arctan(y/x)-np.pi
    if x==0:
        if y>0:
            phi = np.pi/2
        if y<0:
            phi = -np.pi/2
    return phi

def aufgabe5():
    ''''''
    #Wie in 4
    H = np.linspace(start=-1, stop=1, num=512, endpoint=True)
    #B ist das Gitternetz
    B = np.array([[complex(re, im) for re in H] for im in H], dtype=complex)

    f = lambda x: x ** 5 - 1
    df = lambda x: 5 * x ** 4

    # Der letzte Newton iterator für jeden Gitterpunkt
    V = np.empty_like(B)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i][j] = newton(f, df, B[i][j], delta=10**(-14), epsilon=10**(-14), maxIter=10)

    # Die Phase jedes dieser Punkte
    Phase = np.empty(shape=V.shape)
    for i in range(Phase.shape[0]):
        for j in range(Phase.shape[1]):
            Phase[i][j] = phase(V[i][j])
            #Phase[i][j] = np.angle(V[i][j])

    #hsv ist eine Circle Farbkarte, sodass -pi und pi die gleiche Farbe haben.
    plt.imshow(Phase, cmap='hsv')
    plt.title("Aufabe 5")
    plt.show()

def mini(dF, hF, x0, delta=PREC, epsilon=PREC, maxIter=100):
    """Macht genau das gleiche wie die Funktion opt speichert nur auch die zwischen
    Schritt in IterVal"""
    x = x0
    IterVal = []
    for i in range(maxIter):
        IterVal.append(x)
        y = np.copy(x)  # vorheriges folgenglied

        x = x - np.linalg.inv(hF(x)) @ dF(x)
        if norm(x - y) < delta or norm(dF(x)) < epsilon:
            return x, IterVal
    return x, IterVal

def z_function(x,y):
    """Die Funktion aus Aufgabe 6"""
    return (x+1)**4+(y-1)**4

def aufgabe6():
    f = lambda x: (x[0] + 1) ** 4 + (x[1] - 1) ** 4
    df = lambda x: 4 * np.array([(x[0] + 1) ** 3, (x[1] + 1) ** 3])
    hf = lambda x: 12 * np.array([[(x[0] + 1) ** 2, 0], [0, (x[1] - 1) ** 2]])  # pos def
    x0 = [-1.1, 1.1]

    # in ItrVal stehen die die x^(k) für alle Iterationsschritte
    m, ItrVal = mini(df, hf, x0)

    x = np.linspace(-1.5,-0.5,60)
    y = np.linspace(0.5, 1.5,60)

    #Plotten des Graphen
    ax = plt.axes(projection="3d")
    plt.title("Aufgabe 6")
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y)
    ax.plot_surface(X, Y, Z)

    #Beschränken der Achsen. Beim scattern verschiebt sich sonst
    #später vielleich etwas
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = ax.get_xbound(), ax.get_ybound(), ax.get_zbound()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
    ax.set_xlabel("x-Achse")
    ax.set_ylabel("y-Achse")
    ax.set_zlabel("z-Achse")

    #Plotten der konvergenz Punkte lassen sich nur schwer sehen.
    #Besser wenn man den Graphen von unten betrachtet.
    A = np.array([x[0] for x in ItrVal])
    B = np.array([x[1] for x in ItrVal])
    A, B = np.meshgrid(A,B)
    C = z_function(A,B)
    ax.scatter(A, B, C, color='forestgreen', marker='o')

    plt.show()


def exercise7():

    xH = np.linspace(start=-1.5, stop=.5, num=1024, endpoint=True)
    yH = np.linspace(start=-1, stop=1, num=1024, endpoint=True)

    M = np.array([[complex(re, im) for re in xH] for im in yH], dtype=complex)

    C = np.zeros(M.shape)  # anz wo ||<=2
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            count = 0
            c = M[i][j]
            x = M[i][j]
            seq = lambda z: z ** 2 +c
            for k in range(256):
                if abs(x) <= 2:
                    count += 1
                x=seq(x)

            C[i][j] = count

    plt.imshow(C)
    plt.title("Aufgabe 7")
    plt.show()
import numpy as np
import scipy.sparse as sparse
from skimage.io import imread
from skimage import data, io
import matplotlib.pyplot as plt

from scipy.sparse import linalg

#Erste komponte: x, N;  Zweite: y, M


A=np.random.randint(5, size=(5, 7))

bear=imread('bear.jpg')
bird=imread('bird.jpg')
plane=imread('plane.jpg')
water=imread('water.jpg')


## AUFGABE 1 ##

def laplace_operator(N, M):
    # D_M ist eine sparse MxM-Marix mit -2 auf der Diagonalen und 
    # 1sen auf den Nebendiagonalen.

    E = np.ones(max(N,M))
    D_M = sparse.dia_matrix(([E, (-2)*E, E], [-1, 0, 1]), shape=(M, M))
    D_N = sparse.dia_matrix(([E, (-2) * E, E], [-1, 0, 1]), shape=(N, N))

    #spra.eye ist 1-Matrix in (sparse.) dia_matrix Format
    #spar.kron ist das Kronecker-Produkt für Sparse-Matrizen
    
    return sparse.kron(sparse.eye(M), D_N) + sparse.kron(D_M, sparse.eye(N))

def show_laplace():
    plt.figure("Aufgabe 1")
    io.imshow(laplace_operator(5, 7).todense())
    plt.show()


## AUFGABE 2 ##

def vec(f): #gibt spaltenvektor
    return f.T.flatten().T

def zeroed_laplacian(N, M):
    #Überlegung: Da wir die Spalten zu Null-Spalten machen möchten, müsste es
    #bei Sparse matrizen reichen indizes zu löschen.
    #DeltaZ = Laplace_Operator(N, M).tolil()

    A = np.zeros(N)
    B = np.zeros(N)
    B[0] =1
    B[-1]=1
    
    dia = np.array([])
    for i in range(M-2):
        dia = np.concatenate((dia, B))

    dia = np.concatenate((A, dia, A)) #Diagonale
    return sparse.dia_matrix((dia, [0]), shape=(N*M,N*M))


def compute_f(fomega_, g):
    N, M = g.shape
    
    D = laplace_operator(N, M)
    Z = zeroed_laplacian(N, M)

    D2 = D@Z #kennen
    D1 = D - D2
    
    A = D2
    b = D@vec(g)-D1@vec(fomega_)
    
    #print(A.todense())

    return sparse.linalg.cg(A, b, maxiter=20)[0].reshape(M,N).T

def insert(f, g, pos):
    '''Insert g in f. '''
    n = f.copy() 
    x, y = pos
    if x + len(g[0]) > len(f[0]) or y + len(g) > len(f):
        raise ValueError("The picture cannot be fit at this position or is too large. ")

    for i in range(len(g)):
        for j in range(len(g[0])):
            n[i+x][j+y] = g[i][j]
    return n

def seamless_grayclone(f_, g, pos):
    x, y = pos
    N, M = g.shape
    
    fomega_ = f_[x:x+N, y:y+M] #f* eingeschränkt auf omega
    f = compute_f(fomega_, g)

    return insert(f_, f, pos)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ausgabe(L, R):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(L)
    ax2.imshow(R)
    ax1.set_title("Original")
    ax2.set_title("Cloned")

def test():
    g = bear[0:3, 0:3, 0]
    f_ = water[0:3, 0:3, 0]

    seamless_grayclone(f_, g, (0,0))


def main():
    show_laplace()
    
    red_bear = bear[:, :, 0]
    red_water = water[:, :, 0]

    ausgabe(insert(red_water, red_bear, (30,20)), 
            seamless_grayclone(red_water, red_bear, (30,20)))

    return

#test()
main()




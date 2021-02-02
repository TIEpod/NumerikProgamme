import numpy as np
import skimage
from scipy import misc
from skimage import data, io
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import scipy.sparse as spar
from scipy.sparse import linalg
import time
import math

#version1.1 vom Dec 12th, 2020

A=np.random.randint(5, size=(5, 7))

image_bear=imread('bear.jpg')
image_bird=imread('bird.jpg')
image_plane=imread('plane.jpg')
image_water=imread('water.jpg')
#io.imshow(image_water)
#io.show()
#io.imshow(image_bear)
#io.show()


## AUFGABE 1 ##
def laplace_operator(N, M):
    # D_M ist eine sparse MxM-Marix mit -2 auf der Diagonalen und
    # 1sen auf den Nebendiagonalen.

    E = np.ones(max(N, M))
    D_M = spar.dia_matrix(([E, (-2) * E, E], [-1, 0, 1]), shape=(M, M))
    D_N = spar.dia_matrix(([E, (-2) * E, E], [-1, 0, 1]), shape=(N, N))

    # spra.eye ist 1-Matrix in (sparse.) dia_matrix Format
    # spar.kron ist das Kronecker-Produkt für Sparse-Matrizen

    return spar.kron(spar.eye(M), D_N) + spar.kron(D_M, spar.eye(N))

def Laplace_Operator(N,M):
    start=time.time()
    # I_M ist eine sparse Marix der Größe MxM. Auf der Diagonalen stehen -2 en und auf den beiden
    # Nebendiagonalen 1 sen.
    # Mit print(I_M.todense()) kann die Matrix in vollem Format zurück gegeben werdend.
    EM = np.ones(M)
    EN = np.ones(N)
    D_M = spar.dia_matrix(([EM, (-2)*EM, EM], [-1, 0, 1]), shape=(M, M))
    D_N = spar.dia_matrix(([EN, (-2) * EN, EN], [-1, 0, 1]), shape=(N, N))

    #spra.eye ist Einheitsmatrix in (sparse.) dia_matrix Format
    #spar.kron ist das Kronecher-Produkt für Sparse Matrizen
    Delta = spar.kron(spar.eye(M), D_N) + spar.kron(D_M, spar.eye(N))
    ende=time.time()
    plt.figure('Laplace Operator')
    io.imshow(Delta.todense())
    plt.show()
    print('Laufzeit von Laplace_Operator '+'{:5.3f}s'.format(ende-start))
    return Delta
print(Laplace_Operator(5,7).todense())

## AUFGABE 2 ##

def vec(f):
    start=time.time()
    col=f.T.flatten().T
    ende=time.time()
    print('Laufzeit von vec '+'{:5.3f}s'.format(ende-start))
    return col

#f=np.arange(1,17).reshape((4,4))
#print(f)
#print(vec(f))
#print(vec(f).shape)

def diagonalenExtrahieren2(A,N,M):
    A=A.todia()
    diag=A.data
    offset=A.offsets
    for i in range(0,N):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=math.inf
            if 0<=i-offset[m]<N*M:
                diag[m][i] = math.inf
    for i in range(N*(M-1),N*M):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=math.inf
            if 0<=i-offset[m]<N*M:
                diag[m][i] = math.inf
    for i in range(N, N*M, N):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=math.inf
            if 0<= (i-1)+offset[m]<N*M:
                diag[m][i-1+offset[m]]=math.inf
            if 0 <= i - offset[m] <= N * M:
                diag[m][i] = math.inf
            if 0 <= i-1 - offset[m] <= N * M:
                diag[m][i-1] = math.inf
    diagnex = np.zeros(shape=(len(offset),1 ))
    for m in range(len(diag)):
        v = diag[m][diag[m] != math.inf]
        diagnex[m]=v
    Anew=spar.dia_matrix((diagnex,offset), shape=((N-2)*(M-2),(N-2)*(M-2)))
    io.imshow(Anew.todense())
    io.show()
    return Anew

def diagonalenExtrahieren(A,N,M):
    A=A.todia()
    diag=A.data
    offset=A.offsets
    for i in range(0,N):
        for m in range(len(offset)):
            #try:
             #   diag[m][i+offset[m]]=0
            #except:
            #    None
            if 0<=i-offset[m]<N*M:
                diag[m][i] = 0
    for i in range(N*(M-1),N*M):
        for m in range(len(offset)):
            #try:
             #   diag[m][i+offset[m]]=0
            #except:
            #    None
            if 0<=i-offset[m]<N*M:
                diag[m][i] = 0
    for i in range(N, N*M, N):
        for m in range(len(offset)):
            #try:
             #   diag[m][i+offset[m]]=0
              #  diag[m][i-1+offset[m]]=0
            #except:
             #   None
            if 0 <= i - offset[m] <= N * M:
                diag[m][i] = 0
            if 0 <= i-1 - offset[m] <= N * M:
                diag[m][i-1] = 0
    Anew = spar.dia_matrix((diag, offset), shape=(N * M, N * M))
    io.imshow(Anew.todense())
    io.show()
    return Anew
diagonalenExtrahieren2(Laplace_Operator(5,7),5,7)
def zeroed_laplacian(Delta,M, N):
    #Überlegung: Da wir die Spalten zu Null spalten machen möchten, müsste es
    #bei Sparse matrizen reichen indizes zu löschen.
    #DeltaZ = Laplace_Operator(N, M).tolil()

    #entnehmen die Diagonelen (diag aus Delta.data) position der Diagonale ist offset
    #ändern dort die Einträge zu null.
    start=time.time()
    Delta=Delta.todia()
    #io.imshow(Delta.todense())
    #io.show()
    diag = Delta.data
    offset = Delta.offsets
    for i in range(0, len(offset)):
        for j in range(len(diag[i])):
            for k in range(0, N):
                diag[i][k] = 0
            for k in range(N*(M-1), M*N):
                diag[i][k] = 0
            for k in range(1, M - 1):
                diag[i][k*N+offset(i)] = 0
                diag[i][k*N-1] = 0
    D1 = spar.dia_matrix((diag, offset), shape=(N*M,N*M))
    ende=time.time()
    print('Laufzeit von zeroed_laplacian ' + '{:5.3f}s'.format(ende - start))
    return D1
#io.imshow(Laplace_Operator(15,25).todense())
#io.show()
#io.imshow(zeroed_laplacian([0, 1], 12, 14).todense())
#io.show()

#print(Laplace_Operator(9,9).todense())
#print(zeroed_laplacian(9,9).todense())

def berechnug_von_f(fomega_,g,N,M):
    vf_ = vec(fomega_)
    vg = vec(g)

    #Wir zerlegen Den Laplaceoperator in D=(D1+D2)
    #In D1 stehen alle unbekannten Spalten (das innere von Omega)
    #In D2 stehen alle bekannten Spalten (Rand von Omega)
    D = Laplace_Operator(N,M)
    #io.imshow(D.todense())
    #io.show()
    D1=diagonalenExtrahieren(D,N,M)
    start1=time.time()
    D2=D-D1
    ende1=time.time()
    print('Laufzeit von D2 ' + '{:5.3f}s'.format(ende1 - start1))

    D =D.todense()
    start2 = time.time()
    b = (D.dot(vg)-D2.dot(vf_)).T
    ende2 = time.time()
    print('Laufzeit von b ' + '{:5.3f}s'.format(ende2 - start2))

    start3=time.time()
    f = linalg.cg(D1,b,maxiter=20)
    ende3 = time.time()
    print('Laufzeit von cg verfahren' + '{:5.3f}s'.format(ende3 - start3))

    for i in range(len(f[0])):
        if f[0][i]>255:
            f[0][i]=255
        elif f[0][i]<0:
            f[0][i]=0
    return f[0].reshape((N,M),order='F')


def insert(f, g, pos):
    '''Insert g in f. '''
    
    n=f.copy() 
    x,y=pos
    #if x+len(g[0])>len(f[0]) or y+len(g)>len(f):
    #   raise ValueError("The picture cannot be fit at this position or is too large. ")

    for i in range(len(g)):
        for j in range(len(g[0])):
            n[i+x][j+y] = g[i][j]
    return n

def seamless_grayclone(f_, g, pos):
    #f_=f_u[::,::,0]
    #g=gu[::,::,0]
    #io.imshow(g)
    #io.show()
    x,y=pos
    N, M = g.shape[0], g.shape[1]
    #Schneiden den relevanten Teil aus f_ aus.
    fomega_ = f_[x:x+N, y:y+M]
    f = berechnug_von_f(fomega_,g,N,M)
    result = insert(f_,f,pos)
    return result

#v=seamless_grayclone(image_water,image_bear,(50,10))
#io.imshow(v)
#io.show()

#false2(30,40)
def seamless_cloning(f_, g, pos):
    '''Fügt g in f. Mit differenzengleichung. '''

    for i in range(0,3):
        f_gray=f_[::,::,i]
        ggray=g[::,::,i]
        seamless_grayclone(f_gray,ggray,pos)

    
    ##TODO
    
    return f

def ausgabe(L, R):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(L)
    ax2.imshow(R)
    ax1.set_title("Original")
    ax2.set_title("Cloned")

def main():
    print(A, end="\n\n")
    print(laplace(A))
    
    #grayclone
    ins=insert(image_water, image_bear, (20, 50))
    clone=seamless_grayclone(image_water, image_bear, (20, 50))
    ausgabe(ins, clone)

    
#main()
#io.imshow(image_water)
#io.show()
small_bear=image_bear[50:110, 30:110, 0] #Größe 60x190
small_water=image_water[150:350, 0:250, 0] #Größe 200x250
#io.imshow(insert(small_water,small_bear,(70,20)))
#io.show()
#io.imshow(seamless_grayclone(small_water,small_bear,(70,20)))
#io.show()

#io.imshow(bekanntenAnschauung(9,9).todense())
#io.show()

def innen(D,N,M):
    D=D.tocsc()
    for i in range(0,N):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=math.inf
            if 0<=i-offset[m]<N*M:
                diag[m][i] = math.inf
    for i in range(N*(M-1),N*M):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=math.inf
            if 0<=i-offset[m]<N*M:
                diag[m][i] = math.inf
    for i in range(N, N*M, N):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=math.inf
            if 0<= (i-1)+offset[m]<N*M:
                diag[m][i-1+offset[m]]=math.inf
            if 0 <= i - offset[m] <= N * M:
                diag[m][i] = math.inf
            if 0 <= i-1 - offset[m] <= N * M:
                diag[m][i-1] = math.inf
    diagnex = np.zeros(shape=(len(offset),1 ))
    for m in range(len(diag)):
        v = diag[m][diag[m] != math.inf]
        diagnex[m]=v
    Anew=sparse.dia_matrix((diagnex, offset), shape=((N - 2) * (M - 2), (N - 2) * (M - 2)))
    io.imshow(Anew.todense())
    io.show()
    return Anew

import numpy as np
import skimage
import scipy
from scipy import misc
from skimage import data, io
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from scipy.sparse import linalg
import scipy.sparse as spar
from scipy.sparse import csr_matrix
import time

import sys  #Returs the size of an object
#Teil 1.

def Laplace_Operatoreeee(N,M):

    # I_M ist eine sparse Marix der Größe MxM. Auf der Diagonalen stehen -2 en und auf den beiden
    # Nebendiagonalen 1 sen.
    # Mit print(I_M.todense()) kann die Matrix in vollem Format zurück gegeben werdend.
    EM = np.ones(M)
    EN = np.ones(N)
    D_M = spar.dia_matrix(([EM, (-2)*EM, EM], [-1, 0, 1]), shape=(M, M))
    D_N = spar.dia_matrix(([EN, (-2) * EN, EN], [-1, 0, 1]), shape=(N, N))

    #spra.eye ist Einheitsmatrix in dia_matrix Format
    #spar.kron ist das Kronecher-Produkt für Sparse Matrizen
    Lap = spar.kron(spar.eye(M), D_N) + spar.kron(D_M, spar.eye(N))
    #Lap ist nun eine sparse Block matrix
    return Lap

#print(Laplace_Operator(3,4).todense())

def gray_cloning(f_,g):
    N, M = np.shape(g)
    # test v = np.arange(1, 26).reshape(5, 5)
    vec_g = g.flatten(order='F').reshape(25, 1)
    Nabla = Laplace_Operator(N, M)


    f = linalg.cg(Nabla, Nabla*vec_g)
    return f

def angepasste(N,M):
    N = 20
    g = np.arange(1, N * N + 1).reshape(N, N)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            g[i][j] = 0


def side_condition():
    N=20
    g=np.arange(1,N*N+1).reshape(N,N)
    for i in range(1,N-1):
        for j in range(1,N-1):
            g[i][j]=0
    print(g)

    print("Memory utilised (bytes): ", sys.getsizeof(g))
    f_=spar.csr_matrix(g)
    print("Memory utilised (bytes): ", sys.getsizeof(f_))
    f_.todia()
    print("Memory utilised (bytes): ", sys.getsizeof(f_))
    colx=f_[x:x+N][y]
    print(colx)
    coly=f_[x:x+N][y+M]
    print(coly)
#side_condition()

def false1(N, M):
    ENM = np.ones(M*N)
    for i in range(0,M-1):
        ENM[i]=0
    for i in range((N-1)*M,M*N):
        ENM[i]=0
    for i in range(1,N):
        ENM[i*M]=0
        ENM[i*M-1]=0

    D_NM = spar.dia_matrix(([ENM, (-4) * ENM, ENM], [-1, 0, 1]), shape=(N*M, N*M))
    return D_NM

#print(ange_Lap(4,4).todense())


def false2(N, M):
    Delta = Laplace_Operator(N, M)
    io.imshow(Delta.todense())
    io.show()

    # Wir entnehmen die diagonalen und passen diese an
    Diag = Delta.data
    diag_1, diag0, diag1 = Diag

    # wir
    for i in range(0, M - 1):
        diag_1[i] = 0
        diag0[i] = 0
        diag1[i] = 0
    for i in range((N - 1) * M, M * N):
        diag_1[i] = 0
        diag0[i] = 0
        diag1[i] = 0
    for i in range(1, N):
        diag_1[i * M] = 0
        diag_1[i * M - 1] = 0

        diag0[i * M] = 0
        diag0[i * M - 1] = 0

        diag1[i * M] = 0
        diag1[i * M - 1] = 0


#false2(6, 4)

#Nab_Null=spar.dia_matrix(([diag_1, diag0, diag1], [-1, 0, 1]), shape=(N*M, N*M))
 #   return Nab_Null

#print(false(4,4).todense())

#print(ange_Lap(4,4).todense())

def zeroed_laplacianeeee(N, M):
    #Überlegung: Da wir die Spalten zu Null spalten machen möchten, müsste es
    #bei Sparse matrizen reichen indizes zu löschen.
    DeltaZ = Laplace_Operator(N, M).tolil()
    for i in range(0,N*M):
        for j in range(0,M-1):
            DeltaZ[i,j] = 0
        for j in range((N-1)*M,M*N):
            DeltaZ[i,j] = 0
        for j in range(1, N):
            DeltaZ[i,j*M] = 0
            DeltaZ[i,j*M-1] = 0
    return DeltaZ
#io.imshow(Laplace_Operator(9,9).todense())
#io.show()
#io.imshow(zeroed_laplacian(9,9).todense())
#io.show()

#print(Laplace_Operator(9,9).todense())
#print(zeroed_laplacian(9,9).todense())
def vec(f):
    x=f.T.flatten().T
    return x
#f=np.arange(1,17).reshape((4,4))
#print(f)
#print(vec(f))
#print(vec(f).shape)
def bekannten(f_N,M,x,y):
    fomeag = f_[x:N+x][y:M+y]
    vf_ = vec(fomeag)
    Rest=Laplace_Operator(N,M)-zeroed_laplacian(N,M)
    return Rest

def bekanntenAnschaung(N,M):
    Rest = Laplace_Operator(N, M) - zeroed_laplacian(N, M)
    return Rest
#io.imshow(bekannten(9,9).todense())
#io.show()


def false3(grad_g,f_,N,M,x,y):
    b = np.array((N*M,1))
    fomeag = f_[x:N+x][y:M+y]
    vf_ = vec(fomeag)
    for j in range(0, M - 1):
        b[j] = vf_[j]
    for j in range((N - 1) * M, M * N):
        if j%M==0 or j%M==M-1:
            b[j] = grad_g
        else:
            b[j] = vf_[j]
    for j in range(1, N):
        b[j*M] = vf_[j*M]
        b[j*M-1] = vf_[j*1+1]

def Laplace_Operator(N,M):

    # I_M ist eine sparse Marix der Größe MxM. Auf der Diagonalen stehen -2 en und auf den beiden
    # Nebendiagonalen 1 sen.
    # Mit print(I_M.todense()) kann die Matrix in vollem Format zurück gegeben werdend.
    EM = np.ones(M)
    EN = np.ones(N)
    D_M = spar.dia_matrix(([EM, (-2)*EM, EM], [-1, 0, 1]), shape=(M, M))
    D_N = spar.dia_matrix(([EN, (-2) * EN, EN], [-1, 0, 1]), shape=(N, N))

    #spra.eye ist Einheitsmatrix in dia_matrix Format
    #spar.kron ist das Kronecher-Produkt für Sparse Matrizen
    Lap = spar.kron(spar.eye(M), D_N) + spar.kron(D_M, spar.eye(N))
    #Lap ist nun eine sparse Block matrix
    return Lap


def zeroed_laplacian(Delta,N, M):
    #Überlegung: Da wir die Spalten zu Null spalten machen möchten, müsste es
    #bei Sparse matrizen reichen indizes zu löschen.
    #DeltaZ = Laplace_Operator(N, M).tolil()

    #Wir Erstellen eine Einheitsmatrix, löschen in ihr die Null Spalten
    #und multiplizieren Diese an unsere ursprünliche matrix.
    start=time.time()
    ENM=np.ones(shape=(1,N*M))
    for i in range(0,N):
        ENM[0][i] = 0
    for i in range(N*(M-1),M*N):
        ENM[0][i] = 0
    for i in range(1, M):
        ENM[0][i*N] = 0
        ENM[0][i*N-1] = 0
    A=spar.dia_matrix((ENM,[0]), shape=(N*M,N*M))
    #io.imshow(A.todense())
    #io.show()
    Delta1=Detla.multiply(A)
    io.imshow(Delta1)
    io.show()
    #DeltaZ=spar.linalg.LinearOperator.matmat(Lap.todia(),A.todense())
    ende=time.time()
    print('Laufzeit von zeroed_laplacian ' + '{:5.3f}s'.format(ende - start))
    return Delta1
#io.imshow(Laplace_Operator(15,25).todense())
#io.show()

def matrixmultiplikation(N,M):
    Delta=Laplace_Operator(N,M)
    Delta=zeroed_laplacian(Delta,N,M)

    return Lap
#io.imshow(matrixmultiplikation(23,14).todense())
#io.show()
def new_laplacian(Delta,N,M):
    Delta.todia()
    diag = Delta.data
    offset = Delta.offsets
    for i in range(0, len(offset)):
        for j in range(len(diag[i])):
            for k in range(0, N - 1):
                diag[i][k] = 0
            for k in range(N * (M - 1), M * N):
                diag[i][k] = 0
            for k in range(1, M - 1):
                diag[i][k * N] = 0
                diag[i][k * N - 1] = 0
    D1 = spar.dia_matrix((diag, offset), shape=(N * M, N * M))
    return D1

def diagonalenExtrahieren(N,M):
    EN=np.arange(1,N*M+1)
    A=spar.dia_matrix(([10*EN, 10*EN, 10*EN, 10*EN, 10*EN, 10*EN, 10*EN, 10*EN, 10*EN, 10*EN, 10*EN],[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),shape=(N*M,N*M))
    #io.imshow(A.todense())
    #io.show()
    diag=A.data
    offset=A.offsets
    print(A.todense())
    print('\n')
    #io.imshow(diag)
    #io.show()
    print(diag)
    print(offset)
    print('\n')
    for i in range(0,N):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=0
            if 0<=i-offset[m]<N*M:
                diag[m][i] = 0
    for i in range(N*(M-1),N*M):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=0
            if 0<=i-offset[m]<N*M:
                diag[m][i] = 0
    for i in range(N, N*M, N):
        for m in range(len(offset)):
            if 0<= i+offset[m]<N*M:
                diag[m][i+offset[m]]=0
            if 0<= (i-1)+offset[m]<N*M:
                diag[m][i-1+offset[m]]=0
            if 0 <= i - offset[m] <= N * M:
                diag[m][i] = 0
            if 0 <= i-1 - offset[m] <= N * M:
                diag[m][i-1] = 0
    Anew=spar.dia_matrix((diag,offset), shape=(N*M,N*M))
    print(Anew.todense())
    print('\n')
    print(diag)
    #io.imshow(Anew.todense())
    #io.show()
    return Anew
    #io.imshow(Anew.todense())
    #io.show()
diagonalenExtrahieren(3,6)
#f=np.arange(0,81)
#print(f)
#x=f.reshape((9,9),order='F')
#print(x)
"""
def diagonalenExtrahieren(A,N,M):
    io.show()
    diag=A.data
    offset=A.offsets
    io.imshow(diag)
    io.show()
    for i in range(0,len(offset)):
        for j in range(len(diag[i])):
            for k in range(0, N-1):
                diag[i][k] = 0
            for k in range(N*(M-1),M*N):
                diag[i][k] = 0
            for k in range(1, M-1):
                diag[i][k*N] = 0
                diag[i][k*N-1] = 0
    Anew=spar.dia_matrix((diag,offset), shape=(N*M,N*M))
    #io.imshow(Anew.todense())
    #io.show()
    return Anew
def minor(N,M):
    EN = np.arange(0, N * M)
    A = spar.dia_matrix(([10 * EN, 10 * EN, 20 * EN, 12 * EN], [2, -20, 0, 30]), shape=(N * M, N * M))
    io.imshow(A.todense())
    io.show()
    Anew=diagonalenExtrahieren(A,N,M)
    Anex=diagonalenExtrahieren(Anew.T,N,M)
    io.imshow(Anex.todense())
    io.show()


    #io.imshow(Anew.todense())
    #io.show()
#minor(9,9)"""
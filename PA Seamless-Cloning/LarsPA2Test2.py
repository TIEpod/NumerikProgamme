import numpy as np
import skimage
from scipy import misc
from skimage import data, io
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse import linalg
import time
def anpassungsmatrix(N,M):
    r=[]
    for k in range(N,N*(M-1)):
        if k%N!=0 and k%N!=N-1:
            r.append(k)
    row = np.array(r)
    col = np.arange(0,(N-2)*(M-2))
    data = np.ones(shape=((N-2)*(M-2)))

    I_A = sparse.coo_matrix((data,(row,col)),shape=(N*M,(N-2)*(M-2)))
    #io.imshow(I_A.todense())
    #io.show()
    return I_A
anpassungsmatrix(10,6)

def Laplace_Operator(N,M):
    start=time.time()
    # I_M ist eine sparse Marix der Größe MxM. Auf der Diagonalen stehen -2 en und auf den beiden
    # Nebendiagonalen 1 sen.
    # Mit print(I_M.todense()) kann die Matrix in vollem Format zurück gegeben werdend.
    EM = np.ones(M)
    EN = np.ones(N)
    D_M = sparse.dia_matrix(([EM, (-2) * EM, EM], [-1, 0, 1]), shape=(M, M))
    D_N = sparse.dia_matrix(([EN, (-2) * EN, EN], [-1, 0, 1]), shape=(N, N))

    #spra.eye ist Einheitsmatrix in (sparse.) dia_matrix Format
    #spar.kron ist das Kronecher-Produkt für Sparse Matrizen
    Delta = (sparse.kron(sparse.eye(M), D_N) + sparse.kron(D_M, sparse.eye(N)))
    ende=time.time()
    plt.figure('Laplace Operator')
    io.imshow(Delta.todense())
    plt.show()
    print('Laufzeit von Laplace_Operator '+'{:5.3f}s'.format(ende-start))
    return Delta
def innen(N,M):
    D=Laplace_Operator(N,M)
    Inn=D._mul_sparse_matrix(anpassungsmatrix(N,M))
    io.imshow(Inn.todense())
    io.show()
    return None
innen(6,5)
#print(Laplace_Operator(5,7).todense())
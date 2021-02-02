import numpy as np
import random
import math
from skimage import io
def norm(W):
    '''Gibt die Summe der Einträge von W zurück.'''
    total=0
    for i in range(len(W)):
        for j in range(len(W[0])):
            total=total+W[i][j]
    return total
def weighted_mean(B, W):
    '''Berechnet den Mittelwert von A mit dem Filter W'''
    mean = 0
    for i in range(len(B)):
        for j in range(len(B[0])):
            mean += B[i][j]*W[i][j]
    return mean
def weighted_median(B, W):
    #matrix to list
    b = B.flatten()
    w = W.flatten()
    #sort
    b = [x for _,x in sorted(zip(w, b))]
    w.sort()
    weight = w[0]
    index = 0
    while weight < 1/2:
        index += 1
        weight += w[index]
    if weight == 1/2:
        return (b[index]+b[index+1])/2
    return b[index]

def test():
    maxMean=0
    maxMedi=0
    for i in range(100):
        dim=random.randrange(1,100)
        A=np.zeros(shape=(dim,dim))
        for i in range(dim):
            for j in range(dim):
                A[i][j]=random.randrange(0,255)
        W=np.full((dim,dim),1/dim**2)
        Errmean=abs(weighted_mean(A,W)-np.mean(A))
        Errmedian=abs(weighted_median(A,W)-np.median(A))
        maxMean=max(maxMean, Errmean)
        maxMedi=max(maxMedi, Errmedian)

    print('Fehler Mittelwert:'+str(maxMean))
    print('Fehler Median1:'+str(maxMedi))

def gaussgewichte(omega):
    M=3*math.ceil(omega)
    Wtilde = np.zeros(shape=(2 * M + 1, 2 * M + 1))
    for i in range(M + 1):
        for j in range(M + 1):
            Wtilde[i + M][j + M] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
            Wtilde[M - i][M - j] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
            Wtilde[M - i][M + j] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
            Wtilde[M + i][M - j] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
    W = Wtilde / norm(Wtilde)
    io.imshow(W)
    io.show()

test()
gaussgewichte(3)

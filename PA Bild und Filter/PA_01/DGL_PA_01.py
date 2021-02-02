import numpy as np
import math
import skimage
from scipy import misc
from skimage import data, io
from skimage.io import imread, imshow
from termcolor import colored

#print(colored('hello', 'red'), colored('world', 'green'))


def weighted_median(A, W):
    #matrix to list
    a = A.flatten()
    w = W.flatten()

    #sort
    a = [x for _,x in sorted(zip(w,a))]
    w.sort()

    weight = w[0]
    index = 0
    while weight < 1/2:
        index += 1
        weight += w[index]

    if weight == 1/2:
        return (a[index]+a[index+1])/2
    return a[index]

def zerlegungDerEins(W):
    total=0
    for i in range(len(W)):
        for j in range(len(W[0])):
            total=total+W[i][j]
    return total

def gewichteterMittelwert(A,W):
    mittelW = 0
    #gewicheter Mittelwert
    for i in range(len(A)):
        for j in range(len(A[0])):
            mittelW=mittelW+(W[i][j]*A[i][j])
    return mittelW

def bilateralerFilter(A,somega,romega):
    ws = lambda x,y: math.exp(-(x**2+y**2)/(2*somega**2))
    wr = lambda z : math.exp((-z**2)/(2*romega**2))
    zaehler=0
    nenner=0
    k=math.ceil(len(A)/2)
    l=math.ceil(len(A[0])/2)
    for i in range(len(A)):
        for j in range(len(A[0])):
            val=ws(k-l,l-j)*wr(A[k][l]-A[i][j])
            zaehler = zaehler+val*A[i][j]
            nenner = nenner+val

    return zaehler/nenner






def gewichtererMedian(A, W):
    gewicht = 0
    fA = A.flatten()
    fW = W.flatten()
    #io.imshow(W)
    #io.show()
    #pass
    fI = np.argsort(fA)
    k=0
    for i in range(len(fW)):
        gewicht = gewicht+fW[fI[i]]
        if gewicht == 0.5:
            k = i
            break
        elif gewicht > 0.5:
            k = -i
            break
    if k == abs(k):
        medianA = 0.5*(fA[fI[k]]+fA[fI[k+1]])
    elif k < 0:
        medianA=(fA[fI[k]])
    return medianA

def mittelwertfilter(A,s):

    n , m = A.shape
    #Wir erzeugen den Filter mit groesse passend zu s
    W = np.ones((2*s+1, 2*s+1))
    W = W / ((2*s+1)**2)

    #Die Matrix J enthält in der mitte A und einen Rahmen der
    #dicke s.
    J = np.empty(shape=(n+2*s, m+2*s))

    for i in range(0, n):       # in der Mitte von J ist A
        for j in range(0, m):
            J[s+i][s+j] = A[i][j]

    for j in range(0,s):            #ersten s Spalten
        for i in range(0,s):            #erseten Zeilen
            J[i][j]=J[s][s]
        for i in range(s,n+s):          #mittleren Zeilen
            J[i][j]=J[i][s]
        for i in range(n+s-1,n+2*s):    #letzten Zeilen
            J[i][j]=J[n+s-1][j]

    for j in range(s,m+s):          #mittleren Spalten
        for i in range(0,s):            #oberen Zeilen
            J[i][j]=J[s][j]
        for i in range(n+s-1, n+(2*s)):   #unteren Zeilen
            J[i][j]=J[n+s-1][j]

    for j in range(m+s,m+2*s):     #rechen s Spalten
        for i in range(0,s):            # oberen s Zeilen
            J[i][j]=J[s][m+s-1]
        for i in range(s,n+s):          # die mittleren Zeilen
            J[i][j]=J[i][m+s-1]
        for i in range(n+s,n+2*s):
            J[i][j]=J[n+s-1][m+s-1]     # die unteren Zeilen
    B=np.empty_like(A)

    # B soll das fertige Bild werden
    for i in range(s, B.shape[0]+s):
        for j in range(s, B.shape[1]+s):
            B[i-s][j-s] = bilateralerFilter(J[i-s:i+s+1, j-s:j+s+1],3,75)
    return B


def gaussfilter(A,omega,s):

    n , m = A.shape
    #Wir erzeugen den Filter mit groesse passend zu s
    M = 3*math.ceil(omega)
    Wtilde = np.zeros(shape=(2*M+1,2*M+1))
    for i in range(M+1):
        for j in range(M+1):
            Wtilde[i+M][j+M] = math.exp(-(i**2+j**2)/(2*omega**2))
            Wtilde[M-i][M-j] = math.exp(-(i**2+j**2)/(2*omega**2))
            Wtilde[M-i][M+j] = math.exp(-(i**2+j**2)/(2*omega**2))
            Wtilde[M+i][M-j] = math.exp(-(i**2+j**2)/(2*omega**2))
    normierung=zerlegungDerEins(Wtilde)
    W=Wtilde/normierung
    x=zerlegungDerEins(W)

    V=W[M-s:M+s+1,M-s:M+s+1]
    V=V/zerlegungDerEins(V)


    #io.imshow(W)
    #io.show

    #Die Matrix J enthält in der mitte A und einen Rahmen der
    #dicke s.
    J = np.empty(shape=(n+2*s, m+2*s))

    for i in range(0, n):       # in der Mitte von J ist A
        for j in range(0, m):
            J[s+i][s+j] = A[i][j]

    for j in range(0,s):            #ersten s Spalten
        for i in range(0,s):            #erseten Zeilen
            J[i][j]=J[s][s]
        for i in range(s,n+s):          #mittleren Zeilen
            J[i][j]=J[i][s]
        for i in range(n+s-1,n+2*s):    #letzten Zeilen
            J[i][j]=J[n+s-1][j]

    for j in range(s,m+s):          #mittleren Spalten
        for i in range(0,s):            #oberen Zeilen
            J[i][j]=J[s][j]
        for i in range(n+s-1, n+(2*s)):   #unteren Zeilen
            J[i][j]=J[n+s-1][j]

    for j in range(m+s,m+2*s):     #rechen s Spalten
        for i in range(0,s):            # oberen s Zeilen
            J[i][j]=J[s][m+s-1]
        for i in range(s,n+s):          # die mittleren Zeilen
            J[i][j]=J[i][n+s-1]
        for i in range(n+s,n+2*s):
            J[i][j]=J[n+s-1][m+s-1]     # die unteren Zeilen
    B=np.empty_like(A)

    # B soll das fertige Bild werden
    for i in range(s, B.shape[0]+s):
        for j in range(s, B.shape[1]+s):
            B[i-s][j-s] = gewichtererMedian(J[i-s:i+s+1, j-s:j+s+1],V)
    return B


imageB1 = imread('B1.png')
imageB2 = imread('B2.png')
imageC =imread('C.png')
#io.imshow(imageB2)
#io.show()

C = mittelwertfilter(imageC,3)
io.imshow(C)
io.show()
io.imshow(imageC)
io.show()

#D = gaussfilter(imageB2,2,2)
#io.imshow(D)
#io.show()
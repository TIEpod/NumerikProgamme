import numpy as np
import math
import skimage
from scipy import misc
from skimage import data, io
from skimage.io import imread, imshow

#Alles zu Numpy
n=3
m=3
A = np.arange(6).reshape(2, 3)
B=misc.ascent()
#G=sk.img_as_ubyte(B)


def zerlegungDerEins(W):
    total=0
    for i in range(len(W)-1):
        for j in range(len(W[0])-1):
            total=total+W[i][j]
    return total

#E = np.ones(shape=(n,m)) #Array mit einsen als Einträgen
#W = E*(1/(n*m)) #Normiertes Array mit konstanten Einträgen
omega=5
M = math.ceil(omega)
s = M
Wtilde = np.zeros(shape=(2*M+1,2*M+1))
for i in range(M):
    for j in range(M):
        Wtilde[i+M][j+M] = math.exp(-(i**2+j**2)/(2*omega**2))
        Wtilde[M-i][M-j] = math.exp(-(i**2+j**2)/(2*omega**2))
        Wtilde[M-i][M+j] = math.exp(-(i**2+j**2)/(2*omega**2))
        Wtilde[M+i][M-j] = math.exp(-(i**2+j**2)/(2*omega**2))

normierung=zerlegungDerEins(Wtilde)
W=Wtilde/normierung
io.imshow(W)

imageB1 = imread('B1.png')
io.imshow(imageB1)

io.show()





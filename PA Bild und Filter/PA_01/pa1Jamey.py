import numpy as np
from skimage import data, io
from skimage.io import imread, imshow

##  version1.2 from 11/21/2020

def weighted_mean(A, W):
    mean = 0

    for i in range(len(A)):
        for j in range(len(A[0])):
            mean += A[i][j]*W[i][j]
    return mean

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


#s <= min(len(B), len(B[0])
def mean_filter(B, s=1):
    U = np.zeros(B.shape)
    W = np.full((2*s+1, 2*s+1), (2*s+1)**(-2))

    B_ = np.block([[np.flip(B)  [-s:,-s:], np.flipud(B)[-s:], np.flip(B)[-s:,:s] ],
                   [np.fliplr(B)[:,  -s:], B           , np.fliplr(B)[:,:s]],
                   [np.flip(B)  [:s, -s:],  np.flipud(B)[:s], np.flip(B)[:s,:s]  ]])

    for i in range(len(B)):
        for j in range(len(B[0])):       
            T = B_[(i+s)-s:(i+s)+1+s, (j+s)-s:(j+s)+1+s]
            U[i][j] = weighted_mean(W, T)

    return U


def main():
    imageB1 = imread('B1.png')  
    imageB2 = mean_filter(imageB1, 1)

    #print(imageB1)
    #print(imageB2)
    
    io.imshow(imageB1)
    #io.imshow(imageB2)
    
    io.show()
    
main()



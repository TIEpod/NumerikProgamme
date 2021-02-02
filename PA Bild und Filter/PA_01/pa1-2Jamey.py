import numpy as np
from skimage import data, io
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import math

#
##  @Gruppe96, Nov 25th, 2020


## Hilfsfunktionen ##

def norm(W):
    '''Gibt die Summe der Einträge von W zurück.'''
    total = 0
    for i in range(len(W)):
        for j in range(len(W[0])):
            total += W[i][j]
    return total


def frameIt(B,s):
    '''Bildet einen Rahmen der Breite s um das Bild B, dabei ist der Ramen eine
    Spieglung des Bildes. Input: B,s (Bild und Rahmen dicke)
    Output: J ndarray mit B als Zentrum und Rahmen der größe es'''
    B_ = np.block(
        [[np.flip(B)[-s:, -s:], np.flipud(B)[-s:], np.flip(B)[-s:, :s]],
        [np.fliplr(B)[:, -s:], B, np.fliplr(B)[:, :s]],
        [np.flip(B)[:s, -s:], np.flipud(B)[:s], np.flip(B)[:s, :s]]]
    )

    return B_


## AUFGABE 1: Gewichteter Mittelwert und gewichteter Median ##

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
    return #TODO


## AUFGABE 2: Gewichteter Mittelwertfilter ##

def mittelwertfilter(B, s , type = 'gleichverteilt', omega = None):
    #Wir erzeugen den gleichverteilten Filter mit groesse passend zu s

    if type == 'gleichverteilt':
        W = np.ones((2*s+1, 2*s+1))
        W = W / ((2*s+1)**2)            # W wird normiert, so dass die Summe=1

    elif type == 'gauss' and omega:
        M = 3 * math.ceil(omega)
        Wtilde = np.zeros(shape=(2 * M + 1, 2 * M + 1))
        for i in range(M + 1):
            for j in range(M + 1):
                Wtilde[i + M][j + M] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
                Wtilde[M - i][M - j] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
                Wtilde[M - i][M + j] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
                Wtilde[M + i][M - j] = math.exp(-(i ** 2 + j ** 2) / (2 * omega ** 2))
        W = Wtilde / norm(Wtilde)
        if s!=None:
            V = W[M - s:M + s + 1, M - s:M + s + 1]
            W = V * (1 / norm(V))
        else:
            s=M

    B_ = frameIt(B,s) #rahmen B ein
    A = np.empty_like(B) #A soll fertiges Bild werden
    for i in range(s, A.shape[0]+s):
        for j in range(s, A.shape[1]+s):
            A[i-s][j-s] = weighted_mean(B_[i-s:i+s+1, j-s:j+s+1],W)
    return A


## AUFGABE 3: Gewichteter Medianfilter ##

def medianfilter(A, s=1, W=None, type='gleichverteilt',omega=None, saltpepper='out'):

    if s==None and type=='gauss' and omega and omega!=0:
        s=3*math.ceil(omega)

    #Wir erzeugen den Filter mit groesse passend zu s
    if W==None and type=='gauss':
        if omega == None:
            raise TypeError('Es muss bei gauss eine Varianz omega angegeben werde')
        M = 3*math.ceil(omega)
        Wtilde = np.zeros(shape=(2*M+1,2*M+1))
        for i in range(M+1):
            for j in range(M+1):
                Wtilde[i+M][j+M] = math.exp(-(i**2+j**2)/(2*omega**2))
                Wtilde[M-i][M-j] = math.exp(-(i**2+j**2)/(2*omega**2))
                Wtilde[M-i][M+j] = math.exp(-(i**2+j**2)/(2*omega**2))
                Wtilde[M+i][M-j] = math.exp(-(i**2+j**2)/(2*omega**2))
        W=Wtilde/norm(Wtilde)

        V = W[M - s:M + s + 1, M - s:M + s + 1]
        W = V*(1 / norm(V))
    elif W==None and type=='gleichverteilt':
        W = np.full((2*s+1, 2*s+1), 1/(2*s+1)**2)
    elif W and norm(W)!=1:
        print('Die summe der Einträge von W ist nicht Eins sondern: ',x)


    J = frameIt(A,s)
    B = np.empty_like(A)

    # B soll das fertige Bild werden
    if saltpepper=='out':                       #if abfrage hat nichts mit der Aufgabe zu tun
        for i in range(s, B.shape[0]+s):
            for j in range(s, B.shape[1]+s):
                B[i-s][j-s] = weighted_median(J[i-s:i+s+1, j-s:j+s+1],W)

    elif saltpepper=='activ':                   #hat nichts mit aufgabe zu tun
        W[s][s]=0                               # ist saltpepper aktiv, werden nur die Picksel
        W=W*(1/norm(W))                         #beareibeitet die schwarz(255) oder weiss(0) sind.
        for i in range(s, B.shape[0]+s):
            for j in range(s, B.shape[1]+s):
                if J[i][j]==255 or J[i][j]==0:
                    B[i-s][j-s] = weighted_median(J[i-s:i+s+1, j-s:j+s+1],W)
                else:
                    B[i-s][j-s]=J[i][j]

    return B



## AUFGABE 4: Bilateraler Filter ##

def bilateralesGewicht(A,somega,romega):
    '''Rechnet die bilateralen Gewichte aus.'''
    ws = lambda x,y: math.exp(-(x**2+y**2)/(2*somega**2))
    wr = lambda z : math.exp((-z**2)/(2*romega**2))
    zaehler=0
    nenner=0
    k = math.ceil(len(A)/2)
    l = math.ceil(len(A[0])/2)
    for i in range(len(A)):
        for j in range(len(A[0])):
            val=ws(k-l,l-j)*wr(A[k][l]-A[i][j])
            zaehler = zaehler+(val*A[i][j])
            nenner = nenner+val
    return zaehler/nenner

def bilateralerFilter(A, s , somega,romega):
    '''Wendet den bitaleralen Filter auf A an. Die groesse der Maske ist durch
    M=2*s+1 gegeben.'''
    J = frameIt(A, s)
    # B soll das fertige Bild werden
    B = np.empty_like(A)
    for i in range(s, B.shape[0] + s):
        for j in range(s, B.shape[1] + s):
            B[i-s][j-s] = bilateralesGewicht(J[i-s:i+s+1,j-s:j+s+1],somega, romega)
    return B


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def ausgabe(Bild='B2.png'):
    if type(Bild)==str:
        try:
            imageB1= imread(Bild)
        except:
            raise ImportError('Bild kann nicht importiert werden')
    image=imread('B2.png')
    plt.gray()
    fig, ((ax11, ax12, ax13),(ax21, ax22, ax23)) = plt.subplots(nrows=2, ncols=3)

    s12 = 2                 #für Median mit gleichverteilten Gewichten
    s13 = 1                 #für Mittelwert mit gleichverteilten Gewichten
    s21,somega,romega=1,3,75    #für Bilateralenfilter (mit gauss Gewichten)
    s22,omega22=1,2               #für Median mit gauss Gewichten
    s23,omega23 = None,6
    ax11.imshow(image)
    ax12.imshow(medianfilter(image, s=s12))
    ax13.imshow(mittelwertfilter(image, s13))

    ax21.imshow(bilateralerFilter(image,s=s21,somega=somega,romega=romega))
    ax22.imshow(medianfilter(image, s=s22, type='gauss', omega=omega22))
    ax23.imshow(mittelwertfilter(image, s=1, type='gauss', omega=omega23))



    ax11.set_title('Original')
    ax12.set_title('Median s='+str(s12))
    ax13.set_title('Mittel s='+str(s13))
    ax21.set_title('Bilateral o_s='+str(somega)+' o_r='+str(romega))
    ax22.set_title('Median Gauß o='+str(omega22))
    ax23.set_title('Mittel Gauss s='+str(s23)+' o='+str(omega23))

    ax11.axis('off')
    ax12.axis('off')
    ax13.axis('off')
    ax21.axis('off')
    ax22.axis('off')
    ax23.axis('off')

    plt.show()

    
ausgabe()



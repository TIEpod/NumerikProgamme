##alle neuen und veränderten funktionen von mir

def exercise3():
    g  = lambda x: np.array([x[0]**2+x[1]**2-6*x[0], (3/4)*np.exp(-x[0])-x[1]])
    dg = lambda x: np.array([[2*x[0]-6, 2*x[1]], [(-3/4)*np.exp(-x[0]), -1]])

    root = newton(g, dg, np.array([0.08, 0.7]))
    print("Die Nullstellen nach dem Newton-Verfahren sind %f und %f" % (g(root)[0], g(root)[1]))
    return

#nach folie6
def opt(dF, hF, x0, delta = PREC, epsilon = PREC, maxIter = 100):
    x = x0
    
    for i in range(maxIter):
        y = np.copy(x) #vorheriges folgenglied

        x = x - np.linalg.inv(hF(x))@dF(x)
        
        
        if norm(x-y) < delta or norm(dF(x)) < epsilon:
            return x
    return x

def exercise6():
    f  = lambda x: (x[0]+1)**4 + (x[1]-1)**4
    df = lambda x: 4*np.array([(x[0]+1)**3, (x[1]+1)**3])
    hf = lambda x: 12* np.array([[(x[0]+1)**2, 0], [0, (x[1]-1)**2]]) #pos def
    
    x0 = [-1.1, 1.1]
    mini = opt(df, hf, x0)
    
    print("Ein Minima von f liegt bei", list(np.around(mini, decimals=2)))
    print("Da die Hessematrix in allen Stellen streng positiv definit ist, ist dies das einzige Extrema. ")


def exercise7():
    seq = lambda z, c:  z**2 + c  
    
    xH = np.linspace(start=-1.5, stop=1.5, num=1024, endpoint=True)
    yH = np.linspace(start=-1, stop=1, num=1024, endpoint=True)

    M = np.array([[complex(re, im) for re in xH] for im in yH], dtype=complex)

    C = np.zeros(M.shape) #anz wo ||<=2
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            count = 0
            c = M[i][j]
            for k in range(256):
                if abs(c) <= 2:
                    count += 1
                
            
            C[i][j] = count

    plt.imshow(C)
    plt.show()






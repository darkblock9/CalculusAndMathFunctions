from math import pi,sqrt


### Calculadora de Constantes Matematicas ###

def decimal(number):
    # da a ordem de grandeza em numero de
    # casas decimais em um dado numero
    # na forma 1*10^(-n)
    # n = 0,1,2,...
    # ex: decimal(0.001) retorna 3
    l = 0
    m = 1
    while m*number < 1:
        l = l + 1
        m = 10**l
    return l

def calcpi(precision):
    # CALCULA PI A UMA DADA PRECISAO
    # PRECISAO NA FORMA 1E(-X)
    # X = 0,1,2,...
    dec = decimal(precision)
    epsilon = precision/10
    n = 0
    quarterpi = 0
    converged = False
    while not(converged):
        u = ((-1)**n)/(2*n+1)
        if abs(u) < epsilon:
            converged = True
        else:
            quarterpi = quarterpi + u
            n = n + 1
    return round(4*quarterpi,dec)

def ratio(n):
    if n == 0:
        return 1
    else:
        return 1 + 1/ratio(n-1)

def goldenratio(precision):
    # CALCULA RAZAO AUREA A UMA DADA PRECISAO EX: 0.001
    dec = decimal(precision)
    epsilon = precision/10
    n = 0
    r = ratio(n)
    converged = False
    while not(converged):
        n = n + 1
        golden = ratio(n)
        if abs(golden-r) < epsilon:
            converged = True
        else:
            r = golden
    return round(golden,dec)   



### Vetores e Matrizes ###

def ProdEsc(x,y):
    L = len(x)
    S = 0
    for i in range(L):
        S = S + x[i]*y[i]
    return S

def ProdNum(vetor,numero):
    L = len(vetor)
    produto = []
    for i in range(L):
        produto.append(vetor[i]*numero)
    return produto

def ProdVet(x,y):
    MATRIZ = [[0,0,0],x,y]
    ProdV = []
    for i in range(3):
        COORD = ((-1)**i)*det_2x2(submatriz(MATRIZ,1,i+1))
        ProdV.append(COORD)
    return ProdV

def Norma(x):
    Norma = sqrt(ProdEsc(x,x))
    return Norma

def vet_Soma(x,y):
    L = len(x)
    z = []
    for i in range(L):
        S = x[i] + y[i]
        z.append(S)
    return z

def vet_Subtracao(x,y):
    z = vet_Soma(x,ProdNum(y,-1))
    return z

def ehMatriz(A):
    # Verifica se uma lista de listas dada é uma matriz ou 
    # não.
    ehM = True
    # Guarda o número m de linhas e n de colunas
    m = len(A)
    n = len(A[0])
    # Se a matriz for uma matriz-linha, len(A) == 1 e o 
    # programa acaba por aqui.
    if m == 1:
        return ehM
    # Passeia por cada lista (exceto a primeira), e, a cada
    # lista por que passa, verifica se o número de elementos
    # é igual ao da primeira lista (A[0]) e, se algum for
    # diferente, muda ehM para False.
    else:
        for i in range(1,m):     
            if len(A[i]) != n:
                ehM = False
    return ehM

def ehQuadrada(A):
    ehQ = False
    m = len(A)
    n = len(A[0])
    if ehMatriz(A) and m==n:
        ehQ = True
    return ehQ

def ehTriangular(A):
    if ehQuadrada(A):
        n = len(A)
        boole1 = True
        boole2 = True
        for i in range(1,n):
            for j in range(0,i):
                if A[i][j] != 0:
                    boole1 = False
        for i in range(0,n-1):
            for j in range(i+1,n):
                if A[i][j] != 0:
                    boole2 = False
        return boole1 or boole2

def DP(A):
    # Recebe dois tipos de matrizes: 
    # 1) quadradas (n x n)
    # 2) ampliadas por Sarrus (n x (2n-1))
    # devolve, no caso 1, uma lista em que
    # cada elemento é da diagonal principal
    # da matriz. No caso 2, devolve uma
    # lista de n listas correspondentes
    # às n diagonais da matriz ampliada.
    B = []
    if ehQuadrada(A):
        for i in range(len(A)):
            B.append(A[i][i])
        return B
    elif len(A[0]) == 2*len(A) - 1:
        for N in range(len(A)):
            B.append([])   
        for num in range(len(A)):
            k = 0
            while k < len(A):
                for l in range(num,num+len(A)):                 
                    B[num].append(A[k][l])
                    k = k + 1
        return B

def DS(A):
    # Recebe dois tipos de matrizes: 
    # 1) quadradas (n x n)
    # 2) ampliadas por Sarrus (n x (2n-1))
    # devolve, no caso 1, uma lista em que
    # cada elemento é da diagonal secundária
    # da matriz. No caso 2, devolve uma
    # lista de n listas correspondentes
    # às n diagonais secundárias da matriz
    # ampliada.
    B = []
    if ehQuadrada(A):
        j = 0
        for i in range(len(A)-1,-1,-1):
            B.append(A[i][j])
            j = j + 1
        return B
    elif len(A[0]) == 2*len(A) - 1:        
        for N in range(len(A)):
            B.append([])
        for num in range(len(A)):
            j = num
            for i in range(len(A)-1,-1,-1):
                B[num].append(A[i][j])
                j = j + 1
        return B  

def ampliada(A,B):
    m1 = len(A)
    n1 = len(A[0])
    m2 = len(B)
    n2 = len(B[0]) 
    C = []
    if m1 == m2 and ehMatriz(A) and ehMatriz(B):
        for i in range(m1):
            C.append([])
            for j in range(n1):
                C[i].append(A[i][j])
            for k in range(n2):
                C[i].append(B[i][k])
    return C

def submatriz(A,i,j):
    l = len(A)
    S = []
    p = 0
    for k in range(l):
        if k != i-1:
            S.append([])
            for m in range(l):
                if m != j-1:
                    S[p].append(A[k][m])
            p = p + 1
    return S

def esq_Sarrus(A):
    B = []
    for n in range(len(A)):
        B.append([])
    for i in range(len(A)):
        for j in range(len(A)-1):
            B[i].append(A[i][j])    
    B = ampliada(A,B)
    return B

def det_triang(A):
    W = DP(A)
    n = len(W)
    det = 1
    for num in range(n):
        det = det*W[num]
    return det

def det_laplacelinha(A):
    Y = len(A)
    lista = []
    lista.append(A)
    B = lista[0:1]
    L = len(B)
    for N in range(Y,3,-1):
        lista = []
        for k in range(L):
            for j in range(N):
                C = submatriz(B[k],1,j+1)
                lista.append(C)
        L = len(lista)
        B = lista[0:L]
    lista = []
    for num in range(L):
        lista.append(det_3x3(B[num]))
    l = 4
    C = L
    for l in range(4,Y+1):
        listatmp = []
        for u in range(int(C/l)):
            lista2 = []
            for t in range(l):
                lista2.append([])
            for n in range(u*l,(u+1)*l):
                ntmp = n
                for p in range(1,Y-l+1):
                    F = ntmp//(C/(fat(Y)/fat(Y-p)))
                    F = int(F)
                    lista2[n%l].append(F)
                    ntmp = ntmp%(C/(fat(Y)/fat(Y-p)))
                F = int(ntmp) 
                lista2[n%l].append(F)
            determinantes = lista[u*l:(u+1)*l]
            linha = A[Y-l][0:Y]
            lista4 = lista2[0][0:Y-l]                  
            for m in range(Y-l):
                U = len(linha)
                tmp = linha[0:U]
                linha = tmp[0:lista4[m]]
                tmp = tmp[lista4[m]+1:U]
                x = len(tmp)
                for o in range(x):
                    linha.append(tmp[o])
            D = 0
            for I in range(l):
                D = D + ((-1)**I)*determinantes[I]*linha[I]   
            listatmp.append(D)       
        C = len(listatmp)
        lista = listatmp[0:C]   
    return lista[0]

def det_1x1(A):
    return A[0][0]

def det_2x2(A):
    V = (DP(A),DS(A))      
    lista = []
    for num1 in range(2):
        PROD = 1
        for num2 in range(2):
            PROD = V[num1][num2]*PROD
        lista.append(PROD)
    return lista[0]-lista[1]

def det_3x3(A):
    B = esq_Sarrus(A)
    K = (DP(B),DS(B))
    b1 = 0
    for i1 in range(len(K[0])):
        a1 = 1    
        for j1 in range(len(K[0])):
            a1 = a1*(K[0][i1][j1])
        b1 = b1 + a1
    b2 = 0
    for i2 in range(len(K[1])):
        a2 = 1    
        for j2 in range(len(K[1][0])):
            a2 = a2*(K[1][i2][j2])
        b2 = b2 + a2           
    determinante = b1-b2
    return determinante

def det(A):
    if ehQuadrada(A):
        l = len(A)
        if ehTriangular(A):
            determinante = det_triang(A)
        if l == 1:
            determinante = det_1x1(A)
            return determinante
        elif l == 2:
            determinante = det_2x2(A)           
            return determinante
        elif l == 3:
            determinante = det_3x3(A)
            return determinante
        elif l >= 4:
            determinante = det_laplacelinha(A)
            return determinante

def multv(matriz,vetor):
    l = len(matriz) 
    v = []
    for i in range(l):
        x = ProdEsc(matriz[i],vetor)
        v.append(x)
    return v



### Funções ###

def gaussiana1(x):
    return exp(-x**2/2)/sqrt(2*pi)

def gaussiana(X):
### X = ['x','mu','sigma']
    x,mu,sigma=X[0],X[1],X[2]
    A = 1/(sqrt(2*pi)*sigma)
    y = A*exp(-.5*((x-mu)/sigma)**2)
    return y

def gaussiana2D(X):
### X = ['x','y','mux','muy','sigma','cov']
    x,y,mux,muy,sigma,cov=X[0],X[1],X[2],X[3],X[4],X[5]
    Dx=x-mux
    Dy=y-muy
    r = cov/sigma**2
    a = Dx**2-2*r*Dx*Dy+Dy**2
    N = 1/(2*pi*sigma**2*sqrt(1-r**2))
    z = N*exp(-a/(2*sigma**2*(1-r**2)))
    return z


### Série de Fourier ###

def fourier(f,L,x,N):
    U = [-L/2,L/2]
    def A(n):
        def F(t):
            y = f(t)*cos(2*pi*n*t/L)
            return y
        I = 2*integ1(F,U)/L
        return I
    def B(n):
        def F(t):
            y = f(t)*sin(2*pi*n*t/L)
            return y
        I = 2*integ1(F,U)/L
        return I
    S = A(0)/2
    for num in range(1,N+1):
        c = cos(2*pi*num*x/L)
        s = sin(2*pi*num*x/L)
        An = A(num)
        Bn = B(num)
        a = An*c + Bn*s
        S = S + a
    return S
    


### Trigonometria ###

def sin(x):
    N = 10
    theta = abs(x)
    if x >= 0:
        a = 1
    else:
        a = -1
    if theta >= 2*pi:
        theta = theta%(2*pi)
    if 0.5*pi < theta <= pi:
        theta = pi - theta
    elif pi < theta <= 1.5*pi:
        theta = theta - pi
        a = (-1)*a
    elif 1.5*pi < theta < 2*pi:
        theta = 2*pi - theta
        a = (-1)*a
    soma = 0
    for i in range(N):
        D = (-1)**i
        y = D*(theta**(2*i+1))/fat(2*i+1)
        soma = soma + y
    seno = a*soma
    return seno

def cos(x):
    N = 10
    theta = abs(x)
    a = 1
    if theta >= 2*pi:
        theta = theta%(2*pi)
    if 0.5*pi < theta <= pi:
        theta = pi - theta
        a = (-1)*a
    elif pi < theta <= 1.5*pi:
        theta = theta - pi
        a = (-1)*a
    elif 1.5*pi < theta < 2*pi:
        theta = 2*pi - theta
    soma = 0
    for i in range(N):
        D = (-1)**i
        y = D*(theta**(2*i))/fat(2*i)
        soma = soma + y
    cosseno = a*soma
    return cosseno
  
def tan(x):
    return sin(x)/cos(x)

def rad_grau(x):
    return round((180/pi)*x,1)

def arcsin(x):
    def derivada(t):
        y = 1/sqrt(1-t**2)
        return y
    arco = integ1(derivada,[0,x])
    return arco

def arccos(x):
    def derivada(t):
        y = -1/sqrt(1-t**2)
        return y
    arco = integ1(derivada,[0,x]) + pi/2
    return arco

def arctan(x):
    def derivada(t):
        y = 1/(1+t**2)
        return y
    arco = integ1(derivada,[0,x])
    return arco

def arctantaylor(x):
    N = 10
    soma = 0
    for n in range(N):
        a = ((-1)**n)*(x**(2*n+1))/(2*n+1)
        soma = soma + a
    return soma


### Trigonometria hiperbolica ###

def cosh(x):
    return 0.5*(exp(x)+exp(-x))

def sinh(x):
    return 0.5*(exp(x)-exp(-x))

def tanh(x):
    return sinh(x)/cosh(x)

def arcsinh(x):
    return lognatural(x+sqrt(1+x**2))

def arccosh(x):
    return lognatural(x+sqrt(x**2-1))

def arctanh(x):
    return lognatural((1+x)/sqrt(1-x**2))


### Calculadora de logaritmos ###

def lognatural(x):
    def hiperbole(p):
        return 1/p
    ln = integ1(hiperbole,[1,x])
    return ln


def lntaylor(x):
    ### FUNCIONA APENAS PARA 0 < x <= 2
    epsilon = 0.001
    ln = 0
    naoconvergiu = True
    n = 1
    while naoconvergiu:
        a = ((-1)**(n-1))*((x-1)**n)/n
        if abs(a) < epsilon:
            naoconvergiu = False
        else:
            ln = ln + a
            n = n + 1
    return ln



def lntaylor_old(x):
    epsilon = 0.001
    if x > 1:
        g = round(x)
    else:
        g = 1
    boole = True
    if abs(x-g) < epsilon:
        boole = False
    soma = lognatural(g)
    N = 1
    while boole:
        a = ((-1)**(N-1))*(((x/g)-1)**N)/N
        soma = soma + a
        if a < epsilon:
            boole = False
        else:
            N = N + 1
    return soma


def log(logaritmando,base):
    logaritmo = lognatural(logaritmando)/lognatural(base)
    return logaritmo


### -- ###


def squareroot(x):
    epsilon = 0.000001
    boole = True
    if (x in range(2)):
        return x
    elif 0<x<1:
        limSup = 1
        limInf = x
    elif x>1:
        limSup = x
        limInf = 1
    else:
        boole = False
        print('A raiz quadrada não está definida para valores negativos.')
    while boole:
        raiz = (limSup+limInf)/2
        if abs(raiz**2-x) < epsilon:         
            boole = False
        elif raiz**2 > x:
            limSup = raiz
        else:
            limInf = raiz
    if abs(round(raiz)-raiz) < epsilon:
        return round(raiz)
    else:
        return round(raiz,6)

        


def combinacao(x,y):
    prod = 1
    for num in range(y):
        prod = prod*(x-num)
    return prod/fat(y)



def fat(n):
    if n in range(2):
        return 1
    fatorial = n
    for i in range(n-1):
        fatorial = fatorial*(n-i-1)
    return fatorial


def fatrec(N):
    if N == 0:
        return 1
    else:
        return N*fatrec(N-1) 


def e():
    N = 10
    S = 0
    for i in range(N):
        y = 1/fat(i)
        S = S + y
    return S

euler = 2.7182818

def exp(x):
    y = euler**x 
    return y





def numeropi1():
    def funcaoauxpi(X):
        return 1
    def x0(y):
        return -sqrt(1-y**2)
    def x1(y):
        return sqrt(1-y**2)
    area = integ_duplafunc(funcaoauxpi,[[x0,x1],[-1,1]])        
    return area



def numeropi2():
    N = 200
    n = round(N/2)
    conta = 0
    passo = 2/N
    v = []
    p0 = -1+1/N
    for num in range(2):
        v.append(p0)
    for i in range(n):
        v[1] = v[1] + i*passo
        for j in range(n):
            v[0] = v[0] + j*passo
            modulo = Norma(v)
            if modulo < 1:
                conta = conta + 1
    area = 16*(conta/N**2)   
    return area   



def taylor(f,p,a):
    N = 3
    soma = f(p)
    for i in range(1,N):
        soma = soma + (derivN(f,a,i)*(p-a)**i)/fat(i)
    return soma





### LIMITES ###
# Os limites finitos são calculados
# tendo como hipótese de que a sequência converge
# e que existem.



def limitseq(a):
    n0 = 1
    epsilon = 0.0001
    boole = True
    lim = a(n0)
    while boole:
        n0 = n0 + 1
        limite = a(n0)
        if abs(limite-lim) < epsilon:
            boole = False
        else:
            lim = limite
    return limite  


def limitlatdir(f,p):        
    def seq(n):
        a = p + 1/n
        fn = f(a)
        return fn
    limite = limitseq(seq)
    return limite


def limitlatesq(f,p):        
    def seq(n):
        a = p - 1/n
        fn = f(a)
        return fn
    limite = limitseq(seq)
    return limite


def limit(f,p):
    if p == '+inf':
        lim = limitinfMais(f)
    elif p == '-inf':
        lim = limitinfMenos(f)
    else:
        a = limitlatdir(f,p)
        b = limitlatesq(f,p)
        lim = media(a,b)
    return lim


def limitinfMais(f):
    def seq(n):
        fn = f(n)
        return fn
    limite = limitseq(seq)
    return limite


def limitinfMenos(f):
    def seq(n):
        fn = f(-n)
        return fn
    limite = limitseq(seq)
    return limite





###  FUNCOES UTEIS ###  

def somatorio(inicio,fim,somando):
    soma = 0
    for i in range(inicio,fim+1):
        soma = soma + somando(i)
    return soma

def somatorioduplo(inicio1,fim1,inicio2,fim2,somando):
    soma = 0
    for j in range(inicio1,fim1+1):
        for i in range(inicio2,fim2+1):
            A = []
            A.append(i)
            A.append(j)
            soma = soma + somando(A)   
    return soma

def identidade(G):
    return G

def media(A,B):
    return (A+B)/2

def mediaarit(X):
    N = len(X)
    soma = 0
    for i in range(N):
        soma = soma + X[i]
    return soma/N

def mediageo(X):
    N = len(X)
    prod = 1
    for i in range(N):
        prod = prod*X[i]
    return prod**(1/N)

def mediaharm(X):
    N = len(X)
    soma = 0
    for i in range(N):
        soma = soma + 1/X[i]
    return N/soma

def sort(v):
    L = len(v)
    w = []
    for num in range(L):
        w.append(v[num])
    for j in range(L-1):
        for k in range(j+1,L):
            if  w[j] > w[k]:
                tmp = w[k]
                w[k] = w[j]
                w[j] = tmp
    return w     

def sortmatriz(A,j):
### Ordena coluna j da matriz A, ordem crescente com o aumento da linha
    L = len(A)
    M = len(A[0])
    w = []
    for num1 in range(L):
        w.append([])
        for num2 in range(M):
            w[num1].append(A[num1][num2])
    for l in range(L-1):
        for k in range(l+1,L):
            if  w[l][j] > w[k][j]:
                tmp = w[k]
                w[k] = w[l]
                w[l] = tmp
    return w

def funcao_vetor(v,i):
# Essa funcao isola a i-esima
# coordenada de v e escreve
# ela como uma funcao.
    def f(t):
        V = v(t)       
        y = V[i]
        return y
    return f

def funcao1var(f):
# Essa funcao converte uma funcao de um vetor de 1 dimensao
# em uma funcao de 1 coordenada.
    def g(x):
        r = []
        r.append(x)
        func = f(r)
        return func
    return g

def completavetor(P,x,i):
# Essa funcao recebe um vetor P, um escalar x e uma posicao
# i, e completa o vetor P na posicao i com o escalar x
# exemplo: entrada [1,2,3],0,2
#          saida   [1,2,0,3]
    L = len(P)
    X = []
    N = 0
    boole = True
    while N < L:
        if N == i and boole:
            X.append(x)
            boole = False
        else:
            if not(boole):
                boole = True
            X.append(P[N])
        if boole:
            N = N + 1
    if len(X) == len(P):
        X.append(x)
    return X

### DERIVADAS ###


def coef_ang(f,p,delta):
    b = f(p+delta)
    a = f(p)
    tan = (b-a)/delta
    return tan



def deriv(f,p):
    def tan(x):
        m = coef_ang(f,p,x)
        return m
    derivada = limit(tan,0)
    return derivada


def deriv2(f,p): 
    d = derivN(f,p,2)
    return d


def deriv3(f,p): 
    d = derivN(f,p,3)
    return d




def derivrec(f,n):
    if n == 0:
        return f
    else:
        def F(x):
            a = derivrec(f,n-1)
            return deriv(a,x)
        return F 





def derivN(f,p,n):
    g = derivrec(f,n)
    derivada = g(p)
    derivada = round(derivada,3)
    return derivada




def derivparc(f,p,i):
    G = len(p)
    def g(x):
        H = [] 
        for num in range(G):
            if num == i:
                H.append(x)
            else:
                H.append(p[num])
        y = f(H)
        return y
    U = p[i]
    T = deriv(g,U)
    return T




def derivparcrec(f,i,n):
    if n == 0:
        return f
    else:
        def F(X):
            C = derivparc(derivparcrec(f,i,n-1),X,i)
            return C
        return F




def derivparcN(f,p,A):
# Matriz A deve estar na forma:
# [['variavel_1','numero_1'],['variavel_2','numero_2'],...,['variavel_n','numero_n']]
# onde variavel_1,...,variavel_n deve valer 0,...,n-1 respectivamente e
# numero_1,...,numero_n são quantas vezes deseja-se derivar na variavel_i, i = 1,...,n
    L = len(A)
    g = f
    for num in range(L):
        I,N = A[num][0],A[num][1]
        F = derivparcrec(g,I,N)
        g = F      
    Y = g(p)
    return Y

def derivvet(v,x):
    L = len(v(x))
    derivada = []
    for i in range(L):
        f = funcao_vetor(v,i)
        D = deriv(f,x)
        derivada.append(D)
    return derivada

def derivNvet(v,x,n):
    L = len(v(x))
    derivada = []
    for i in range(L):
        f = funcao_vetor(v,i)
        D = derivN(f,x,n)
        derivada.append(D)
    return derivada

def derivparcvet(v,x,i):
    L = len(v(x))
    derivada = []
    for j in range(L):
        f = funcao_vetor(v,j)
        D = derivparc(f,x,i)
        derivada.append(D)
    return derivada

def derivparcNvet(v,x,A):
    L = len(v(x))
    derivada = []
    for i in range(L):
        f = funcao_vetor(v,i)
        D = derivparcN(f,x,A)
        derivada.append(D)
    return derivada

### INTEGRAIS ###

def soma_Riemann(f,a,b,N):
    dx = (b-a)/N
    def C(i):
        x = a + i*dx
        return f(x)
    return dx*somatorio(1,N-1,C)
    
    


def integ1(f,I):
    def r(n):
        return soma_Riemann(f,I[0],I[1],n)
    integral = limitseq(r)
    return integral


def integ(f,P,I,i):
# P é o ponto que informa o valor que assumem 
# as (N-1) variáveis que não são
# de integração. variável i, i=0,...,N-1
# informa a posição da variável em que se deseja 
# integrar a função f, de N variáveis.
# I é lista do intervalo dos extremos de
# integração, e pode ter função como elemento.
    def G():
        return None
    a = type(G)
    u = []
    for t in range(2):
        u.append(type(I[t])==a)
    def F(x):
        X = completavetor(P,x,i)
        y = f(X)
        return y   
    Q = []
    for k in range(2):
        if u[k] == True:
            H = I[k]
            Q.append(H(P))
        else:
            Q.append(I[k])
    Z = integ1(F,Q)
    return Z


def integ_func(f,I,i):
    def F(X):
        y = integ(f,X,I,i)
        return y
    return F


def integ_dupla(f,M):
    I1 = M[0]
    g = integ_func(f,I1,0)
    g = funcao1var(g)
    I2 = M[1]
    I = integ1(g,I2)    
    return I

def integ_duplafunc(f,M):
    A = M[1]
    def funcao(y):
        P = []
        P.append(y)
        V = []
        for n in range(2):
            q = M[0][n]
            V.append(q(y))
        F = integ(f,P,V,0)
        return F
    I = integ1(funcao,A)
    return I
            


def integ_rec(f,M,i):
# M é da forma [[a0,b0,v0],...,[a(N-1),b(N-1),v(N-1)]]
# onde ai e bi, i = 0,...,N-1 são as extremidades de
# integração inferior e superior, respectivamente, e
# vi são as variáveis sobre as quais deseja-se 
# integrar.
    if i == 0:
        return f
    else:
        v = M[i-1]
        I,x = v[0:2],v[2]
        h = integ_rec(f,M,i-1)
        F = integ_func(h,I,x)
        return F


def integ_Ninc(f,M,P):
# A função calcula a integral em N variáveis de uma função
# de m variáveis, dado que m>N, em um ponto P.
# A função usa integ_rec para iterar as integrações.
# A função precisa ordenar a matriz M segundo a 3a
# coluna porque integra no vetor ponto da funcao f
# da última até a primeira posição, mantendo intacta
# a posição da próxima variável de integração.
# Para não confundir os índices durante a integração,
# a função integra de trás pra frente, isto é,
# a ordem de integração é feita de maneira decrescente
# nos índices do ponto P (a integral mais interna é sobre
# a variável de maior índice de P).
# Para isso, ela vira a matriz
# de "ponta-cabeça"
    L = len(M)
    U = len(M[0])
    Q = sortmatriz(M,2)
    E = []
    for lin in range(L):
        E.append([])
        for col in range(U):
            E[lin].append(Q[-lin-1][col])
    F = integ_rec(f,E,L)
    Y = F(P)
    return Y


def integ_Nfull(f,M):
# A função calcula a integral em N variáveis de uma função
# de N variáveis. A matriz M é da forma [[a1,b1],...,[aN,bN]]
# onde ai,bi, i = 1,..,N são os extremos de integração da
# variável na posição (i-1), a ordem dos [ai,bi] em M 
# é da integral mais interna até a mais externa.
    L = len(M)
    E = []
    for lin in range(L):
        E.append([])
        for col in range(len(M[0])):
            E[lin].append(M[-lin-1][col])    
    K = E[1:L]
    for i in range(L-1):
        K[i].append(i+1)
    def r(x):
        X = []
        X.append(x)
        y = integ_Ninc(f,K,X)
        return y
    I = integ1(r,E[0])
    return I


def integ_func2(f,G,i):
    def funcao(X):
        V = 0
        for u in range(2):
            g = G[u]
            V.append(g(X))
        I = integ(f,X,V,i)
        return I
    return funcao


def integ_rec2(f,M,i):
# M é da forma [[a0,b0,v0],...,[a(N-1),b(N-1),v(N-1)]]
# onde ai e bi, i = 0,...,N-1 são as extremidades de
# integração inferior e superior, respectivamente, e
# vi são as variáveis sobre as quais deseja-se 
# integrar, ai e bi funções.
    if i == 0:
        return f
    else:
        v = M[i-1]
        B,x = v[0:2],v[2]
        h = integ_rec2(f,M,i-1)
        F = integ_func2(h,B,x)
        return F


def integ_Ninc2(f,M,P):
# A função calcula a integral em N variáveis de uma função
# de m variáveis, dado que m>N, em um ponto P.
# A função usa integ_rec para iterar as integrações.
# A função precisa ordenar a matriz M segundo a 3a
# coluna porque integra no vetor ponto da funcao f
# da última até a primeira posição, mantendo intacta
# a posição da próxima variável de integração.
# Para não confundir os índices durante a integração,
# a função integra de trás pra frente, isto é,
# a ordem de integração é feita de maneira decrescente
# nos índices do ponto P (a integral mais interna é sobre
# a variável de maior índice de P).
# Para isso, ela vira a matriz
# de "ponta-cabeça"
# Extremos na forma de função
    L = len(M)
    U = len(M[0])
    Q = sortmatriz(M,2)
    E = []
    for lin in range(L):
        E.append([])
        for col in range(U):
            E[lin].append(Q[-lin-1][col])
    F = integ_rec2(f,E,L)
    Y = F(P)
    return Y


def integ_Nfull2(f,M):
# A função calcula a integral em N variáveis de uma função
# de N variáveis. A matriz M é da forma [[a1,b1],...,[aN,bN]]
# onde ai,bi, i = 1,..,N são os extremos de integração da
# variável na posição (i-1) em forma de função, a ordem dos
# [ai,bi] em M é da integral mais interna até a mais externa. 
    L = len(M)
    E = []
    for lin in range(L):
        E.append([])
        for col in range(len(M[0])):
            E[lin].append(M[-lin-1][col])
    K = E[1:L]
    for i in range(L-1):
        K[i].append(i+1)
    def r(x):
        X = []
        X.append(x)
        y = integ_Ninc2(f,K,X)
        return y
    I = integ1(r,E[0])
    return I

### INTEGRAIS IMPROPRIAS ### 

def integinfmais1(f,a):
    epsilon = 0.001
    boole = True
    delta = 1
    R = [a,a+delta]
    I = integ1(f,R)
    while boole:
        deltatmp = delta
        delta = delta + 1
        R = [a+deltatmp,a+delta]
        dI = integ1(f,R)
        I = I + dI
        if abs(dI) < epsilon:
            boole = False
    return I


def integinfmenos1(f,b):
    epsilon = 0.001
    boole = True
    delta = 1
    R = [b-delta,b]
    I = integ1(f,R)
    while boole:
        deltatmp = delta
        delta = delta + 1
        R = [b-delta,b-deltatmp]
        dI = integ1(f,R)
        I = I + dI
        if abs(dI) < epsilon:
            boole = False
    return I


def integinf1(f): 
    I1 = integinfmais1(f,0)
    I2 = integinfmenos1(f,0)
    I = I1 + I2
    return I


def integinf(f,P,i):
# P é o ponto que informa o valor que assumem 
# as (N-1) variáveis que não são
# de integração. variável i, i=0,...,N-1
# informa a posição da variável em que se deseja 
# integrar a função f, de N variáveis.
    def F(x):
        X = completavetor(P,x,i)
        y = f(X)
        return y   
    I = integinf1(F)
    return I


def integinf_func(f,i):
    def F(X):
        y = integinf(f,X,i)
        return y
    return F

def integinf_rec(f,M,i):
# M é da forma [v0,...,v(N-1)]
# onde vi são as variáveis sobre as quais deseja-se 
# integrar.
    if i == 0:
        return f
    else:
        x = M[i-1]
        h = integinf_rec(f,M,i-1)
        F = integinf_func(h,x)
        return F

def integinf_Ninc(f,M,P):
# A função calcula a integral em N variáveis de uma função
# de m variáveis, dado que m>N, em um ponto P.
    L = len(M)
    F = integinf_rec(f,M,L)
    Y = F(P)
    return Y

def integinf_Nfull(f,N):
# A função calcula a integral em N variáveis de uma função
# de N variáveis.
    L = N - 1
    V = []
    for i in range(L):
        V.append(i)
    F = integinf_rec(f,V,L)
    F = funcao1var(F)
    I = integinf1(F)
    return I

### INTEGRAIS DE VETOR ###
### versao vetorial das integrais da secao anterior

def integ1vet(v,T):
    a,b = T[0],T[1]
    L = len(v(a))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integ1(f,a,b)
        F.append(I)
    return F    

def integvet(v,P,T,i):
    a,b = T[0],T[1]
    X = completavetor(P,a,i)
    K = len(v(X))
    F = []
    for num in range(K):
        f = funcao_vetor(v,num)
        I = integ(f,P,T,i)
        F.append(I)
    return F

def integ_funcvet(v,T,i):
    a,b = T[0],T[1]
    def F(X):
        Y = integvet(v,X,T,i)
        return Y
    return F

def integ_Nincvet(v,A,P):
    M = sortmatriz(A,2)
    U = len(M)
    R = P
    for k in range(U):
        x = M[k][0]
        p = M[k][2]
        R = completavetor(R,x,p)
    L = len(v(R))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integ_Ninc(f,M,P)
        F.append(I)
    return F

def integ_Nfullvet(v,A):
    R = []
    for num in range(len(A)):
        R.append(A[num][0])
    L = len(v(R))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integ_Nfull(f,A)
        F.append(I)
    return F

def integ_Nfull2vet(v,A):
    R = []
    for num in range(len(A)):
        R.append(A[num][0])
    L = len(v(R))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integ_Nfull2(f,A)
        F.append(I)
    return F

def integinf1vet(v):
    L = len(v(0))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integinf1(f)
        F.append(I)
    return F

def integinfvet(v,P,k):
    a = completavetor(P,0,k)
    L = len(v(a))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integinf(f,P,k)
        F.append(I)
    return F

def integinf_Nincvet(v,A,P):
    M = sort(A)
    U = len(A)
    a = P
    for k in range(U):
        q = M[k]
        a = completavetor(a,0,q)
    L = len(v(a))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integinf_Ninc(f,A,P)
        F.append(I)
    return F

def integinf_Nfullvet(v,N):
    a = []
    for num in range(N):
        a.append(0)
    L = len(v(a))
    F = []
    for i in range(L):
        f = funcao_vetor(v,i)
        I = integinf_Nfull(f,N)
        F.append(I)
    return F


### INTEGRAIS DE LINHA E DE SUPERFÍCIE, FLUXO ###

def integlin(F,gamma,I):
# F é o campo vetorial (R^n em R^n)
# gamma é a curva em forma de função parametrizada (R em R^n)
# se I = [a,b], a e b são os extremos de integração (1D)
    def f(t):
        D = derivvet(gamma,t)
        X = gamma(t)
        y = ProdEsc(D,F(X))
        return y
    integral = integ1(f,I)
    return integral


## INTEGRAL DE LINHA ALTERNATIVA NUMA CURVA COPLANAR 
## NO PLANO CARTESIANO

def integlinplano(F,curva):
# curva é uma lista de N pontos (matriz Nx2)
# com todos os pontos que pertencem a curva
# a ordem dela indica a trajetória na 
# integral. Sentido (0 ou 1)
# determina se vai do primeiro ponto ao
# último (0) ou o contrário (1).
    L = len(curva)
    C = curva
    S = 0
    for num in range(L-1):
        ds = vet_Subtracao(C[num+1],C[num])
        S = S + ProdEsc(F(C[num]),ds)
    return S




def integsup(f,sigma,A):
# F é o campo vetorial ou escalar (R^3 em R ou R^n)
# sigma é a superfície em forma de função parametrizada (R^2 em R^3)
# A é uma matriz na forma A = [[u0,u1],[v0,v1]], onde u e v são
# os parâmetros da superfície.
    boole = True
    if type(f(sigma(A)))==type([]):
        boole = False
    def funcao(X):
        U,P = derivparcvet(sigma,X,0),derivparcvet(sigma,X,1)
        W = Norma(ProdVet(U,P))
        if boole:
            return f(sigma(X))*W
        else:
            return ProdNum(f(sigma(X)),W)
    if boole:
        return integ_Nfull(funcao,A)
    else:
        return integ_Nfullvet(funcao,A)


def fluxo(F,sigma,A):
    def f(X):
        U,P = derivparcvet(sigma,X,0),derivparcvet(sigma,X,1)
        W = ProdVet(U,P)
        return ProdEsc(F(sigma(X)),W)
    return integ_Nfull(f,A)
       
    

### OPERADORES DIFERENCIAIS ###


def gradiente(f,P):
    grad = []
    for i in range(len(P)):
        G = derivparc(f,P,i)
        grad.append(G)
    return grad


def rotacional(F,P):
    rot = []
    M = [[1,2],[0,2],[0,1]]
    B = [False,True]
    for i in range(3):
        K = 0
        a,b = M[i][0],M[i][1]
        for u in range(2):
            if B[u]:
                tmp = b
                b = a
                a = tmp
            f = funcao_vetor(F,b)
            K = K + ((-1)**u)*derivparc(f,P,a)
        if i == 1:
            K = (-1)*K
        rot.append(K)
    return rot


def divergente(F,P):
    div = 0
    for i in range(len(P)):
        f = funcao_vetor(F,i)
        div = div + derivparc(f,P,i)
    return div


def laplaciano(f,P):
    def grad(P):
        g = gradiente(f,P)
        return g
    lapl = divergente(grad,P)
    return lapl


def laplacianovet(F,P):
    L = len(F(P))
    lapl = []
    for i in range(L):
        f = funcao_vetor(F,i)
        lapl.append(laplaciano(f,P))
    return lapl


# --
def jacobiano(v,x):
    J = []
    L = len(v(x))
    M = len(x)
    for i in range(L):
        J.append([])
        f = funcao_vetor(v,i)
        for j in range(M):
            F = derivparc(f,x,j)
            J[i].append(F)
    return J

def jacobianodet(v,x):
    J = det(jacobiano(v,x))
    return J


def teste_erro_digitacao(M,passo):
    return True


def dominio(M,passo):
# 1) "Matriz" M deve ser escrita na forma
# [[a_1,b_1],[a_2,b_2],...,[a_n,b_n],[c_1,...,c_l]]
# onde [a_i,b_i], i = 1,..,n são intervalos de extremidades 
# fechadas nos reais, se a_i,b_i finitos, mas a_i pode ser
# escrito como a string '-inf' e b_i como '+inf', se o
# intervalo começar em -(infinito) e +(infinito), 
# respectivamente. c_1,...,c_l são os pontos a serem
# excluidos do intervalo final, que será a união dos
# intervalos digitados. Os intervalos precisam ter
# intersecção nula entre si, ou seja, se i != j, i,j em
# {1,...,n} temos [a_i,b_i](intersecção)[a_j,b_j] = (vazio)
# ou seja, no final, devemos ter 
# (união(i=1,n))([a_i,b_i]) \ (união(m=1,l))({c_m})
# 2) Para todo i = 1,..,n, deve ter (b_i-a_i)%passo == 0,
# ou seja, deve ser possível escrever b_i como a_i + N*passo,
# onde N é um número natural.
    boole0 = False
    entradavalida = teste_erro_digitacao(M,passo)
    if entradavalida:
        L = len(M)
        matriz = []
        vetor = []
        l = 0
        for i in range(L-1):
            a = '-inf' in M[i]
            b = '+inf' in M[i]
            if not(a or b):
                N = int((M[i][1]-M[i][0])/passo)
                matriz.append([])
                for num in range(N+1):
                    R = M[i][0] + num*passo
                    c = R in M[L-1]
                    if not(c):
                        matriz[i-l].append(R)
            else:
                vetor.append(i)
                l = l+1 
        T = len(vetor)
        A = []
        K = 0
        for u in range(T):
            j = vetor[u]
            A.append(M[j])
            t = A[u][0] in M[L-1]
            d = A[u][1] in M[L-1]
            if t:
                K = 1
                A[u][0] = A[u][0] + passo
            elif d:
                A[u][1] = A[u][1] - passo        
        r = L - T - 1
        if r != 0:
            A.append([])
            for m in range(r):
                y = len(matriz[m])
                for n in range(y):
                    A[T].append(matriz[m][n])
            A[T] = sort(A[T])
        boole0 = True
    if boole0: 
        return A


'''
Fazer:
LIMITES:
limites laterais -feito!-
limites no infinito -feito!-
limites infinitos

DERIVADAS:
derivadas parciais -feito!-
derivadas n-esimas -feito!-
derivadas parciais n-esimas -feito!-
derivadas parciais n-esimas cruzadas -feito!-
derivadas de vetor:
 n-esimas -feito!-
 parciais -feito!-
 parciais n-esimas -feito!-
 parciais n-esimas cruzadas -feito!-
derivadas direcionais
operadores diferenciais:
 gradiente -feito!-
 rotacional -feito!-
 divergente -feito!-
 laplaciano -feito!-

INTEGRAIS:
integral -feito!-
integral n-esima -feito!*-
integral impropria -feito!*-
integral n-esima impropria -feito!*-
integral vetorial -feito!*-
integral n-esima vetorial -feito!*-
integral impropria vetorial -feito!*-
integral n-esima impropria vetorial -feito!*-
integrais de linha -feito!*-
integrais superfície/fluxo -feito!*-
jacobiano -feito!-
serie de fourier -feito!-
produto interno de funcoes

* - falta testar
** - refazer
'''
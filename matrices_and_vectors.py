from math import *

def matrizid(n):
    I = []
    for i in range(n):
        I.append([])
        for j in range(n):
            if i == j:
                I[i].append(1)
            else:
                I[i].append(0)
    return I

def matrizcte(m,n,K):
    A = []
    for i in range(m):
        A.append([])
        for j in range(n):
            A[i].append(K)
    return A

def transposta(M):
    A = []
    lin = len(M)
    col = len(M[0])
    for i in range(col):
        A.append([])
        for j in range(lin):
            A[i].append(M[j][i])
    return A

def fat(n):
    if n in range(2):
        return 1
    else:
        N = n
        for i in range(n-1):
            N = N*(n-1-i)
        return N


def imprimematriz(A):
    m = len(A)
    n = len(A[0])
    for i in range(m):
        for j in range(n):
            print(A[i][j],end = '\t')
        print('\n')


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

def ehSimetrica(A):
    return transposta(A) == A

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


def esq_Sarrus(A):
    B = []
    for n in range(len(A)):
        B.append([])
    for i in range(len(A)):
        for j in range(len(A)-1):
            B[i].append(A[i][j])    
    B = ampliada(A,B)
    return B


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


def det_2x2(A):
    V = (DP(A),DS(A))      
    lista = []
    for num1 in range(2):
        PROD = 1
        for num2 in range(2):
            PROD = V[num1][num2]*PROD
        lista.append(PROD)
    return lista[0]-lista[1]


def det_1x1(A):
    return A[0][0]



def det_triang(A):
    W = DP(A)
    n = len(W)
    det = 1
    for num in range(n):
        det = det*W[num]
    return det


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
        
        


def cofator4x4(matriz4x4,i,j):
    cof = ((-1)**(i+j))*det_3x3(submatriz(matriz4x4,i,j))        
    return cof

def cofatorNXN(matriz,i,j):
    cof = ((-1)**(i+j))*det(submatriz(matriz,i,j))     
    return cof


def det_laplacelinha4x4(A,i):
    soma = 0
    for j in range(4):
        soma = soma + A[i-1][j]*cofator(A,i,j+1)
    return soma



'''Para matrizes M maiores que 4x4:
Separamos em duas etapas:
1) O algoritmo deve calcular paralelamente as submatrizes sucessivas da matriz
M dada, armazenando-as numa lista que será atualizada a cada redução, até
que a ordem da última submatriz seja 3. Ao final, se N é a ordem da matriz M,
deve apresentar uma lista com N!/3! elementos. Por exemplo, se N é igual a 5,
deve armazenar 5 matrizes 4x4, contendo uma para cada entrada da primeira linha
na ordem correspondente, e depois limpar a lista e fazer o mesmo para cada matriz
4x4, até obter 20 matrizes 3x3. 

2) Deve então calcular o determinante de cada uma dessas matrizes 3x3 e armazena-
los numa lista e, a partir dessa lista, calcular N!/4! determinantes das 
matrizes 4x4 formadas por cada 4 matrizes 3x3 e depois N!/5! determinantes das matrizes
5x5 formadas por cada grupo de 5 matrizes 4x4, e assim sucessivamente, até chegar
em uma lista com 1 determinante correspondente ao da matriz M.
'''

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
            
            
                    
def tr(A):
    if ehQuadrada(A):
        x = DP(A)
        a = 0
        for num in range(len(x)):
            a = a + x[num]
        return a                      


def produtomatricial(A,B):
    produtoexiste = False
    if not(len(A) == 0 or len(B) == 0) and len(A[0]) == len(B):
        produtoexiste = True
    if produtoexiste:
        C = []
        for i in range(len(A)):
            C.append([])
            for j in range(len(B[0])):
                SOMA = 0
                for k in range(len(B)):
                    a = A[i][k]*B[k][j]
                    SOMA = a + SOMA
                C[i].append(SOMA)
    return C

def mult(escalar,matriz):
    m = len(matriz)
    n = len(matriz[0])
    matriznova = []
    for i in range(m):
        matriznova.append([])
        for j in range(n):
            matriznova[i].append(escalar*matriz[i][j])
    return matriznova

def adj(A):
    p = len(A)
    cofatores = []
    for i in range(p):
        cofatores.append([])
        for j in range(p):
            cofatores[i].append(cofatorNXN(A,i+1,j+1))         
    adjunta = transposta(cofatores)
    return adjunta
           

'''
A função abaixo det(A) é a 'main' das funções
de determinante
'''

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


'''
A função abaixo calcula a inversa de qualquer matriz NxN
'''

def inversa(A):
    determinante = det(A)
    B = adj(A)
    inv = mult((1/determinante),B)     
    return inv

def ehIgual(A,B):
    epsilon = 0.000000001
    boole = True
    x1 = len(A)
    x2 = len(B)
    y1 = len(A[0])
    y2 = len(B[0])
    if x1 != x2 or y1 != y2:
        boole = False
    else:
        for i in range(x1):
            for j in range(y1):
                if boole:
                    u = abs(A[i][j]-B[i][j])
                    if u > epsilon:
                        boole = False
    return boole



def vet_ProdEsc(x,y):
    L = len(x)
    S = 0
    for i in range(L):
        S = S + x[i]*y[i]
    return S

def multv(matriz,vetor):
    l = len(matriz) 
    v = []
    for i in range(l):
        x = vet_ProdEsc(matriz[i],vetor)
        v.append(x)
    return v


###
#SOLUÇÃO DE SISTEMAS LINEARES POSSÍVEIS E DETERMINADOS 
#USANDO MATRIZ INVERSA (método de Cramer)
#SISTEMA LINEAR:
#[MATRIZ]*[VETOR1] = [VETOR2]
#MATRIZ, VETOR2 - DADOS
#VETOR1 - INCÓGNITA


def sol_SL(matriz,resultado):
    matrizinv = inversa(matriz)
    solucao = multv(matrizinv,resultado)
    return solucao


###
def sort(v):
    N = len(v)
    for i in range(N-1):
        for j in range(i+1,N):
            if v[i] > v[j]:
                temp = v[i]
                v[i] = v[j]
                v[j] = temp
    return v

def ehInversa(A,B):
    epsilon = 0.001
    ehInversa = True
    l = len(A)
    I = matrizid(l)
    C = produtomatricial(A,B)
    for i in range(l):
        for j in range(l):
            if abs(C[i][j]-I[i][j]) > epsilon:
                ehInversa = False    
    return ehInversa


def existeInversa(A):
    epsilon = 0.001
    existe = True
    if abs(det(A)-0) < epsilon:
        existe = False
    return existe




def elementar_1(linha1,linha2,ordem):
    m = linha1 - 1
    n = linha2 - 1
    I = matrizid(ordem)
    E = []
    for i in range(ordem):
        E.append([])
        if i == m:
            for j in range(ordem):
                E[i].append(I[n][j])
        elif i == n:
            for j in range(ordem):
                E[i].append(I[m][j]) 
        else:
            for j in range(ordem):
                E[i].append(I[i][j])
    return E    
     


def elementar_2(linha,escalar,ordem):
    m = linha - 1
    I = matrizid(ordem)
    E = []
    for i in range(ordem):
        E.append([])
        if i == m:
            for j in range(ordem):
                E[i].append(escalar*I[m][j])
        else:
            for j in range(ordem):
                E[i].append(I[i][j])
    return E

def elementar_3(linha1,linha2,escalar,ordem):
    m = linha1 - 1
    n = linha2 - 1
    I = matrizid(ordem)
    E = []
    for i in range(ordem):
        E.append([])
        if i == m:
            for j in range(ordem):
                E[i].append(I[m][j] + escalar*I[n][j])
        else:
            for j in range(ordem):
                E[i].append(I[i][j])    
    return E

def soma(A,B):
    m = len(A)
    n = len(A[0])
    matriznova = []
    for i in range(m):
        matriznova.append([])
        for j in range(n):
            matriznova[i].append(A[i][j]+B[i][j])
    return matriznova



#Soluções para equações matriciais:

'''
EQ. do tipo
aX + B = C
'''

def sol_eq(matrizesq,matrizdir,escalar):
    escalarinv = 1/escalar
    matriznova = soma(matrizdir,mult(-1,matrizesq))
    solucao = mult(escalarinv,matriznova)
    return solucao

'''
VETORES
'''


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

def ProdEsc(x,y):
    L = len(x)
    S = 0
    for i in range(L):
        S = S + x[i]*y[i]
    return S

def ProdVet(x,y):
    MATRIZ = [[0,0,0],x,y]
    ProdV = []
    for i in range(3):
        COORD = ((-1)**i)*det_2x2(submatriz(MATRIZ,1,i+1))
        ProdV.append(COORD)
    return ProdV

def ProdNum(vetor,numero):
    L = len(vetor)
    produto = []
    for i in range(L):
        produto.append(vetor[i]*numero)
    return produto

def Norma(x):
    Norma = sqrt(ProdEsc(x,x))
    return Norma

def vet_ehIgual(u,v):
    epsilon = 0.000000001
    b = True
    L1 = len(u)
    L2 = len(v)
    if L1 != L2:
        b = False
    else:
        for i in range(L1):
            if abs(u[i]-v[i]) > epsilon:
                b = False
    return b

def multv(matriz,vetor):
    l = len(matriz) 
    v = []
    for i in range(l):
        x = ProdEsc(matriz[i],vetor)
        v.append(x)
    return v


'''
QUADRADOS MÁGICOS

EXEMPLOS:

CLÁSSICOS:
[[16,3,2,13],[5,10,11,8],[9,6,7,12],[4,15,14,1]]
[[2,7,6],[9,5,1],[4,3,8]]
[[6,7,2],[1,5,9],[8,3,4]]
[[4,9,2],[3,5,7],[8,1,6]]
[[2,9,4],[7,5,3],[6,1,8]]
[[4,3,8],[9,5,1],[2,7,6]]
[[8,3,4],[1,5,9],[6,7,2]]
[[8,1,6],[3,5,7],[4,9,2]]
[[6,1,8],[7,5,3],[2,9,4]]

NÃO-CLÁSSICOS:
[[9,17,10],[13,12,11],[14,7,15]]
[[4,12,5],[8,7,6],[9,2,10]]
[[8,15,7],[9,10,11],[13,5,12]]
[[3,10,5],[8,6,4],[7,2,9]]
[[12,19,8],[9,13,17],[18,7,14]]

'''



def wt(QuadradoMagico):
    peso = tr(QuadradoMagico)
    return peso



def ehQuadradoMagico(A):
    if ehQuadrada(A):
        ehQM = True
        B = []
        a = tr(A) 
        diagsec = DS(A)
        m = len(diagsec)
        b = 0
        for i1 in range(m):
            b = b + diagsec[i1]
        B.append(b)
        for i2 in range(m):
            c = 0
            for j in range(m):
                c = c + A[i2][j]
            B.append(c)
        for NUM1 in range(m):
            d = 0
            for NUM2 in range(m):
                d = d + A[NUM2][NUM1]
            B.append(d)           
        for NUM3 in range(2*m+1):
            if a != B[NUM3]:
                ehQM = False
        return ehQM


def ehQuadradoMagicoClassico(A):
    if ehQuadradoMagico(A):
        ehQMC = False
        N = len(A)
        B = []
        for m in range(N):
            for n in range(N):
                B.append(A[m][n])
        B = sort(B)
        C = list(range(1,N**2+1))
        if B==C:
            ehQMC = True
        return ehQMC

'''
Fazer algoritmo:
-em que o usuário realiza operações 
elementares sobre uma matriz-FEITO-
-que verifica igualdade entre dois
vetores -FEITO-
-com operações com vetores -FEITO-
-módulo de um vetor -FEITO-
'''
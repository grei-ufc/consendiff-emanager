############################ Definicao dos Agentes ################################

''' 
Dados do Problema:
    
Agente 00: Carga 
Agente 01: Renovaveis
Agente 02: Geracao Classica
Agente 03: Bateria
Agente 04: PCC
'''

################################# Importando Bibliotecas ###################################

print("Iniciando codigo")

import numpy as np
import matplotlib.pyplot as plt
import time

################################# Modelando os cinco agentes ###########################
#Identificacao
Carga=[0]
Ren=[1]
Desp=[2]
Bat=[3]
Pcc=[4]

#Agente Bateria
pmaxess = 40
alp_ess = 0.3
beta = 114
soc=0.9
bet_ess = alp_ess*pmaxess*(1-soc) + beta

#Agente Carga
Dem=np.array([40,0,0,0,0])

#Agente Geracao Renovavel
P_ren=np.array([0,10,0,0,0])

#Demanda liquida
P_carga=np.array([30,0,0,0,0])

#Parametros de Custo
alp = np.array([0,0,0.18,0.3,0]) 
bet = np.array([1,1,97,bet_ess,112.5])

#Numero de agentes despachaveis:
nDG=2

#Numero de Agentes:
nG = len(alp)


################## Inicializacao dos Parametros do Consenso  ###################

epil = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

#Matriz de Adjacencias:
A = np.array([[0,1,0,0,1],
              [1,0,1,0,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [1,0,0,1,0]])

#Matriz de Grau:
D=np.diag(np.sum(A, axis=1))   

#Matriz Laplaciana:
L = D - A  

#Matriz Identidade:
I=np.identity(len(A))

#Matriz Mean Metropolis:
MM=np.zeros([nG,nG])
for i in range (0,nG):
    for j in range(0,nG):
        if(i!=j):
             MM[i,j] = 2./(D[i,i] + D[j,j] + 1)
MM=np.multiply(A,MM)
for i in range(0,nG):
    MM[i,i] = 1 - sum(MM[i,:])

#Numero Maximo de Iteracoes:
N_max=15000
N_max+=1 
  
#Parametro de Parada:
diff=10                           # Diferenca inicial
diff_min=0.0001                   # Diferenca minima de convergencia


n_algorit=0
flag=np.zeros(nG)

Plim_sup = np.array([0,0,50,(soc*pmaxess),0])
Plim_inf = np.array([0,0,0,-(1-soc)*pmaxess,0])
#Agentes que ultrapassaram a Potencia Maxima:
P_u_max=[]    

#Agentes que ultrapassaram a Potencia Minima:
P_u_min=[]

#Agentes que nao ultrapassaram os limites de potencia :
P_n=[]

############################ Algoritmo de Consenso ##################################
tempo1=time.time()
while(sum(flag)!=0 or n_algorit==0):
    #Inicializando as Potencias dos Agente:
            
    #Numero Maximo de Iteracoes:
    N_max=15000
    N_max+=1 
    Pg=np.zeros([N_max,nG])
    
    #Custo Incremental:
    r = np.zeros([N_max,nG]) 
    
    #Inicializando o Custo Incremental dos Agente:

    #Agente PCC
    for i in Pcc:
        r[0,i]=bet[i]
        
    #Agente Renovavel:   
    for i in Ren:
        r[0,i]=0
        
    #Agente Carga:
    for i in Carga:
        r[0,i]=0
        
    #Agente Despachavel:
    for i in Desp:
        r[0,i]=bet[i]
    
    #Agente Bateria:
    for i in Bat:
        r[0,i]=bet[i]
    
    #Inicializando o Power Mismatch:
    Pd=np.zeros([N_max,nG])
    Pd[0]=sum(P_carga)/nG

    #Flag para rodar o while
    flag=np.zeros(nG)
    #Critério de Parada:
    diff=10  
    i=0   
    while(i!=N_max-1 and diff>diff_min):     
        d=np.zeros(nG)
        for j in range(0,len(MM)):
            if (j!=4):
                r[i+1,j] = MM[j,:]@r[i,:] + epil[j]*Pd[i,j]
            if (j==4):
                r[i+1,j] = 112.5  
            if j in Ren:
                Pg[i+1,j]=0
            if j in Carga:
                Pg[i+1,j]=0
            if j in Bat:
                if j in P_n or n_algorit==0:
                    Pg[i+1,j] = (r[i+1,j] - bet[j])/(2*alp[j])
                elif j in P_u_max:
                    Pg[i+1,j]=Plim_sup[j]
                elif j in P_u_min:
                    Pg[i+1,j]=Plim_inf[j]
            if j in Desp:
                if j in P_n or n_algorit==0:
                    Pg[i+1,j] = (r[i+1,j] - bet[j])/(2*alp[j]) 
                elif j in P_u_max:
                    Pg[i+1,j]=Plim_sup[j]
                elif j in P_u_min:
                    Pg[i+1,j]=Plim_inf[j]
            if j in Pcc:
                Pg[i+1,j] = sum(P_carga)-((Pg[i+1,2] + Pg[i+1,3]))
            
            Pd[i+1,j]=Pd[i,:]@MM[j,:] - (Pg[i+1,j] - Pg[i,j])
        d=abs(r[i+1]-r[i])
        if(i==0):
            diff=1
        else:
            diff=max(d) 
        i+=1
#Corte do i:
    r=r[:i,:]  
    Pg=Pg[:i,:]
    inter=i
    print("Pg:",Pg[-1])
    #Limites de Potencia
    for i in range(0,nG):
        if(Pg[-1,i]>=Plim_sup[i]):
            if i in Desp:
                if n_algorit==0:
                    P_u_max.append(i)
                    flag[j]=1
                else:
                    if i in P_n:
                        P_n.remove(i)
                        P_u_max.append(i)
                        flag[i]=1
                    elif i in P_u_min:
                        P_u_min.remove(i)
                        P_u_max.append(i)
                        flag[i]=1
            if i in Bat:
                if n_algorit==0:
                    P_u_max.append(i)
                    flag[i]=1
                else:
                    if i in P_n:
                        P_n.remove(i)
                        P_u_max.append(i)
                        flag[i]=1
                    elif i in P_u_min:
                        P_u_min.remove(i)
                        P_u_max.append(i)
                        flag[i]=1
        elif(Pg[-1,i]<=Plim_inf[i]):
            if i in Desp:
                if n_algorit==0:
                    P_u_min.append(i)
                    flag[i]=1
                else:
                    if i in P_n:
                        P_n.remove(i)
                        P_u_min.append(i)
                        flag[i]=1
                    elif i in P_u_max:
                        P_u_max.remove(i)
                        P_u_min.append(i)
                        flag[i]=1
            if i in Bat:
                if n_algorit==0:
                    P_u_min.append(i)
                    flag[i]=1
                else:
                    if i in P_n:
                        P_n.remove(i)
                        P_u_min.append(i)
                        flag[i]=1
                    elif i in P_u_max:
                        P_u_max.remove(i)
                        P_u_min.append(i)
                        flag[i]=1
        else:
            if i in P_n:
                pass #
            elif i in P_u_max:
                P_u_max.remove(i)    
                P_n.append(i)
            elif i in P_u_min:
                P_u_min.remove(i)
                P_n.append(i)
            else:
                P_n.append(i)  
# Soma das potencias dos agentes despachaveis (verificar equilibrio de potencia) 
    P_sup=0
    for  a in P_u_max:
        P_sup += Plim_sup[a]
    P_inf=0
    for a in P_u_min:
        P_inf += Plim_sup[a]
    if P_sup > sum(Carga):
        flag_sem_limites=0
        flag_limite_superior =1
    n_algorit+=1
    print("flag:",flag)
tempo2=time.time()
print("\nTempo Consenso com restricao:",tempo2-tempo1)    
print("\n Parou na interacao:",inter) 
       
r=r[:inter,:]  #Vai cortar a matriz até a parte util,se parar por diferenca
Pg=Pg[:inter,:]
soc = (soc*40 + Pg[-1, 3])/40

######################## Exibindo os Dados sem restricoes na tela ###########################

print("\n A potencia de cada agente quando nao ha restricoes:",Pg[-1])
print("\n O custo incremental quando nao ha restricoes sera:",round(max(r[-1]),2))
print("\n O custo incremental quando nao ha restricoes sera:", r[-1])
print("\n ###################### ")
############################# Plotando os graficos ##############################
for i in range(0,nG):
    plt.plot(r[:,i],label=i+1)
plt.legend()
plt.title("Cenario 5-Consenso")
plt.xlabel("Iteracao")
plt.ylabel("Custo Incremental")
plt.grid()
plt.show()

for i in range(0,nG):
    plt.plot((Pg[:,i])*-1,label=i+1)
plt.legend()
plt.title("Power in each agent")
plt.xlabel("Iteration")
plt.ylabel("Power in each agent")
plt.grid()
plt.show()

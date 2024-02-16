############################ Definição dos Agentes ################################

''' 
Dados do Problema:
    
Agente 00: Carga 
Agente 01: Renováveis
Agente 02: Geração Clássica
Agente 03: Bateria
Agente 04: PCC
'''

################################# Importando Bibliotecas ###################################

print("Iniciando código")

import numpy as np
import matplotlib.pyplot as plt
import time
import Matrizes_de_Pesos2 as mp

################################# Modelando os cinco agentes ###########################
#Identificação
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

#Agente Geração Renovável
P_ren=np.array([0,10,0,0,0])

#Demanda líquida
P_carga=np.array([30,0,0,0,0])

#Parâmetros de Custo
alp = np.array([0,0,0.18,0.3,0]) 
bet = np.array([1,1,97,bet_ess,112.5])

#Número de agentes despacháveis:
nDG=2

#Número de Agentes:
nG = len(alp)


################## Inicialização dos Parâmetros do Consenso  ###################

epil = np.array([0.6227639, 0.79950991, 0.97012083, 0.94810546, 0.70734475])

#Matriz de Adjacências:
A = np.array([[0,1,0,0,1],
              [1,0,1,0,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [1,0,0,1,0]])

MM,epil2=mp.Mean_Metropolis(A,epil)


#Número Máximo de Iterações:
N_max=15000
N_max+=1 
  
#Parâmetro de Parada:
diff=10                           # Diferença inicial
diff_min=0.0001                   # Diferença mínima de convergência


n_algorit=0
flag=np.zeros(nG)

Plim_sup = np.array([0,0,50,(soc*pmaxess),0])
Plim_inf = np.array([0,0,0,-(1-soc)*pmaxess,0])
#Agentes que ultrapassaram a Potência Máxima:
P_u_max=[]    

#Agentes que ultrapassaram a Potência Mínima:
P_u_min=[]

#Agentes que não ultrapassaram os limites de potência :
P_n=[]

############################ Algoritmo de Difusão sem Restrições ##################################
tempo1=time.time()
while(sum(flag)!=0 or n_algorit==0):
    #Inicializando as Potências em kda Agente:
            
    #Número Máximo de Interações:
    N_max=15000
    N_max+=1 
    Pg=np.zeros([N_max,nG])
    
    #Custo Incremental:
    r = np.zeros([N_max,nG]) 
    
    #Inicializando o Custo Incremental em kda Agente:
    #Agente PCC
    for i in Pcc:
        r[0,i]=bet[i]
        
    #Agente Renovável:   
    for i in Ren:
        r[0,i]=0
        
    #Agente Carga:
    for i in Carga:
        r[0,i]=0
        
    #Agente Despachável:
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
    #Váriaveis de difusão
    var=np.zeros([N_max,nG])
    var[0]=r[0]
    i=0
    flag_sem_limites = 1
    flag_limite_superior = 0
    #Difusão 
    while(i!=N_max-1 and diff>diff_min):
        d=np.zeros(nG)
        for j in range (0,len(MM)):
            if (j!=4):
                r[i+1,j]=MM[j,:]@var[i,:]
            if (j==4):
                r[i+1,j]=112.5
            var[i+1,j]=r[i+1,j]-epil[j]*(sum(Pg[i])-sum(P_carga))        
            for j in Ren:
                Pg[i+1,j]=0
            for j in Carga:
                Pg[i+1,j]=0
            for j in Bat:
                if j in P_n or n_algorit==0:
                    Pg[i+1,j] = (r[i+1,j] - bet[j])/(2*alp[j])
                elif j in P_u_max:
                    Pg[i+1,j]=Plim_sup[j]
                elif j in P_u_min:
                    Pg[i+1,j]=Plim_inf[j]
            for j in Desp:
                if j in P_n or n_algorit==0:
                    Pg[i+1,j] = (r[i+1,j] - bet[j])/(2*alp[j]) 
                elif j in P_u_max:
                    Pg[i+1,j]=Plim_sup[j]
                elif j in P_u_min:
                    Pg[i+1,j]=Plim_inf[j]
            for j in Pcc:
                Pg[i+1,j] = sum(P_carga)-((Pg[i+1,2] + Pg[i+1,3]))           
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
    #Limites de Potência
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
                pass #Não fazer nd
            elif i in P_u_max:
                P_u_max.remove(i)    
                P_n.append(i)
            elif i in P_u_min:
                P_u_min.remove(i)
                P_n.append(i)
            else:
                P_n.append(i)  # Se n_algorit==0 , cai nessa condicional
# Vai pegar a soma das potências dos agentes despacháveis 
    P_sup=0
    for  a in P_u_max:
        P_sup += Plim_sup[a]
#Vai pegar a soma dos complementares da lista P_u_min e verificar se seus 
#limites máximos realmente irão obedecer a restrição de potência.
    P_inf=0
    for a in P_u_min:
        P_inf += Plim_sup[a]
    if P_sup > sum(P_carga):
        flag_sem_limites=0
        flag_limite_superior =1
    n_algorit+=1
    print("flag:",flag)
tempo2=time.time()
print("\nTempo Consenso com restrição:",tempo2-tempo1)    
print("\n Parou na interação:",inter)
#Colocar isso dentro do algoritmo         
r=r[:inter,:]  #Vai cortar a matriz até a parte útil,se parar por diferença
Pg=Pg[:inter,:]

######################## Exibindo os Dados sem restrições na tela ###########################

print("\n A potência de cada agente quando não há restrições:",Pg[-1])
print("\n O custo incremental quando não há restrições será:",round(max(r[-1]),2))
print("\n O custo incremental quando não há restrições será:", r[-1])
print("\n ###################### ")
############################# Plotando os gráficos ##############################

for i in range(0,nG):
    plt.plot(r[:,i],label=i+1)
plt.legend()
plt.title("Cenário 5-Difusão")
plt.xlabel("Iteração")
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



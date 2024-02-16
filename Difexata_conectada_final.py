############################ Corte de Carga sem Restrições ################################

''' 
Dados do Problema:
    
Agente 00: Carga 
Agente 01: Renováveis
Agente 02: Geração Clássica
Agente 03: Bateria
Agente 04: PCC
'''

################################# Importando Bibliotecas ##########################################

import numpy as np
import matplotlib.pyplot as plt
import Matrizes_de_Pesos2 as mp
import time

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
'''
#Episolon ótimo usando a Hasting_Rules:
epil=np.array([0.3864533, 0.25327445,  0.50653835, 0.66691278, 0.71372608])
'''

#Episolon ótimo usando a Averaging_Rule:
epil=np.array([0.98452443, 0.93130314,  0.75564272, 0.94805752, 0.97524851])

'''
epil = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
'''
'''
#Episolon ótimo usando a Relative_degree_rule:
epil=np.array([0.59286538, 0.7220888,  0.9448206, 0.89829282, 0.62444724])
'''
'''
#Episolon ótimo usando a Mean_Metropolis:
epil=np.array([0.90739497, 0.90598153, 0.83881186, 0.98911294, 0.2532484]) 
'''

#Matriz de Adjacências:
A = np.array([[0,1,0,0,1],
              [1,0,1,0,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [1,0,0,1,0]])

#Matriz Identidade:
I=np.identity(len(A))

MM,epil2=mp.Averaging_Rule(A,epil)
MMi=(MM+I)/2

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
    #Agente Pcc:
    for i in Pcc:
        r[0,i]=bet[i]
    #Flag para rodar o while
        flag=np.zeros(nG)
        #Critério de Parada:
        diff=10  
        #Váriaveis de difusão
        var=np.zeros([N_max,nG])
        var[0]=r[0]
        var1=np.zeros([N_max,nG])
    i=0
    flag_sem_limites = 1
    flag_limite_superior = 0    
    #Consenso em si:
    while(i!=N_max-1 and diff>diff_min):     
        d=np.zeros(nG)
        #var[i+1]=r[i]-epil2*(sum(Pg[i])-sum(P_ren))
        #var1[i+1]=var[i+1] + r[i] - var[i]
        #r[i+1] = MMi@var1[i+1]
        for j in range (0,len(MM)):
            if (j!=4):
                r[i+1,j]=MMi[j,:]@var[i,:]
            if (j==4):
                r[i+1,j]=112.5
            var[i+1,j]=r[i+1,j]-epil2[j]*(sum(Pg[i])-sum(P_carga))
            var1[i+1]=var[i+1] + r[i] - var[i]
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
                P_n.append(i)  

# Soma das potências dos agentes despacháveis 
    P_sup=0
    for  a in P_u_max:
        P_sup += Plim_sup[a]

#Analise de Equilíbrio entre oferta e demana
    P_inf=0
    for a in P_u_min:
        P_inf += Plim_sup[a]
    if P_sup > sum(P_carga):
        flag_sem_limites=0
        flag_limite_superior =1
    n_algorit+=1
tempo2=time.time()

print("\nTempo Difusão com restrição:",tempo2-tempo1)    
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
plt.title("Cenário 5-Difusão Exata")
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

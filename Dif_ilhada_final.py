############################ Corte de Carga sem Restrições ################################

''' 
Dados do Problema:
    
   Agente 01: Carga não crítica
   Agente 02: Carga crítica
   Agente 03: Renováveis
   Agente 04: Geração Classica
   
   Todos os Parâmetros dos Agentes Foram retirados do artigo do IEEE  
'''

################################# Importando Bibliotecas ##########################################

import numpy as np
import matplotlib.pyplot as plt
import time
import Matrizes_de_Pesos2 as mp

#################################### Modelagem dos agentes ########################################
#Identificação
Carga=[0,1]
Ren=[2]
Desp=[3]
Bat=[4]

#Agente Bateria
pmaxess = 40
alp_ess = 2
beta = 8

'''
#Carga Baixa:
soc=0.2
'''
'''
#Carga Média:
soc = 0.4
'''

#Carga Alta:   
soc=0.1

bet_ess = alp_ess*pmaxess*(1-soc) + beta

#Agente Geração Renovável

'''
#Carga Baixa
P_ren=np.array([0,0,10,0,0])
'''

'''
#Carga Média
P_ren=np.array([0,0,8.5,0,0]) 
'''

#Carga Alta:
P_ren=np.array([0,0,13.15,0,0])

#Agente Carga
#Limites de Carga:
'''  
#Carga Baixa:    
Carga_lim=[18.6,30]     
'''

'''  
#Carga Média:
Carga_lim=[31.34,32.15]    
'''

#Carga Alta:
Carga_lim=[30,42.9]

#Parâmetros da Função Utilidade :

'''
#Baixa e Média
w=[-36.33,-44.46,0,0,0 ] 
u=[w[0]/Carga_lim[0],w[1]/Carga_lim[1],0,0,0]
'''


#Alta:
w=[-200.25059791, -900.44001645]
u=[w[0]/30,w[1]/44.9,0,0,0]


#Parâmetros de Custo
alp = np.array([0,0,0,0.18,2]) 
bet = np.array([1,1,1,97,bet_ess])

#Número de agentes despacháveis:
nDG=2 

#Número de Agentes:
nG = len(alp)

################## Inicialização dos Parâmetros da Difusão sem restrições #############


#Episolon:
epil = np.array([0.47227199, 0.46591606, 0.41601468, 0.46032257, 0.44916037])
   
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


Plim_inf = np.array([0,0,0,-50,-(soc*pmaxess)])
Plim_sup = np.array([0,0,0,0,(1-soc)*pmaxess])

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
        r[0,i]=w[i]
        
    #Agente Despachável:
    for i in Desp:
        r[0,i]=bet[i]
    
    #Agente Bateria:
    for i in Bat:
        r[0,i]=bet[i]
    #Flag para rodar o while
        flag=np.zeros(nG)
        #Critério de Parada:
        diff=10  
        #Váriaveis de difusão
        var=np.zeros([N_max,nG])
    i=0 

#Vai rodar o consenso apropriadamente dito 
    while(i!=N_max-1 and diff>diff_min):     
        d=np.zeros(nG)
        var[i+1]=r[i]-epil*(sum(Pg[i])-sum(P_ren))
        r[i+1] = MM@var[i+1]
        for j in Ren:
            Pg[i+1,j]=0
        for j in Carga:
            if(abs(Pg[i,j])>=abs(w[j]/u[j])):
                Pg[i+1,j]=w[j]/u[j]
            else:
                Pg[i+1,j] = -(r[i+1,j]-w[j])/(u[j])
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
#Verificação balanço de potência
    P_inf=0
    for a in P_u_min:
        P_inf += Plim_sup[a]
    if P_sup > sum(Carga_lim):
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
print("\n ###################### ")

######################## Exibindo os Dados sem restrições na tela ###########################

print("\n A potência de cada agente quando não há restrições:",Pg[-1])
print("\n O custo incremental quando não há restrições será:",round(max(r[-1]),2))
print("\n ###################### ")
############################# Plotando os gráficos ##############################

for i in range(0,nG):
    plt.plot(r[:,i],label=i+1)
plt.legend()
plt.title("Cenário 3- Difusão")
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




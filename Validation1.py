# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:05:45 2020

@author: jonathan Beaulieu-Emond

Dans le cadre de cours PHY3075 : Méodélisation physique
"""

#Importation des modules
from numba import njit,prange,guvectorize,float64,int64
import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
from matplotlib import animation

#@guvectorize([(float64[:,:], float64,int64, float64[:,:])],"(ij),(k),(l)->(ij)")
def gpu(spin,H,N,E) :
    for i in range(1,N):
        for j in range(1,N) :
            E[i][j]=-spin[i][j]*(spin[i][j+1]+spin[i][j-1]+spin[i-1][j]+spin[i+1][j])

#@njit()
def  Energie(spin,N,k,T,H,J,couleur) :


        #Calcul E et Eprime
        up=spin[1:N+1,2:N+2]

        down=spin[1:N+1,0:N]
        left=spin[0:N,1:N+1]
        right=spin[2:N+2,1:N+1]
        E=-J*spin[1:N+1,1:N+1]*(up+down+left+right)-H*spin[1:N+1,1:N+1]
        #gpu(spin,H,N,E)

        #Ef=E[cx,cy]



        #Ef=E[::2][::2]


        #print(Ef)

        p=np.exp((2*E)/T)



        #p1=np.where(p>1,1,p) # condition minimum de l'equation 3.25

        #[p1]=1

        p2=np.reshape(np.random.rand(N**2),(N,N))
        p3=np.where(np.logical_and(p2<p,couleur==1),-1,1)
        spin[1:N+1,1:N+1]=spin[1:N+1,1:N+1]*p3



        return spin,E
#@njit()
def periodic(f,N):

    #périodicite horizontale et verticale
    f[1:N+1,0]=f[1:N+1,N]
    f[1:N+1,N+1]=f[1:N+1,1]
    f[0,1:N+1]= f[N,1:N+1]
    f[N+1,1:N+1]=f[1,1:N+1]

    #coins

    f[0,0]=f[N-1,N-1]
    f[N+1,N+1]=f[2,2]
    f[0,N+1]=f[N-1,2]
    f[N+1,0]=f[2,N-1]

    return f
#---------------------------------------------------
#@njit()
def main(H0,T,P,k) :
    #Définition des paramètres globaux

    N=64
    nIter=100

    J=1
    #Déclaration des tableaux
    spin=-1+2*np.random.randint(low=0,high=2,size=(N+2,N+2)) #spin aléatoire initial [-1,1]



    #tableau d'output
    E_moyen=np.zeros(nIter)
    M=np.zeros(nIter)
    spinNoir = np.ones([N+2,N+2])
    spinNoir[::2,::2] = 0
    spinNoir[1::2,1::2] = 0
    #spinNoir=np.where(spinNoir==0)
    #spinBlanc=np.where(spinNoir==0)
    spinBlanc = np.ones([N+2,N+2])
    spinBlanc[1::2,::2] = 0
    spinBlanc[::2,1::2] = 0
    spinBlanc=spinBlanc[1:N+1,1:N+1]
    spinNoir=spinNoir[1:N+1,1:N+1]

    #Itération temporelle
    for i in range(0,nIter) :
        #H=H0*np.sin(2*pi*nIter/P)
        H=H0

        #Vrm pas sure pour Eprime
        #operation sur les blancs

        spin,E=Energie(spin,N,k,T,H,J,spinBlanc)

        #operation sur les noirs

        spin,E=Energie(spin,N,k,T,H,J,spinNoir)



        E_mean=np.mean(E)
        #print(E_mean)
        #print(E)
        #print(spin[1:N+1,1:N+1])
        m=np.mean(spin[1:N+1,1:N+1])
        E_moyen[i]=E_mean
        M[i]=m
        #Animation.append([plt.imshow(spin, animated=True)])
        spin=periodic(spin,N)

    return E_moyen,M,spin

H0=0 # à modifier#!!!!!!!!!!!!
T=1.5#!!!!!!!!!!!!!!!
P=1#!!!!!!!!!!!!
k=1

j=0

plt.figure(dpi=1000)
for T in [1.5,2.35,5] :
    E_mean=[]
    for i in range(0,300) :
        M,E,spin=main(H0,T,P,k)
        E_mean.append(E)

    E2=np.zeros(len(E))
    E_err=np.zeros(len(E))
    E_mean=np.array(E_mean)
    for i in range(0,len(E)) :
        E2[i]=np.mean(E_mean[:,i])
        E_err[i]=np.std(E_mean[:,i])
    x=np.arange(1,len(E)+1,1)
    plt.errorbar(x,E2,yerr=E_err,label='T='+str(T),fmt='.')


plt.title('Magnétisation moyenne en fonction de la température')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Magnétisation moyenne ')
plt.legend()
"""
    plt.plot(E,label='T='+str(T))
    plt.title('Énergie moyenne en fonction du temps')
    plt.xlabel('iteration')
    plt.ylabel('Énergie/spin')
    plt.legend()
plt.show()
"""
"""
    plt.plot(M)
    plt.title('Magnétisation moyenne en fonction du temps')
    plt.xlabel('iteration')
    plt.ylabel('Magnétisation/spin')
    plt.show()
"""


"""
fig = plt.figure()
ani= animation.ArtistAnimation(fig,Animation,interval=50, blit=True,
                                repeat_delay=1000)
ani.save('dynamic_images.gif')
plt.show()
"""


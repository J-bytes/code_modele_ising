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


#@njit
def diffusif_leapfrog(c,N,D,dt,dx):

    return D*dt/dx**2*((c[1:N+1,2:N+2] + c[2:N+2,1:N+1] - 4*c[1:N+1,1:N+1] + c[1:N+1,0:N] + c[0:N,1:N+1]))


#@njit(cache=True)
def  Energie(spin,N,T,H,J,couleur) :


        #Calcul E et Eprime
        up=spin[1:N+1,2:N+2]

        down=spin[1:N+1,0:N]
        left=spin[0:N,1:N+1]
        right=spin[2:N+2,1:N+1]
        E=-J*spin[1:N+1,1:N+1]*(up+down+left+right)-H*spin[1:N+1,1:N+1]
        #gpu(spin,H,N,E)

        p=np.exp((2*E)/T[1:N+1,1:N+1])


        p2=np.reshape(np.random.rand(N**2),(N,N))
        p3=np.where(np.logical_and(p2<p,couleur==1),-1,1)
        spin[1:N+1,1:N+1]=spin[1:N+1,1:N+1]*p3



        return spin,E


#@njit(cache=True)
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
#@njit
def main(H0,T0,P,R,t0,dT,sigma,D) :
    #Définition des paramètres globaux

    N=64
    nIter=200

    J=1
    #Déclaration des tableaux
    #spin=-1+2*np.random.randint(low=0,high=2,size=(N+2,N+2)) #spin aléatoire initial [-1,1]
    spin=np.zeros((N+2,N+2))+1 #spin initial

    #Calcul préalable de la région chauffée
    x0=np.int(N/2) # on centre notre laser
    y0=x0
    f=np.zeros((N,N))
    T2=[]
    for i in range(0,N) :
        for j in range(0,N) :
                r=(j-y0)**2+(i-x0)**2
                if r<=R**2 :
                    f[i][j]=1

    #tableau d'output
    E_moyen=np.zeros(nIter)
    M=np.zeros(nIter)

    #Tableau d'indice
    spinNoir = np.ones((N+2,N+2))
    spinNoir[::2,::2] = 0
    spinNoir[1::2,1::2] = 0
    #spinNoir=np.where(spinNoir==0)
    #spinBlanc=np.where(spinNoir==0)
    spinBlanc = np.ones((N+2,N+2))
    spinBlanc[1::2,::2] = 0
    spinBlanc[::2,1::2] = 0
    spinBlanc=spinBlanc[1:N+1,1:N+1]
    spinNoir=spinNoir[1:N+1,1:N+1]

    #Champs magnétique et Température
    T=np.zeros((N+2,N+2))+T0
    H=H0
    #Itération temporelle
    for i in range(0,nIter) :
        #H=H0*np.sin(2*pi*nIter/5)

        Tsource=dT*f*np.exp(-((i-t0)/sigma)**2)

        T[1:N+1,1:N+1]+=diffusif_leapfrog(T,N,D,1,1)+Tsource
        T=periodic(T,N)
        #Vrm pas sure pour Eprime
        #operation sur les blancs

        spin,E=Energie(spin,N,T,H,J,spinBlanc)

        #operation sur les noirs

        spin,E=Energie(spin,N,T,H,J,spinNoir)



        E_mean=np.mean(E)
        #print(E_mean)
        #print(E)
        #print(spin[1:N+1,1:N+1])
        m=np.mean(spin[1:N+1,1:N+1])
        E_moyen[i]=E_mean
        M[i]=m
        #Animation.append([plt.imshow(spin, animated=True)])
        spin=periodic(spin,N)

        T2.append(T[32,32])
        #if i%10==0 :
           # plt.imshow(spin)
           # plt.show()
    plt.plot(T2)
    plt.show()
    return E_moyen,M,spin
###Paramètre du modèle
#@njit()
H0=-0.2
T0=0.1
P=1
R=10
t0=100
dT=0.1
sigma=10
D=0.01
#Température maximale=environ  niter*dT/10
E_moyen,M,spin=main(H0,T0,P,R,t0,dT,sigma,D)
plt.imshow(spin)
plt.show()

"""
@njit(parallel=True)
def optimisation33() :
    P=1

    #Paramètre à modifier
    H0=-0.2

    R=10
    dT=2.4
    t0=100
    T0=0.1
    sigma=10
    stable=np.zeros((400000,7))
    Tarray=np.arange(0,5,0.1)
    dTarray=np.arange(0,3,0.2)
    H0array=np.arange(0,1,0.1)
    i=0
    for D in prange (1,10,1) :

        for T0 in Tarray:
            for dT in dTarray :
                for R in prange(3,20):
                    for sigma in prange(1,20) :
                        for H0 in H0array :
                            E,M,spin=main(H0,T0,P,R,t0,dT,sigma,D/1000)

                            #Vérifions s'il existe un bit stable
                            spin_moyen_attendu=-2*R**2/64**2+1
                            if np.abs(np.mean(spin)-spin_moyen_attendu)<5 :
                                stable[i:]=np.array([True,D,T0,dT,R,sigma,H0])
                                i+=1
                            #plt.imshow(spin)
                            #plt.show()
    return stable


stable=optimisation33()
np.savetxt('stable.txt',stable)
"""
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


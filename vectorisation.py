# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:05:45 2020

@author: jonathan Beaulieu-Emond

Dans le cadre de cours PHY3075 : Méodélisation physique
"""

#Importation des modules
from numba import njit,prange,guvectorize,float64,int64,float32,int32,jit
import numba
import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
from matplotlib import animation

@guvectorize(['void(float64[:,:], int32,float64)'], '(n,n), (),()',target='parallel')
def diffusif_leapfrog(T,N,D):

     T[1:N+1,1:N+1]=D*((T[1:N+1,2:N+2] + T[2:N+2,1:N+1] - 4*T[1:N+1,1:N+1] + T[1:N+1,0:N] + T[0:N,1:N+1]))


@guvectorize(['void(int8[:,:],float64[:,:], int32,float64[:,:],float64,float64[:,:])'], '(m,m),(j,j),(),(l,l),(),(n,n)',target='parallel')
def  Energie(spin,E,N,T,H,couleur) :

        J=1
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




@guvectorize(['void(float64[:,:], int32)'], '(n,n),()',target='parallel')
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


#---------------------------------------------------
#@guvectorize(['void(float64[:],float64[:],float64,float64,int32,int32,int32,float64,int32,float64,int32)'], '(n),(m),(), (),(),(),(),(),(),(),()',target='parallel')
def main(E_moyen,M,H0,T0,P,R,t0,dT,sigma,D,nIter) :
    #Définition des paramètres globaux

    N=64



    #Déclaration des tableaux
    #spin=-1+2*np.random.randint(low=0,high=2,size=(N+2,N+2)) #spin aléatoire initial [-1,1]
    spin=np.zeros((N+2,N+2),dtype=np.int8)+1 #spin initial

    #Calcul préalable de la région chauffée
    x0=np.int(N/2) # on centre notre laser
    y0=x0
    f=np.zeros((N,N),dtype=np.int8)
    for i in range(0,N) :
        for j in range(0,N) :
                r=(j-y0)**2+(i-x0)**2
                if r<=R**2 :
                    f[i][j]=1



    #Tableau d'indice
    spinNoir = np.ones((N+2,N+2),dtype=np.int8)
    spinNoir[::2,::2] = 0
    spinNoir[1::2,1::2] = 0
    #spinNoir=np.where(spinNoir==0)
    #spinBlanc=np.where(spinNoir==0)
    spinBlanc = np.ones((N+2,N+2),dtype=np.int8)
    spinBlanc[1::2,::2] = 0
    spinBlanc[::2,1::2] = 0
    spinBlanc=spinBlanc[1:N+1,1:N+1]
    spinNoir=spinNoir[1:N+1,1:N+1]

    #Champs magnétique et Température
    T=np.zeros((N+2,N+2),dtype=np.float64)+T0
    H=H0
    #Itération temporelle
    for i in range(0,nIter) :
        #H=H0*np.sin(2*pi*nIter/5)
        E=np.zeros((N,N))
        Tsource=dT*f*np.exp(-((i-t0)/sigma)**2)


        diffusif_leapfrog(T,N,D)
        T[1:N+1,1:N+1]+=Tsource
        #Vrm pas sure pour Eprime
        #operation sur les blancs

        Energie(spin,E,N,T,H,spinBlanc)

        #operation sur les noirs

        Energie(spin,E,N,T,H,spinNoir)



        E_mean=np.mean(E)
        #print(E_mean)
        #print(E)
        #print(spin[1:N+1,1:N+1])
        m=np.mean(spin[1:N+1,1:N+1])
        E_moyen[i]=E_mean
        M[i]=m
        #Animation.append([plt.imshow(spin, animated=True)])
        periodic(spin,N)
        periodic(T,N)
        #if i%10==0 :
           # plt.imshow(spin)
           # plt.show()





#@njit(parallel=True)
def optimisation33() :
    P=1

    #Paramètre à modifier
    H0=-0.2
    nIter=200
    R=10
    dT=2.4
    t0=100
    T0=0.1
    sigma=10
    stable=np.zeros((400000,7))
    Tarray=np.arange(0,5,0.1)
    dTarray=np.arange(0,3,0.3)
    H0array=np.arange(0,1,0.1)
    i=0
    for D in prange (1,10,1) :

        for T0 in Tarray:
            for dT in dTarray :
                for R in prange(3,20):
                    for sigma in prange(1,20) :
                        for H0 in H0array :
                              #tableau d'output
                            E=np.zeros(nIter,dtype='float64')
                            M=np.zeros(nIter,dtype='float64')
                            main(E,M,H0,T0,P,R,t0,dT,sigma,D/1000,nIter)

                            #Vérifions s'il existe un bit stable
                            spin_moyen_attendu=-2*R**2/64**2+1
                            if np.abs(M[len(M)-1]-spin_moyen_attendu)<5 :
                                stable[i:]=np.array([True,D,T0,dT,R,sigma,H0])
                                i+=1
                                print(i)
                            #plt.imshow(spin)
                            #plt.show()
    return stable


stable=optimisation33()
np.savetxt('stable.txt',stable)
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


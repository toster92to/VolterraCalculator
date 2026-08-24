import numpy as np
import matplotlib.pyplot as plt
import Volterra_Operations as vo
from Volterra_Kernels import calc_cxx

from Volterra_Kernels import identifyKernelsMultivar
import Decorrelation_OCL

import pandas as pd

from Volterra_Files import writeKernels



length1=10
length2=10
length3=10

	

#Set up systems

#Zero (empty) system
z0=0
z1=np.zeros(length1)
z2=np.zeros((length2,length2))
z3=np.zeros((length3,length3,length3))

#Linear System
h0=0
h1=np.zeros(length1)
h2=np.zeros((length2,length2))
h3=np.zeros((length3,length3,length3))

h1[0]=1
h1[1]=0.05
h1[2]=-0.07
h1[3]=0.1
h1[4]=0.05
h1[5]=0.5

#Nonlinear System
g0=0
g1=np.zeros(length1)
g2=np.zeros((length2,length2))
g3=np.zeros((length3,length3,length3))

g1[1]=1
g2[1,1]=1
g2[2,2]=-0.5
g2[3,3]=0.3
g2[1,2]=-0.5
g2[2,1]=-0.5
g2[1,3]=0.3
g2[3,1]=0.3

g3[0,0,0]=1
g3[0,0,1]=0.5
g3[0,1,0]=0.5
g3[1,0,0]=0.5

#Create two systems
u0,u1,u2,u3=vo.chainVolterraSystems(g0,g1,g2,g3,h0,h1,h2,h3, skew=0)#Nonlinear->Linear System
v0,v1,v2,v3=vo.chainVolterraSystems(h0,h1,h2,h3,g0,g1,g2,g3, skew=0)#Linear-> Nonlinear System


def totuple(a):#Convert array to tuple
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def getDiagonal(array, offsets, length):#Calculate offset diagonals of n-dim. kernels
	diagonal=np.zeros(2*length)
	for i in np.arange(length):
		diagonal[i]=array[totuple(i+offsets)]
	return diagonal[:length]	


def findKernelCorrelation(h0,h1,h2,h3):
	
	#Extend kernels
	length2=len(h2)
	h2ext=np.zeros((2*length2,2*length2))
	h2ext[0:length2,0:length2]=h2
	
	length3=len(h3)
	h3ext=np.zeros((2*length3,2*length3,2*length3))
	h3ext[0:length3,0:length3,0:length3]=h3
	
	#Calculate correlations accross kernels
	correlation2=np.zeros(length2)
	for i in np.arange(length2):
		correlation2 += calc_cxx(0, length2, length2, getDiagonal(h2ext, (0,i), length2))[:length2]
	correlation2=correlation2/correlation2[0]	
	
	correlation3=np.zeros(length3)
	for i in np.arange(length3):
		for j in np.arange(length3):
			#correlation3 += calc_cxx(0, length3, length3, getDiagonal(h3ext, (0,i,j), length3))[:length3]#	diagonal
			correlation3 += calc_cxx(0, length3, length3, h3ext[:,i,j])[:length3]#	on-axis
	correlation3=correlation3/correlation3[0]	
	
	#Sum all correlations
	overallCorrelation=correlation3#+correlation2
	return(overallCorrelation)

#Compute residual correlations
corrU=(findKernelCorrelation(u0,u1,u2,u3))
corrV=(findKernelCorrelation(v0,v1,v2,v3))



if(1):#De-Echoing of kernels
	for j in np.arange(10):
		corrU=findKernelCorrelation(u0,u1,u2,u3)
		corrV=findKernelCorrelation(v0,v1,v2,v3)

		corrUFFT=np.fft.fft(corrU)
		corrUInv=np.real(np.fft.ifft(1/corrUFFT))
		#u0,u1,u2,u3=vo.chainVolterraSystems(u0,u1,u2,u3,z0,corrUInv,z2,z3, skew=0) 	#diagonal
		u0,u1,u2,u3=vo.chainVolterraSystems(z0,corrUInv,z2,z3,u0,u1,u2,u3, skew=0)	#on-axis


#Plot

plt.stem(corrU, 'r', label='u', markerfmt = 'Dr')
plt.stem(corrV, 'b', label='v')
plt.stem(h1, 'g', label='h')
plt.legend()
plt.show()




#Plot

start=0
stop=10#system.len3
h2slice=u2[start:stop,start:stop]
#fig8, axh8 = plt.subplots(1, 1, figsize=(8,5), dpi=90, constrained_layout=False)
fig8 = plt.figure(figsize=(8, 5), dpi=90)
axh8 = fig8.add_axes(rect=(.1,.1,.8,.8))
x,y = np.meshgrid(np.arange(start, stop), np.arange(start,stop))
factor2=400/np.max(np.abs(h2slice)**3)
im2=axh8.scatter(x,y, c=h2slice, s=factor2*abs(h2slice)**3, cmap='PuOr_r', vmin=-1.2*np.max(np.abs(h2slice)), vmax=1.2*np.max(np.abs(h2slice)))
markers=np.zeros(shape=(stop-start,stop-start))
for i in range(stop-start): markers[i,i]=1
axh8.scatter(x,y, c='black', s=10*markers, marker='x')
axh8.set_xlabel(r'$\tau_1$')
axh8.yaxis.tick_right()
axh8.yaxis.set_label_position("right")
axh8.set_ylabel(r'$\tau_2$')
axh8.set_title('')
axh8.grid(visible=False)
clb8=fig8.colorbar(im2, ax=axh8, fraction=0.02, pad=0.1, location='left')
clb8.ax.tick_params(labelsize=8) 
clb8.ax.set_title(r'$u_2(\tau_1, \tau_2)$',fontsize=12, rotation=0)


start=0
stop=10#system.len3
h2slice=v2[start:stop,start:stop]
#fig9, axh9 = plt.subplots(1, 1, figsize=(9,5), dpi=90, constrained_layout=False)
fig9 = plt.figure(figsize=(8, 5), dpi=90)
axh9 = fig9.add_axes(rect=(.1,.1,.8,.8))
x,y = np.meshgrid(np.arange(start, stop), np.arange(start,stop))
factor2=400/np.max(np.abs(h2slice)**3)
im3=axh9.scatter(x,y, c=h2slice, s=factor2*abs(h2slice)**3, cmap='PuOr_r', vmin=-1.2*np.max(np.abs(h2slice)), vmax=1.2*np.max(np.abs(h2slice)))
markers=np.zeros(shape=(stop-start,stop-start))
for i in range(stop-start): markers[i,i]=1
axh9.scatter(x,y, c='black', s=10*markers, marker='x')
axh9.set_xlabel(r'$\tau_1$')
axh9.yaxis.tick_right()
axh9.yaxis.set_label_position("right")
axh9.set_ylabel(r'$\tau_2$')
axh9.set_title('')
axh9.grid(visible=False)
clb9=fig9.colorbar(im3, ax=axh9, fraction=0.02, pad=0.1, location='left')
clb9.ax.tick_params(labelsize=8) 
clb9.ax.set_title(r'$v_2(\tau_1, \tau_2)$',fontsize=12, rotation=0)



plt.show()







length1=150
length2=50
length3=50

#--------------------------------------
wdin='Volterra_Input_Data'
filename='12_Output_mono'
print('File: '+filename+'.csv')
df=pd.read_csv(wdin+'/'+filename+'.csv', sep=',', header=None, dtype=str)#Oscilloscope Data



#normal file
times		= np.transpose(df.values[0:])[0].astype(float)/44100
x  			= np.transpose(df.values[0:])[1].astype(float)/2**16#Start with 0th row

print(len(x))


times=times[::10]
x=x[::10]

print(np.max(x))
print(np.min(x))

xMultivar=np.array([x,x,x,x])
timesMultivar=np.array([times,times,times,times])
xModMultivar=xMultivar

cxxInputNoise=np.zeros(2*length1)
nr_of_workers=1
xModMultivar[0]=Decorrelation_OCL.deCorrelateFast(timesMultivar[0], xMultivar[0], xMultivar[0], 0.03, 0.001, 0.001, length1=length1+1, length2=length2+1, length3=length3+1, write=True, name='test', iterations=50, chunks=1024)
h0,h1,h2,h3 = identifyKernelsMultivar(xModMultivar, xMultivar, length1, length2, length3, nr_of_workers, cxxInputNoise, iterations = 5, alpha1=0.8, AntiAliasing=False, write=True, name='test12')	

corrH=findKernelCorrelation(h0,h1,h2,h3)


#Zero (empty) system
z0=0
z1=np.zeros(length1)
z2=np.zeros((length2,length2))
z3=np.zeros((length3,length3,length3))

corrHInv=np.zeros(length1)

if(1):#De-Echoing of kernels
	for j in np.arange(3):
		print('Off-diagonal deconvolution, iteration '+str(j))
		corrH=findKernelCorrelation(h0,h1,h2,h3)

		corrHFFT=np.fft.fft(corrH)
		corrHInv[:length3]=np.real(np.fft.ifft(1/corrHFFT))
		
		#h0,h1,h2,h3=vo.chainVolterraSystems(h0,h1,h2,h3,z0,corrHInv,z2,z3, skew=0)	#diagonal
		
		h0,h1,h2,h3=vo.chainVolterraSystems(z0,corrHInv,z2,z3,h0,h1,h2,h3, skew=0)	#on-axis

writeKernels(h0,h1,h2,h3, name='test12_d')
#-------------------------------
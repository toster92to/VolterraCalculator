import numpy as np 
import matplotlib.pyplot as plt
import time

from Volterra_In_Out_6 import VolterraResponse


#Create systems
xSystems=2
ySystems=2
system=np.ndarray(shape=(xSystems,ySystems), dtype=object)

length1=20
length2=20
length3=20
length4=20
length5=1

for i in np.arange(xSystems):
	for j in np.arange(ySystems):
		system[i,j]=VolterraResponse(use_opencl = True, kernelName='OSC', kernelPath='', profiling_data=False)
		
		h0=0
		h1=np.zeros(length1)
		h2=np.zeros(shape=(length2,length2))
		h3=np.zeros(shape=(length3,length3,length3))
		h4=np.zeros(shape=(length4,length4,length4,length4))
		h5=np.zeros(shape=(length5,length5,length5,length5,length5))
		
		# h1[5]=-1.3+np.random.normal(loc=0.0, scale=0.05)
		# h1[6]=-0.6
		# h3[5,5,5]=1
		# h3[6,6,6]=0.5

		h1[14]=0.24
		h1[15]=0.48
		h1[16]=0.26

		
		h2[13,13]=-0.1
		h2[14,14]=-0.051
		h2[15,15]=0.62
		h2[16,16]=-0.25
		h2[17,17]=-0.1
		

		h4[15,15,15,15]=-0.2
		h4[14,14,14,14]=0.05
		h4[16,16,16,16]=0.05
		h4[13,13,13,13]=0.05
		h4[17,17,17,17]=0.05
		
		h3[15,15,15]=-0.1


		
		system[i,j].len1=length1
		system[i,j].len2=length2
		system[i,j].len3=length3
		system[i,j].len4=length4
		system[i,j].len5=length5
		
		system[i,j].set_kernels(h0, h1, h2, h3, h4, h5)



		
#Create system linking weights (input weight)
weight=np.zeros(shape=(xSystems,ySystems,5))#last index is number of direction. North, east, south, west, self = 0,1,2,3,4

#Self-coupling (Oscillator feedback)
	#  x y d
weight[0,0,4]=0#0.6
weight[0,1,4]=0#0.6
weight[1,0,4]=0#0.6
weight[1,1,4]=0#0.6
	
#Cross-Coupling (Exchange interaction)	
weight[0,1,2]=1#0.03
weight[1,0,0]=1#0.03
weight[1,1,3]=1#0.03
weight[0,0,1]=1#-0.03

#create system outputs
ltest=6000
out=np.zeros(shape=(xSystems,ySystems,ltest))
inp=np.zeros(shape=(xSystems,ySystems,ltest))

inp[0,0,60]=0.05
inp[0,0,61]=0.1
inp[0,0,62]=0.15
inp[0,0,63]=0.2
inp[0,0,64]=0.3
inp[0,0,65]=0.3
inp[0,0,66]=0.2
inp[0,0,67]=0.15
inp[0,0,68]=0.1
inp[0,0,69]=0.05

inp[0,0,55]=0.55

inp=inp*3

for iteration in np.arange(system[0,0].len1,ltest-1):#Go through all time steps
	for i in np.arange(xSystems):#go through all volterra elements in x direction
		for j in np.arange(ySystems):#go through all volterra elements in y direction
			
			for k in np.arange(5):#go one step into all directions and take input from there
				try: 
					if k==0: inp[i,j,iteration]+=out[i,j+1,iteration-1]*weight[i,j,k]
				except: pass
				try: 	
					if k==1: inp[i,j,iteration]+=out[i+1,j,iteration-1]*weight[i,j,k]
				except: pass
				try: 	
					if k==2: inp[i,j,iteration]+=out[i,j-1,iteration-1]*weight[i,j,k]
				except: pass
				try: 	
					if k==3: inp[i,j,iteration]+=out[i-1,j,iteration-1]*weight[i,j,k]
				except: pass
				try: 	
					if k==4: inp[i,j,iteration]+=out[i,j,iteration-1]*weight[i,j,k]
				except: pass
				
			out[i,j,iteration]=system[i,j].H(x=inp[i,j, iteration-system[i,j].len1+1:iteration+1], oversampleFactor=1)[-1]



# plt.plot(inp[0,0,:])
# plt.plot(inp[0,1,:])
# plt.plot(inp[1,0,:])
# plt.plot(inp[1,1,:])

plt.plot(out[0,0,:])
plt.plot(out[0,1,:])
plt.plot(out[1,0,:])
plt.plot(out[1,1,:])

a=np.zeros(3000)
a[79::64]=1
plt.plot(a)

plt.ylim((-2,2))
plt.show()


fstep=1/91e-12/ltest
plt.plot(np.arange(ltest)*fstep, 20*np.log10(np.abs(np.fft.fft(out[0,0,:]))))

plt.show()



#Old--------------------------------------------------------------------------
system=system[0,0]
for test in np.arange(5):

	system.h1[:10]=1.55*np.array([-0.1,0.2,0.3,-0.4,-0.9,-0.4,0.3,0.2,-0.1,0])

	system.h2[4,4]=test
	
	system.h3[1,1,1]=-0.5
	system.h3[4,4,4]=1
	system.h3[7,7,7]=-0.5
	#system.h3=system.h3*(0.5+test/5)

	
	
	
	out=np.zeros(ltest)
	inp=np.zeros(ltest)
	inp[60]=0.1
	
	fexcitation=1130e6
	excitationI=0.03*np.sin(2*np.pi*fexcitation*91e-12*np.arange(ltest))
	excitationQ=0.03*np.sin(np.pi/2+2*np.pi*fexcitation*91e-12*np.arange(ltest))
	
	inp[1500:]=excitationI[1500:]
	
	sec1=time.time()
	for iteration in np.arange(system.len1,ltest-1):
		#out[iteration]=system.H(x=inp+0.5*out, oversampleFactor=1)[iteration]
		out[iteration]=system.H(x=inp[iteration-system.len1+1:iteration+1]+0.5*out[iteration-system.len1:iteration], oversampleFactor=1)[-1]
	sec2=time.time()
	print(sec2-sec1)
	

	
	plt.plot(inp)
	plt.plot(out)
	
	demodI=out*excitationI
	demodQ=out*excitationQ
	
	demodLPI=np.convolve(demodI,0.05*np.ones(20), mode='same')
	demodLPQ=np.convolve(demodQ,0.05*np.ones(20), mode='same')
	
	#plt.plot(demodLPI)
	#plt.plot(demodLPQ)
	
	phi=np.arctan2(demodLPQ,demodLPI)
	#plt.plot(phi)
	
plt.show()

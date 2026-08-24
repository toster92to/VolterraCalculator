import numpy as np
import matplotlib.pyplot as plt

from Volterra_In_Out_6 import VolterraResponse
from Volterra_File_Operations import readKernels


system=readKernels('R')

systemComp=readKernels('H')



timestepSim=0.05
timestepCtrl=0.25
simLength=10000



u=np.zeros(simLength)
e=np.zeros(simLength)
s=np.zeros(simLength)
slin=np.zeros(simLength)
l=np.zeros(simLength)

sysout=np.zeros(simLength)
sysoutlin=np.zeros(simLength)
sysout2=np.zeros(simLength)
times=np.arange(simLength)*timestepSim

u[2000:3000]=np.ones(1000)*2.3
u[6000:7000]=np.ones(1000)*-2.3


k1,_=systemComp.H1_opencl(u[::5])
k2,_=systemComp.H2_opencl(u[::5])
k3,_=systemComp.H3_opencl(u[::5])
k4,_=systemComp.H4_opencl(u[::5])
k5,_=systemComp.H5_opencl(u[::5])
sysoutComp=k1+k2+k3+k4+k5
sysoutComp=np.repeat(sysoutComp, 5)





class RLCsys():
	def __init__(self):
		self.x=0
		self.xp=0
		self.xpp =0
		self.old_xs=0

		self.RL=0.15#0.15
		self.RC=5#5.0
		self.C=1.0
		self.L=1.0
		self.dt=timestepSim
		self.L_nonlinear=1

	def simulateStep(self, s):
		self.L_nonlinear = self.L-self.L*(self.xp*self.C+self.x/self.RC)*1.5;
		if (self.L_nonlinear<=0):
			self.L_nonlinear=0.1
			print("L ist zu klein at "+str(i*timestepSim)+"s");

		self.xpp= -(self.RL/self.RC+1.0)/(self.L_nonlinear*self.C)*self.x - self.xp*(self.RL*self.C+self.L_nonlinear/self.RC)/(self.L_nonlinear*self.C) + s/(self.L_nonlinear*self.C);
		self.xp+=self.xpp*self.dt;
		self.x+=self.xp*self.dt;






RLC=RLCsys()#volterra controller
RLClin=RLCsys()#linear controller
RLC2=RLCsys()#no controller

for i in np.arange(2000,simLength):
	
	e[i]=u[i] - sysout[i-5]
	
	if i%5==0:
		h1,_=system.H1_opencl(e[i-1500:i:5])
		h2,_=system.H2_opencl(e[i-1500:i:5])
		#h3,_=system.H3_opencl(e[i-1500:i:5])
		#h4,_=system.H4_opencl(e[i-1500:i:5])
		#h5,_=system.H5_opencl(e[i-1500:i:5])
		s[i:i+5] = np.ones(5)*(h1[-1]+h2[-1])#+h3[-1]+h4[-1]+h5[-1])
		slin[i:i+5] = np.ones(5)*(h1[-1])
	
	RLC.simulateStep(s[i])
	RLClin.simulateStep(slin[i])
	RLC2.simulateStep(u[i])		
	sysout[i]=RLC.x	
	l[i]=RLC.L_nonlinear
	sysoutlin[i]=RLClin.x	
	sysout2[i]=RLC2.x
		
		


dt=0.05
tau=2
fir=np.zeros(600)
factor=np.sum(np.exp(-np.arange(300)*dt/tau))
for i in np.arange(300,600):
    fir[i]=np.exp(-(i-300)*dt/tau)/factor
G=0.95*np.convolve(fir, fir, mode='same')


ideal=np.convolve(u,G, mode='same')

print(np.var(ideal-sysout))
print(np.var(ideal-sysoutlin))
	
plt.plot(np.convolve(u,G, mode='same'), label='G', color='lightsteelblue')	
plt.plot(s, label='s', color='green')
plt.plot(slin, label='slin', color='lightgreen', linestyle='--')
#plt.plot(l, label='L', color='black')
plt.plot(sysout, label='x', color='darkorange')
#plt.plot(sysoutComp, label='x_comparison', color='salmon')
plt.plot(sysoutlin, label='xlin', linestyle='--', color='orange')
#plt.plot(sysout2, label='x2', color='navajowhite')
plt.plot(e, label='e', color='red')
plt.plot(u, label='u', color='blue')

plt.legend()
plt.show()	
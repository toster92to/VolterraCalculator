import numpy as np
import math
from scipy import signal as sg
import matplotlib.pyplot as plt

def systemFunction(x, bw, sr, dt, oversampleFactor, asymmetry=0, delay=87e-12):
	#delay=40e-12
	#delay=0
	#stepResponseInput=([0, 0, 0, 0, 1.02, 0, -0.22, 0.3, -0.1])
	#x=np.convolve(x, stepResponseInput, mode='same')
	
	xOversampled=np.zeros(len(x)*oversampleFactor)
	yOversampled=np.zeros(len(x)*oversampleFactor)
	
	#initial calculations
	dtOversampled=dt/oversampleFactor
	
	tau=1/(2*np.pi*bw)
	delaySamples=int(delay/(dtOversampled))
	
	#Calculate fft of input signal
	xfft=np.fft.fft(x)
	xfftOversampled=np.zeros(len(x)*oversampleFactor, dtype=complex)

	
	#Calculate oversampled input signal
	xfftOversampled[0:int(len(x)/2)]=xfft[0:int(len(x)/2)]
	xfftOversampled[-int(len(x)/2):]=xfft[-int(len(x)/2):]
	xOversampled=np.real(np.fft.ifft(xfftOversampled))*oversampleFactor


	#2nd order low-pass, dual slew-limited system
	if(0):
		zOversampled=np.zeros(len(xOversampled))
		for n in np.arange(len(xOversampled)):
			yOversampled[n]=yOversampled[n-1]+sr*dtOversampled*math.tanh(dtOversampled/tau*1/(sr*dtOversampled)*(xOversampled[n]-yOversampled[n-1]))#Simple model of a slew-rate limited device, max. SR = 0.2/timestep; if not slew-limited, it behaves as a 1st order lp with tau=2.5 steps
			zOversampled[n]=zOversampled[n-1]+sr*dtOversampled*math.tanh(dtOversampled/tau*1/(sr*dtOversampled)*(yOversampled[n]-zOversampled[n-1]))
		yOversampled=zOversampled


	#2nd order low-pass, single slew-limited system
	if(1):
		
		zOversampled=np.zeros(len(xOversampled))
		for n in np.arange(len(xOversampled)):
			yOversampled[n]=yOversampled[n-1]+sr*dtOversampled*(math.tanh(asymmetry + dtOversampled/tau*1/(sr*dtOversampled)*(xOversampled[n]-yOversampled[n-1]))-math.tanh(asymmetry))
			zOversampled[n]=zOversampled[n-1] + 3*dtOversampled/tau*(yOversampled[n]-zOversampled[n-1])
		yOversampled=zOversampled
		
		#Delaying the output
		if delaySamples!=0:
			yOversampled[delaySamples:]=yOversampled[0:-delaySamples]
			yOversampled[0:delaySamples]=np.zeros(delaySamples)
		
		
		
		
	#print(yOversampled)
	#Calculate y by sampling oversampled signal
	y=yOversampled[0::oversampleFactor]
	#print(y)
	return(y)


def boostConverter(x, dt, oversampleFactor):

	#Circuit Parameters
	Uin=12
	I_Load=0.5
	
	L=5e-9
	C=0.5e-9
	
	#Controller Parameters
	P=0.07
	I=0.002e9
	Diff=4e-9
	
	#Initial Values
	D_0=0.5
	Uint_0=0.5
	Uout_0=24
	I_0=1
	
	
	
	
	#Oversample Input----------------------------------------------------
	xOversampled=np.zeros(len(x)*oversampleFactor)
	
	#initial calculations
	dtOversampled=dt/oversampleFactor
		
	#Calculate fft of input signal
	xfft=np.fft.fft(x)
	xfftOversampled=np.zeros(len(x)*oversampleFactor, dtype=complex)

	
	#Calculate oversampled input signal
	xfftOversampled[0:int(len(x)/2)]=xfft[0:int(len(x)/2)]
	xfftOversampled[-int(len(x)/2):]=xfft[-int(len(x)/2):]
	xOversampled=np.real(np.fft.ifft(xfftOversampled))*oversampleFactor
	
	x = xOversampled
	dt=dtOversampled
	#--------------------------------------------------------------------
	
	gaussian=sg.gaussian(200,30,sym=True)#0.7 seems to be a good value for std deviation
	gaussian=gaussian/np.sum(gaussian)
	
	x=np.convolve(x, gaussian, mode='same')
	Usoll=x+24
	
	
	#Prepare Arrays
	Iind=np.ones(len(x))*I_0#Inductor currents
	D=np.ones(len(x))*D_0#Duty cycles
	U_err=np.zeros(len(x))#Measured voltage errors
	U_L=np.zeros(len(x))#Voltages accross inductor
	Uout=np.ones(len(x))*Uout_0#Output voltages
	I_C=np.zeros(len(x))#Output capacitor currents
	U_diff=np.zeros(len(x))#Output voltages of PID differentiator
	U_int=np.ones(len(x))*Uint_0#Output voltages of PID integrator
	
	#Calculate Difference Equations
	for i in np.arange(1, len(x)):
		
		#Error Voltage
		U_err[i]=Usoll[i]-Uout[i-1]
		
		#PI Controller
		U_int[i]=U_int[i-1]+dt*I*U_err[i]
		U_diff[i] = ((U_err[i]-U_err[i-1])/dt*0.05 + U_diff[i-1]*0.95) * Diff
		
		D[i]=U_err[i]*P + U_int[i] + U_diff[i]
		D[i]=np.clip(D[i], 0.1, 0.9)
		
		#Main Inductor Voltage
		U_L[i]=Uin + (D[i]-1)*Uout[i]
		
		#Inductor Current
		Iind[i]=Iind[i-1]+dt*1/L*U_L[i]
		
		#Output Capacitor Current
		I_C[i] = Iind[i]*(1-D[i])-I_Load
		
		#Output Voltage
		Uout[i]=Uout[i-1]+dt*1/C*I_C[i]
		
	return(Uout[0::oversampleFactor], D[0::oversampleFactor], Iind[0::oversampleFactor])
		
	
# x=np.random.normal(loc=0, scale=1, size=1000)*0.01
# plt.plot(x)	
# plt.plot(systemFunction(x, 3000e6, 1000e6, dt=25e-12, oversampleFactor=5, asymmetry=0.1, delay=150e-12))	
# plt.show()
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from scipy import signal as sg
import csv
import math
import time
from Simulate_System import systemFunction
from Simulate_System import boostConverter
from Volterra_Downsample import downsample
from Volterra_Downsample import oversample
from Volterra_Downsample import LPfilter
from Volterra_Downsample import HPfilter
from Volterra_Downsample import fineShift

from Volterra_Files import readKernels
from Volterra_Operations import offDiagonalTruncation

from matplotlib.font_manager import FontProperties
import matplotlib.font_manager

from itertools import combinations_with_replacement
from itertools import permutations

import Volterra_PlotSystem

#import Volterra_In_Out_GUI


darkmode=False
if(darkmode):
	plt.rcParams.update({
		#"lines.color": "white",
		#"patch.edgecolor": "white",
		#"text.color": "black",
		"axes.facecolor": "lightgrey",
		"axes.edgecolor": "black",
		#"axes.labelcolor": "white",
		#"xtick.color": "white",
		#"ytick.color": "white",
		#"grid.color": "lightgray",
		"figure.facecolor": "darkgrey",
		"figure.edgecolor": "black",
		"savefig.facecolor": "grey",
		"savefig.edgecolor": "black"})
#Some inspiration from:
# Volterra Kernel Interpolation for System Modeling and .
# Predistortion Purposes
# Peter Singerl*t, Heinz Koeppl*
# *Christian Doppler Laboratory for.Nonlinear Signal Processing

#(Oversampling the volterra system with zero-padded kernels. When inverting a system, a higher input sampling rate can be used, 
#to accurately model input signal). For this purpose, functions H(x) now also have the argument "oversampleFactor"

class VolterraResponse:
	def __init__(self, kernelName='DUT', kernelPath='', kernelLengths=np.array([15,10,10])):#A path below the folder '/Volterra_Kernels' can be specified. If left empty, kernels are taken directly from '/Volterra_Kernels'
		self.kernelName = kernelName
		self.kernelPath = kernelPath
	
		if self.kernelName=='empty':
			self.h0=1e-9
			self.h1=1e-9*np.ones(shape=(kernelLengths[0]))
			self.h2=1e-9*np.ones(shape=(kernelLengths[1],kernelLengths[1]))
			self.h3=1e-9*np.ones(shape=(kernelLengths[2],kernelLengths[2],kernelLengths[2]))		
		else:
			self.h0,self.h1,self.h2,self.h3=readKernels(kernelName, kernelPath)
		
		self.len1=len(self.h1)
		self.len2=len(self.h2)
		self.len3=len(self.h3)


	#An oversampling factor can be specified
	def H0(self, x, oversampleFactor=1):			
		H0x=np.zeros(len(x))
		for n in np.arange(len(x)): 
			H0x[n]=self.h0
		return(H0x)
		
		
	def H1(self, x, oversampleFactor=1):
		H1x=np.zeros(len(x))
		#first order:
		for tau1 in np.arange(self.len1):
			vecX1=np.zeros(len(x))
			lenFilled = len(x)-tau1*oversampleFactor
			vecX1[tau1*oversampleFactor:]=self.h1[tau1]*x[len(x)-lenFilled-tau1*oversampleFactor: len(x)-tau1*oversampleFactor]

			H1x+=vecX1	
		return(H1x)

			
	def H2(self, x, oversampleFactor=1):
		H2x=np.zeros(len(x))
		for [tau1, tau2] in list(combinations_with_replacement(np.arange(self.len2), 2)):
			perm=list(set(permutations([tau1,tau2])))
		
			vecX2=np.zeros(len(x))
			lenFilled = len(x)-max(tau1*oversampleFactor,tau2*oversampleFactor)
			vecX2[max(tau1*oversampleFactor, tau2*oversampleFactor):] = self.h2[tau1, tau2] * x[len(x)-tau1*oversampleFactor-lenFilled:len(x)-tau1*oversampleFactor]*x[len(x)-tau2*oversampleFactor-lenFilled:len(x)-tau2*oversampleFactor]
			for [t1,t2] in perm:#Do it for all unique combinations of tau1,2,3. Alternatively (and probably faster): just compute number of unique combinations, and then G3x[n]+=number*temp
				H2x+=vecX2
						
		return(H2x)


	def H3(self, x, oversampleFactor=1):
		H3x=np.zeros(len(x))
		for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(self.len3), 3)):
			perm=list(set(permutations([tau1,tau2, tau3])))
		
			vecX3=np.zeros(len(x))
			lenFilled = len(x)-max(tau1*oversampleFactor, tau2*oversampleFactor, tau3*oversampleFactor)
			vecX3[max(tau1*oversampleFactor, tau2*oversampleFactor, tau3*oversampleFactor):] = self.h3[tau1, tau2, tau3] * x[len(x)-tau1*oversampleFactor-lenFilled:len(x)-tau1*oversampleFactor]*x[len(x)-tau2*oversampleFactor-lenFilled:len(x)-tau2*oversampleFactor]*x[len(x)-tau3*oversampleFactor-lenFilled:len(x)-tau3*oversampleFactor]
			for [t1,t2,t3] in perm:#Do it for all unique combinations of tau1,2,3. Alternatively (and probably faster): just compute number of unique combinations, and then G3x[n]+=number*temp
				H3x+=vecX3
						
		return(H3x)


	def H(self, x, oversampleFactor=1):
		Hx=self.H1(x, oversampleFactor)+self.H3(x, oversampleFactor)+self.H2(x, oversampleFactor)#+self.H0(x, oversampleFactor)
		return(Hx)
		
	def H_p(self, x, oversampleFactor=1):
		H_px=self.H3(x, oversampleFactor)+self.H2(x, oversampleFactor)
		return(H_px)
		
	def Volterra_In_Out(self, x, oversampleFactor=1):
		y=self.H(x, oversampleFactor)
		return(y)
		
	def H1_inv(self, x, oversampleFactor=1, padding=0):
		H1_invx=np.zeros(len(x))
		impulseResponse=np.zeros(2*self.len1)
		
		
		if oversampleFactor==1: impulseResponse[self.len1+1:]=self.h1[:-1]#Use this if oversampleFactor=1
		if oversampleFactor!=1: impulseResponse[self.len1:]=self.h1#Use this if oversampleFactor != 1
		
		
		if (padding !=0):
			impulseResponse=np.concatenate((np.zeros(padding), impulseResponse, np.zeros(padding)))

		#Generate prototype impulse
		impulse=np.zeros(len(impulseResponse))
		impulse[0]=1
		#self.h1 is 1st order impulse response
		

		#Calculate FFT of ideal impulse, and of impulse response. Ideal impulse FFT is truncated, because pulse should be band-limited for downsampling.
		impulseFFT=np.fft.fft(impulse)
		
		impulseResponseFFT=np.fft.fft(impulseResponse)
		#Make non-invertible systems invertible by limiting how small its FFT can be (to prevent division by 0)
		for n in range(len(impulseResponseFFT)):
			if (abs(impulseResponseFFT[n])<0.05):
				impulseResponseFFT[n]=0.05*impulseResponseFFT[n]/abs(impulseResponseFFT[n])
		
		#Calculate inverse response
		inverseResponse=np.zeros(len(impulseResponseFFT))
		inverseResponse=impulseFFT/impulseResponseFFT
		
	#TEST------------------------------------------------------------------------------+1, 1::
		#Calculate time-domain kernel (FIR coeffs) from inverse response
		if oversampleFactor!=1: inverseKernel=np.zeros(2*self.len1*oversampleFactor-1+2*padding*oversampleFactor)
		if oversampleFactor==1: inverseKernel=np.zeros(2*self.len1*oversampleFactor+2*padding*oversampleFactor)
		
		inverseKernel[::oversampleFactor]=np.real(np.fft.ifft(inverseResponse))		

		
		inverseKernel=np.roll(inverseKernel, -1)#Shift to the right by one. A truncating shift should be used instead of np.roll, but error is small.

		#remove padding
		#inverseKernel = inverseKernel[padding*oversampleFactor:-padding*oversampleFactor]

		
		H1_invx=np.convolve(x, inverseKernel, mode='same')
		
		return(H1_invx)
	


class Experiment:
	def __init__(self, order, inputBandwidth, inputlength, Amplitude, refAttenuation, skew, ssAttenuation, Gain, Device, kernelName, kernelPath):
		self.order = order
		self.inputBandwidth=inputBandwidth
		self.inputlength=inputlength
		self.Amplitude=Amplitude
		self.refAttenuation=refAttenuation
		self.skew=skew
		self.ssAttenuation=ssAttenuation
		self.Gain=Gain
		self.Device=Device
		self.kernelName=kernelName
		self.kernelPath=kernelPath

	def simulateSR(self):
	#Generate an input array for testing the system. Select one of the test functions here------------------------------
		system=VolterraResponse(self.kernelName, self.kernelPath)

		timestep=1/(2*self.inputBandwidth)
		times=np.arange(self.inputlength)*timestep
		input=np.zeros(self.inputlength)
		
		

		#----------------------------------------------
		refFactor=10**(-self.refAttenuation/20)
		ssFactor=10**(-self.ssAttenuation/20)
		
		#Simulated System:
		if(1):	
			#vals=np.array([0,0,0,0,0,0,0,0,0,0,0,1,-1,2,-2,3,-3,0,0,0,0,0,0,0,0,0,0,0,0])/3*Amplitude#Zoom=3
			vals=np.array([0,0,3,3,3,-3,-3,-3,3,3,3,0,0])/3*self.Amplitude#Zoom=1
			if(1):
				for n in np.arange(len(input)):
					#input[n]=Amplitude*np.sign(math.sin(2*np.pi*4*n/inputlength))#Squarewave
					#input[n]=Amplitude*math.sin(2*np.pi*3*n/inputlength)#Sinewave
					input[n]=vals[int(len(vals)*n/self.inputlength)]#Arbitrary


		
			#Filter input data for better depiction
			if(1):
				#Apply Gaussian Filter to input data
				gaussian=sg.windows.gaussian(21,0.01,sym=True)#0.7 seems to be a good value for std deviation
				gaussian=gaussian/np.sum(gaussian)

				#Calculate filtered output
				input=np.convolve(input, gaussian, mode='same')
			

			#Remove attenuation difference
			input=input/refFactor
			
			output=system.H(input)
			
			srMax=np.max((output[1:]-output[:-1])/timestep)
			srMin=np.min((output[1:]-output[:-1])/timestep)
			return(srMax, srMin)

				


if __name__ == '__main__':

#-----------------------
	# nrExperiments=80
	# Exp=np.ndarray(shape=nrExperiments, dtype=Experiment)
	# srMax=np.zeros(nrExperiments)
	# srMin=np.zeros(nrExperiments)
	# Amplitudes=np.linspace(0, 0.5, nrExperiments)
	
	# for i in np.arange(nrExperiments):
		# Exp[i]=Experiment(order=3, inputBandwidth=4.8e9, inputlength=240, Amplitude=Amplitudes[i], refAttenuation=0, skew=3, ssAttenuation=7, Gain=1.51, Device='LMH3401', kernelName='DUT', kernelPath='')
		# srMax[i],srMin[i] = Exp[i].simulateSR()
		
	# plt.plot(2*Amplitudes*2*4.8e9, srMax, label='SR, max')
	# plt.plot(2*Amplitudes*2*4.8e9, np.abs(srMin), label='SR, min')
	
	# slope=(srMax[1]-srMax[0])/(2*Amplitudes[1]*2*4.8e9-2*Amplitudes[0]*2*4.8e9)
	# plt.plot(2*Amplitudes*2*4.8e9, slope*2*Amplitudes*2*4.8e9, label='slope')
	
	# # for i in np.arange(nrExperiments)[::-1]:
		# # if srMax[i]<slope*0.916*2*Amplitudes[i]*2*4.8e9: srActualPos=2*srMax[i];
		# # if abs(srMin[i])<slope*0.916*2*Amplitudes[i]*2*4.8e9: srActualNeg=2*srMin[i];	
	# # print(srActualPos, srActualNeg)
		
	# plt.legend()
	# plt.show()
#---------------------------		
		
	system=VolterraResponse(kernelName='DUT', kernelPath='')#kernelPath=''
	#system.h3=system.h3-0.0001*np.ones(shape=(system.len3,system.len3,system.len3))
	
	if(0):
		dist=9
		print('Truncating off-diagonal elements, distance = '+str(dist))
		system.h0, system.h1, system.h2, system.h3 = offDiagonalTruncation(system.h0,system.h1,system.h2,system.h3, distance=dist)
			
		
	if(0):
		val=0.08
		print('Truncating small elements < '+str(val))
		#system.h0[np.abs(system.h0) < val] = 0
		#system.h1[np.abs(system.h1) < val] = 0
		system.h2[np.abs(system.h2) < val] = 0
		system.h3[np.abs(system.h3) < val] = 0
		
	if(0):
		print('removing means')
		system.h2 -= np.mean(system.h2)
		system.h3 -= np.mean(system.h3)		
	
	order=3
	inputlength=7200*order#240*order
	
	



	#Generate an input array for testing the system. Select one of the test functions here------------------------------
	inputBandwidth=2.6e9
	timestep=1/(2*inputBandwidth*order)
	
	times=np.arange(inputlength)*timestep
	input=np.zeros(inputlength)#sampled at each timestep, which is volterraTimestep/order. 
	
	#Settings-----------------------
	
	#General
	Amplitude=0.2#0.3

	#RefAttenuations
	#8.25	for ADA4960
	#8.25 for ADL5565
	#0 for LMH5401
	#0 for LMH3401
	refAttenuation=0#dB, attenuation inequality between DUT and reference channel (aRef-aDUT)

	skew=-32#41.2[Samples] Skew output channel this much
	skewMismatch=-30e-12#[s] Skew mismatch between both measured curves. Happens due to inconsistent triggering

	ssAttenuation=4#dB, attenuation difference between both step responses
	
	#Simulated output
	#Gains
	#1.76 for LMH3401
	#1.25 for LMH5401
	#1.58 for ADA4960
	#2.1 for ADL5565
	Gain=1.51#Gain of the amplifier
	
	#Measured output
	Device='LMH3401'
	


	#----------------------------------------------
	refFactor=10**(-refAttenuation/20)
	ssFactor=10**(-ssAttenuation/20)
	
	#Simulated System:
	if(1):

			
		#Step
		if(0):
			input[0]=0
			input[int(inputlength/2):]=np.ones(int(inputlength/2))*Amplitude
			
		#WGN		
		if(0):
			np.random.seed(98798)
			input=np.random.normal(loc=0.0, scale=Amplitude, size=inputlength)

		#Sinewave	
		if(1):
			for n in np.arange(100,len(input)-100):
				input[n]= 0.3*math.sin(2*np.pi*3*n/(128*order)) + 0.02*math.sin(2*np.pi*34.5*n/(128*order)) 

			
		#Other functions	
		#vals=np.array([0,0,0,0,0,0,0,0,0,0,0,1,-1,2,-2,3,-3,0,0,0,0,0,0,0,0,0,0,0,0])/3*Amplitude#Zoom=3
		vals=np.array([0,0,1,2,3,0,0,0,-1,-2,-3,0,0])/3*Amplitude#Zoom=1
		if(0):
			for n in np.arange(len(input)):
				#input[n]=Amplitude*np.sign(math.sin(2*np.pi*4*n/inputlength))#Squarewave
				#input[n]=Amplitude*math.sin(2*np.pi*3*n/inputlength)#Sinewave
				input[n]=vals[int(len(vals)*n/inputlength)]#Arbitrary


		#Random bits
		#Bit length in Volterra timesteps
		bitLength=2
		if(0):
			for n in np.arange(len(input)):
				if n%bitLength==0:
					input[n]=2*np.random.randint(2)*Amplitude - Amplitude
				else:
					input[n]=input[n-1]

				
		input2=np.zeros(inputlength)
		input2=input*ssFactor
	
		#Filter input data for better depiction
		if(1):
			#Apply Gaussian Filter to input data
			gaussian=sg.windows.gaussian(20,2.0,sym=True)#0.7 seems to be a good value for std deviation
			gaussian=gaussian/np.sum(gaussian)

			#Calculate filtered output
			input=np.convolve(input, gaussian, mode='same')
			input2=np.convolve(input2, gaussian, mode='same')
		
		if(1):
			#Apply LP Filter to input data
			input=np.pad(input, (1000, 1000), 'constant', constant_values=(0, 0))
			input2=np.pad(input2, (1000, 1000), 'constant', constant_values=(0, 0))
			
			input,_=LPfilter(input, times, fstop=2.5e9)
			input2,_=LPfilter(input2, times, fstop=2.5e9)
			
			input=input[1000:-1000]
			input2=input2[1000:-1000]
		
		yMeasured=np.ones(inputlength)*1e-12
		yMeasured2=np.ones(inputlength)*1e-12
		
	#Measured System:
	#Read a measured trace for comparison, and use measured input as input to volterra system--------------------------------------------------------
	if(0):
		Filename1=Device+'step-6.csv'
		Filename2=Device+'step-10.csv'
		mdin='Volterra_Input_Data'
		
	#File1	
		#Input to the system
		df=pd.read_csv(mdin+'/'+Filename1, sep=',', header=None, dtype=str)#Oscilloscope Data
		input=np.transpose(df.values[0:])[1].astype(float)#Start with 0th row, take 1st column
		inputTimes=np.transpose(df.values[0:])[0].astype(float)
		#input2=input*ssFactor
		
		#Output of the system
		#df=pd.read_csv(mdin+'/'+Filename1, sep=',', header=None, dtype=str)#Oscilloscope Data
		yMeasured=np.transpose(df.values[0:])[2].astype(float)#Start with 0th row, take 2nd column		
		
	#File2
		#Input to the system
		df=pd.read_csv(mdin+'/'+Filename2, sep=',', header=None, dtype=str)#Oscilloscope Data
		input2=np.transpose(df.values[0:])[1].astype(float)#Start with 0th row, take 1st column
		#input2[1:]=input2[:-1]
		
		
		#Output of the system
		yMeasured2=np.transpose(df.values[0:])[2].astype(float)#Start with 0th row, take 2nd column
		#yMeasured2[1:]=yMeasured2[:-1]

		
		
		if(1):#Downsample
			fstop=inputBandwidth
		
			yMeasured,timesMeasured =downsample(yMeasured, inputTimes,fstop*order)
			yMeasured2,timesMeasured =downsample(yMeasured2, inputTimes,fstop*order)
			
			input,__ =downsample(input, inputTimes,fstop*order)
			input2,inputTimes =downsample(input2, inputTimes,fstop*order)


		#Remove attenuation difference
		input=input/refFactor
		input2=input2/refFactor
		

		#Remove mean
		yMeasured-=np.mean(yMeasured)
		yMeasured2-=np.mean(yMeasured2)
		
		input-=np.mean(input)
		input2-=np.mean(input2)
		
		
		
		_, input=fineShift(inputTimes, input, shiftSamples=0, shiftTime=skewMismatch)
		_, yMeasured=fineShift(inputTimes, yMeasured, shiftSamples=skew, shiftTime=skewMismatch)

		_, input2=fineShift(inputTimes, input2, shiftSamples=0, shiftTime=0e-12)
		_, yMeasured2=fineShift(inputTimes, yMeasured2, shiftSamples=skew, shiftTime=0e-12)

		#input+=0.01
		#input2+=0.0108

		#Limit length of the measured input/output
		yMeasured=yMeasured[int(len(yMeasured)/2-inputlength/2):int(len(yMeasured)/2+inputlength/2)]
		yMeasured2=yMeasured2[int(len(yMeasured2)/2-inputlength/2):int(len(yMeasured2)/2+inputlength/2)]
		input=input[int(len(input)/2-inputlength/2):int(len(input)/2+inputlength/2)]
		input2=input2[int(len(input2)/2-inputlength/2):int(len(input2)/2+inputlength/2)]

		if(0):
			#Remove constant value
			yMeasured-=yMeasured[50]
			yMeasured2-=yMeasured2[50]
			
			input-=input[50]
			input2-=input2[50]
		
		# #Smoothen measured input/output
		if(0):
			gaussian=sg.windows.gaussian(21,0.55,sym=True)#0.55 seems to be a good value for std deviation
			#gaussian=np.array([0.5, 1, 0.5])
			gaussian=gaussian/np.sum(gaussian)
			
			yMeasured=np.convolve(yMeasured, gaussian, mode='same')#Smoothen it for better depiction
			yMeasured2=np.convolve(yMeasured2, gaussian, mode='same')#Smoothen it for better depiction
			
			input=np.convolve(input, gaussian, mode='same')
			input2=np.convolve(input2, gaussian, mode='same')
		
		

	#--------------------------------------------------------------------------------------------


	#Plot some kernel info-------------------------------------------------------
	print('Kernel Dimensions:')
	print('h1: '+str(system.len1))
	print('h2: '+str(system.len2))
	print('h3: '+str(system.len3))
	

	# spectrum=np.concatenate((np.fft.fft(system.h1)[:31], np.fft.fft(system.h1)[-30:], [np.fft.fft(system.h1)[0]]))
	#plt.plot(np.abs(spectrum))
	#plt.plot(np.abs(np.fft.fft(system.h1)))
	#plt.plot(np.angle(spectrum))
	# h1low=np.fft.ifft(spectrum)
	# plt.plot(np.convolve(h1low, ([0.25,0.5,0.25]), mode='same'))
	# plt.show()
	
	

	# print('2-norm of kernels:')
	# print('h1:'+str(np.linalg.norm(h1)))
	# print('h2:'+str(np.linalg.norm(h2)))
	# print('h3:'+str(np.linalg.norm(h3)))

	# print('Expected contribution at full Amplitude:')
	# print('h1: '+str(Amplitude*np.linalg.norm(h1)))
	# print('h2: '+str(Amplitude**2*np.linalg.norm(h2)))
	# print('h3: '+str(Amplitude**3*np.linalg.norm(h3)))




	#Calculating output of reference system-----------------------------------------


	#Simulation model:
	#reference=systemFunction(Gain*input, bw=1.5e9, sr=0.5e9, dt=200e-12, oversampleFactor=40, asymmetry=0.2)
	#reference2=systemFunction(Gain*input2, bw=1.5e9, sr=0.5e9, dt=200e-12, oversampleFactor=40, asymmetry=0.2)
	
	
	#reference=systemFunction(Gain*input, bw=2.3e9, sr=2e9, dt=timestep, oversampleFactor=40, asymmetry=-0.1)
	#reference2=systemFunction(Gain*input2, bw=2.3e9, sr=2e9, dt=timestep, oversampleFactor=40, asymmetry=-0.1)

	#reference=systemFunction(systemFunction(Gain*input, bw=2.3e9, sr=4e9, dt=200e-12, oversampleFactor=40, asymmetry=-0.1), bw=2.3e9, sr=4e9, dt=200e-12, oversampleFactor=40, asymmetry=-0.1)
	#reference2=systemFunction(systemFunction(Gain*input2, bw=2.3e9, sr=4e9, dt=200e-12, oversampleFactor=40, asymmetry=-0.1), bw=2.3e9, sr=4e9, dt=200e-12, oversampleFactor=40, asymmetry=-0.1)

	#Fitted to ADA4960:
	
	
	#reference=systemFunction(Gain*input, bw=7e9, sr=4.2e9, dt=timestep, oversampleFactor=40, asymmetry=+0.1, delay=140e-12)#*1.02
	#reference2=systemFunction(Gain*input2, bw=7e9, sr=4.2e9, dt=timestep, oversampleFactor=40, asymmetry=+0.1, delay=140e-12)#*1.02

	# filter=np.array([0,0,0.125,0.75,0.125])
	# reference=np.convolve(0.5*np.tanh(2*np.convolve(input, filter, mode='same')), filter, mode='same')*1.58
	# reference2=np.convolve(0.5*np.tanh(2*np.convolve(input2, filter, mode='same')), filter, mode='same')*1.58
	#Add skew to output signal
		
	reference=yMeasured*1
	reference2=yMeasured2*1
	
	# reference,_,_ = boostConverter(input, dt=times[1]-times[0], oversampleFactor=40)
	# reference2,_,_ = boostConverter(input2, dt=times[1]-times[0], oversampleFactor=40)
	# reference-=24
	# reference2-=24


	
	#--------------------------------------------------------------------------------


	
#test----------------------------------
	if(0):
		system.h0=0
		system.h1=1e-9*np.ones(shape=50)
		system.h2=1e-9*np.ones(shape=(25,25))
		system.h3=1e-9*np.ones(shape=(25,25,25))
		
		system.h1[7:18]=np.array([-0.05,-0.1,0.2,0.3,-0.4,-0.9,-0.4,0.3,0.2,-0.1,-0.05])

		#system.h2[12,12]=test
		
		system.h3[10,10,10]=-0.5
		system.h3[12,12,12]=1
		system.h3[14,14,14]=-0.5
		#system.h3=system.h3

		
		
		ltest=3000
		out=np.zeros(ltest)
		inp=np.zeros(ltest)
		inp[60]=0.1
		
		fexcitation=2100e6
		excitationI=0.03*np.sin(2*np.pi*fexcitation*91e-12*np.arange(ltest))
		excitationQ=0.03*np.sin(np.pi/2+2*np.pi*fexcitation*91e-12*np.arange(ltest))
		
		inp[1500:]=excitationI[1500:]
		
		sec1=time.time()
		for iteration in np.arange(system.len1,ltest-1):
			#out[iteration]=system.H(x=inp+0.5*out, oversampleFactor=1)[iteration]
			out[iteration]=system.H(x=inp[iteration-system.len1+1:iteration+1]+0.5*out[iteration-system.len1:iteration], oversampleFactor=1)[-1]
		sec2=time.time()
		print(sec2-sec1)
		

		
		#plt.plot(inp)
		#plt.plot(out)
		
		demodI=out*excitationI
		demodQ=out*excitationQ
		
		demodLPI=np.convolve(demodI,0.05*np.ones(20), mode='same')
		demodLPQ=np.convolve(demodQ,0.05*np.ones(20), mode='same')
		
		#plt.plot(demodLPI)
		#plt.plot(demodLPQ)
		
		phi=np.arctan2(demodLPQ,demodLPI)
		plt.plot(phi)
		
		plt.show()
	
	
#-------------------------------------
	#Calculate output of Volterra System
	
	output=system.H(x=input*1.0, oversampleFactor=order)
	output2=system.H(x=input2*1.0, oversampleFactor=order)
	output21=system.H(x=input*1.0/100, oversampleFactor=order)*100

#Test---	
	# output,_=HPfilter(output, np.arange(len(output))/(3*128), fstop=23)
	# #output=np.cumsum(output)
	# plt.plot(np.arange(len(output))/len(output)*3*128, 20*np.log10(np.abs(np.fft.fft(output))))
	# plt.show()
#---
	
	#Calculate difference between Volterra System and reference system
	nrmsd =np.sqrt(np.mean((reference[50:-50]-output[50:-50])**2))/np.sqrt(np.mean((reference[50:-50])**2))
	nrmsd2=np.sqrt(np.mean((reference2[50:-50]-output2[50:-50])**2))/np.sqrt(np.mean((reference2[50:-50])**2))
	#Error when small-signal model is applied to large-signal conditions
	nrmsd21=np.sqrt(np.mean((reference[50:-50]-output21[50:-50])**2))/np.sqrt(np.mean((reference[50:-50])**2))
	print('Normalized Root mean square deviations:')
	print('Large-Signal, Volterra model: '+'{:.3f}%'.format(100*nrmsd))
	print('Small-Signal, Volterra model:  '+'{:.3f}%'.format(100*nrmsd2))
	print('Large-Signal, linear model:  '+'{:.3f}%'.format(100*nrmsd21))


	#Oversampling for better depiction(equals sinc interpolation)
	
	visualOversample=10
	
	inputlength = inputlength*visualOversample
	_, input=oversample(times, input, visualOversample)
	_, input2=oversample(times, input2, visualOversample)
	_, output=oversample(times, output, visualOversample)
	_, output2=oversample(times, output2, visualOversample)
	#_, yMeasured = oversample(times, yMeasured, visualOversample)
	#_, yMeasured2 = oversample(times, yMeasured2, visualOversample)
	_, reference=oversample(times, reference, visualOversample)
	times, reference2=oversample(times, reference2, visualOversample)
	#Plot outputs-----------------------------------------------------
	
	color7b ='#f5a300'#medium orange
	color7c ='#d28700'#dark orange
	color1c ='#004e8a'#dark blue
	color10c ='#951169'#dark violet
	color11a = '#804597'
	color10a = '#c9308e'
	color1a  = '#5d85c3'

	font = FontProperties()
	font.set_family('serif')
	font.set_name('Times New Roman')
	#font.set_style('italic')
	font.set_size('12')


#Plot the system output, and reference outputs


	if(1):
		zoomFactor=1
		#16
		indexShift=int(0*inputlength/zoomFactor)#float value determines shift to the left
		#1*
		indexZero=int(inputlength/2+indexShift)
		indexLow=int(inputlength/2-inputlength/2/zoomFactor + indexShift)
		indexHigh=int(inputlength/2+inputlength/2/zoomFactor + indexShift)

		fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=90, constrained_layout=True)
		
		fontproperties=font
		

		#ax.set_xticks(np.arange(0, 1+len(input), len(input)/4))
		#ax.set_yticks(np.arange(-0.6, 0.8, 0.2))
		#ax.set_ylim(-0.7, 0.7)

		markertype='o'
		#markertype=''
		
		#Volterra outputs
		ax.plot(times[indexLow:indexHigh]-times[indexZero], output[indexLow:indexHigh], color=color1c, linewidth=3.5, label='Volterra Response, LS', marker=markertype, markevery=visualOversample)
		ax.plot(times[indexLow:indexHigh]-times[indexZero], output2[indexLow:indexHigh]/ssFactor, color=color7c, linewidth=3.5, label='Volterra Response, SS +'+str(ssAttenuation)+r'$\, dB$', marker=markertype, markevery=visualOversample)
		
		ax.plot(times[indexLow:indexHigh]-times[indexZero], input[indexLow:indexHigh], color=color10c, linewidth=2.5, label='Input, LS', marker=markertype, markevery=visualOversample)
		ax.plot(times[indexLow:indexHigh]-times[indexZero], input2[indexLow:indexHigh]/ssFactor, color='red', linewidth=2.5, label='Input, SS +'+str(ssAttenuation)+r'$\, dB$', marker=markertype, markevery=visualOversample)
		
		# if(0):#Measured 
			# ax.plot(times[indexLow:indexHigh]-times[indexZero], yMeasured[indexLow:indexHigh], color='lightblue', linewidth=1.5, label='Actual Response, LS')
			# ax.plot(times[indexLow:indexHigh]-times[indexZero], yMeasured2[indexLow:indexHigh]/ssFactor, color='orange', linewidth=1.5, label='Measured Response, SS +'+str(ssAttenuation)+'$\, dB$')

		if(1):#Simulated 
			ax.plot(times[indexLow:indexHigh]-times[indexZero], reference[indexLow:indexHigh], color='lightblue', linewidth=1.5, linestyle='--', label='Actual System, LS', marker=markertype, markevery=visualOversample)
			ax.plot(times[indexLow:indexHigh]-times[indexZero], reference2[indexLow:indexHigh]/ssFactor, color='orange', linewidth=1.5, linestyle='--', label='Actual System, SS +'+str(ssAttenuation)+r'$\, dB$', marker=markertype, markevery=visualOversample)

		

		ax.set_xlabel('Time $[s]$', fontproperties=font)
		ax.set_ylabel('Voltage $[V]$', fontproperties=font)
		ax.legend()
		ax.grid(visible=True)
		
		
		plt.savefig('Plots/FigureOutput/Results_InOut.pdf', dpi=90, transparent=False, bbox_inches='tight')

		
		
	if(0):#Eye diagram
		fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(6,4), dpi=90, constrained_layout=True)
		
		
		outputsSliced=  output[indexLow+11:indexHigh-(2*bitLength*visualOversample-11)].reshape((int(inputlength/(2*bitLength*visualOversample))-1, 2*bitLength*visualOversample))
		inputsSliced=  input[indexLow:indexHigh+0-(2*bitLength*visualOversample-0)].reshape((int(inputlength/(2*bitLength*visualOversample))-1, 2*bitLength*visualOversample))
		
		ax31.hist2d(np.tile(timestep/visualOversample*np.arange(2*bitLength*visualOversample), len(outputsSliced)), np.ravel(outputsSliced), bins=(2*bitLength*visualOversample, 200), c='inferno')
		ax32.hist2d(np.tile(timestep/visualOversample*np.arange(2*bitLength*visualOversample), len(inputsSliced)), np.ravel(inputsSliced), bins=(2*bitLength*visualOversample, 200), c='inferno')
	

	#Plotting
	Volterra_PlotSystem.plotOverview(system, Amplitude, order*timestep)
	
	
	
	Volterra_PlotSystem.plot1(system, 0, system.len1)
	Volterra_PlotSystem.plot2(system, 0, system.len2)
	Volterra_PlotSystem.plot3(system, 0, system.len3)
	

		
	plt.show()
	
	# plt.plot(sg.minimum_phase(np.convolve(system.h1,np.flip(system.h1), mode='full'), method='homomorphic', n_fft=3000))
	# plt.plot(system.h1)
	# plt.show()
	
	#plt.plot(np.abs(np.fft.fft(sg.minimum_phase(np.convolve(system.h1,np.flip(system.h1), mode='full'), method='homomorphic', n_fft=3000))))
	#plt.plot(np.abs(np.fft.fft(system.h1)))
	#plt.show()

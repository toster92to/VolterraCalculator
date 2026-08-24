import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal as sg

from Decorrelation5_OCL import deCorrelateFast
from Volterra_Kernels import koepplDownsampling
from Volterra_Kernels import calc_cyx
from Volterra_Kernels import calc_cxx


def findSystemResponse(x, y, length1):
	xDec = deCorrelateFast(times, x, x, alpha1=0.7, alpha2=0.5, alpha3=0.5, alpha4=0.5, alpha5=0.5, length1=length1, length2=2, length3=2, length4=2, length5=2, iterations=3, iterations_c2=5, iterations_initial=35, write=True, name='', chunks=128, debug=False)

	#Calculate overall system properties, and noise properties. 
	hOverall=calc_cyx(0, 1000000, 2*length1-1, y, xDec[length1:])[:2*length1-1]/np.var(xDec)
	hNoise=calc_cyx(0, 1000000, 2*length1-1, x, xDec[length1:])[:2*length1-1]/np.var(xDec)
	
	#Isolate DUT from the overall system, by removing noise properties.
	hDUT=np.real(np.fft.ifft(np.fft.fft(hOverall)/np.fft.fft(hNoise)))
	hDUT=np.roll(hDUT, length1)
	
	# plt.stem(hOverall, 'r')
	# plt.stem(hNoise, 'g')
	# plt.stem(hDUT, 'b')
	# plt.show()

	return(hDUT)



def findFrontendNoiseKernel(xBar, frontendNoise, length1):
	
	xBar2=xBar+frontendNoise

	xBar-=np.mean(xBar)
	xBar2-=np.mean(xBar2)

	##Very old version
	# #Prototype System
	# y=xBar
	
	# #Calculate the kernels describing the relationship between xBar/y and xBar2/y
	# g=findSystemResponse(xBar, y, length1)
	# h=findSystemResponse(xBar2, y, length1)
	
	# gDivh=np.real(np.fft.ifft(np.fft.fft(g)/np.fft.fft(h)))#[:length1]
	# gDivh=np.roll(gDivh, length1-1)
	
	# hDivg=np.real(np.fft.ifft(np.fft.fft(h)/np.fft.fft(g)))#[:length1]
	# hDivg=np.roll(hDivg, length1-1)
	
	# frontendNoiseKernel=gDivh
	# invFrontendNoiseKernel=hDivg
	
	##Note: This is overly complicated. It could probably also be done by the following code snippets
	
	##New Version. Tested to work. Can also be adapted to volterra noise systems, later:------------------
	# h=findSystemResponse(xBar2, xBar, length1)
	# frontendNoiseKernel=np.real(np.fft.ifft(1/np.fft.fft(h)))
	# invFrontendNoiseKernel=h
	
	#frontendNoiseKernel=np.real(np.fft.ifft(1/np.fft.fft(h)))
	#------------------
	
	#Another version. This only works for linear noise systems, but is much faster to calculate:
	cxbxb=calc_cxx(0, 1000000, length1, xBar)[:length1]
	cxbxb=np.concatenate((cxbxb[-1:0:-1],cxbxb))
	
	cn1n1=calc_cxx(0, 1000000, length1, frontendNoise)[:length1]
	cn1n1=np.concatenate((cn1n1[-1:0:-1],cn1n1))
	
	frontendNoiseKernel=np.real(np.fft.ifft((np.fft.fft(cxbxb)+np.fft.fft(cn1n1))/np.fft.fft(cxbxb)))
	invFrontendNoiseKernel=np.real(np.fft.ifft(1/np.fft.fft(frontendNoiseKernel)))
	
	frontendNoiseKernel=np.roll(frontendNoiseKernel, length1-1)
	invFrontendNoiseKernel=np.roll(invFrontendNoiseKernel, length1-1)
	
	

	#frontendNoiseKernel=np.roll(frontendNoiseKernel, -1)
	#invFrontendNoiseKernel=np.roll(invFrontendNoiseKernel, -1)
	# plt.stem(g, 'r')
	# plt.stem(h, 'b')
	# plt.stem(gDivh, 'g')
	# plt.stem(hDivg, 'y')
	# plt.show()
	
	
	return(frontendNoiseKernel, invFrontendNoiseKernel)




datastart=0
datalength=1000000000
length1=100


times=np.arange(datalength)

#Simulation----------------------------------
if(0):
	xOrig=np.random.normal(0,1,datalength)#original signal
	#xOrig=xOrig+0.03*xOrig**3+0.03*xOrig**2
	x=np.convolve(xOrig, ([0,0,0,0.9,0.2,0.15,-0.1]), mode='same')#signal including transfer function
	xReference=np.copy(x)

	frontendNoise=np.random.normal(0,0.2,datalength)
	frontendNoise2=np.random.normal(0,0.2,datalength)
	frontendNoise=np.convolve(frontendNoise, ([0,0,0,0.9,-0.2,0.05,-0.1]), mode='same')#signal including transfer function
	frontendNoise2=np.convolve(frontendNoise2, ([0,0,0,0.9,-0.2,0.05,-0.1]), mode='same')#signal including transfer function


	xBar=x+frontendNoise#signal including frontend noise

#-------------------------------------
	#Measurement-------------------------------------
if(1):	
	filename = 'noise0'
	wdin='Volterra_Input_Data'
	if os.name == 'nt':  # Windows
		print('File: '+filename+'.csv')
		df = pd.read_csv(wdin+'/'+filename+'.csv', sep=',',
						 header=None, dtype=str)  # Oscilloscope Data
	else:  # Other OS (Linux, MacOS)
		absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filename}.csv'
		print('File: ' + absolute_path)
		df = pd.read_csv(absolute_path, sep=',',
						 header=None, dtype=str)

	# normal file
	times = np.transpose(df.values[0:])[0].astype(float)
	x = np.transpose(df.values[0:])[1].astype(
		float)  # Start with 0th row
	y = np.transpose(df.values[0:])[2].astype(
		float)  # Start with 0th row
	# decorrelated file
	# times		= np.transpose(df.values[0:])[0].astype(float)
	# xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
	# x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
	# y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

	# Check if data is too long or to short
	if (len(times) >= datastart+datalength):
		times = times[datastart:datastart+datalength]
		x = x[datastart:datastart+datalength]
		y = y[datastart:datastart+datalength]
	datalength = len(times)

	x0=np.copy(x)
	
	
	
	# filename = 'noise8'
	# wdin='Volterra_Input_Data'
	# if os.name == 'nt':  # Windows
		# print('File: '+filename+'.csv')
		# df = pd.read_csv(wdin+'/'+filename+'.csv', sep=',',
						 # header=None, dtype=str)  # Oscilloscope Data
	# else:  # Other OS (Linux, MacOS)
		# absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filename}.csv'
		# print('File: ' + absolute_path)
		# df = pd.read_csv(absolute_path, sep=',',
						 # header=None, dtype=str)

	# # normal file
	# times = np.transpose(df.values[0:])[0].astype(float)
	# x = np.transpose(df.values[0:])[1].astype(
		# float)  # Start with 0th row
	# y = np.transpose(df.values[0:])[2].astype(
		# float)  # Start with 0th row
	# # decorrelated file
	# # times		= np.transpose(df.values[0:])[0].astype(float)
	# # xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
	# # x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
	# # y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

	# # Check if data is too long or to short
	# if (len(times) >= datastart+datalength):
		# times = times[datastart:datastart+datalength]
		# x = x[datastart:datastart+datalength]
		# y = y[datastart:datastart+datalength]
	# datalength = len(times)

	# xBar8=np.copy(x)




	filename = 'noiseempty'
	wdin='Volterra_Input_Data'
	if os.name == 'nt':  # Windows
		print('File: '+filename+'.csv')
		df = pd.read_csv(wdin+'/'+filename+'.csv', sep=',',
						 header=None, dtype=str)  # Oscilloscope Data
	else:  # Other OS (Linux, MacOS)
		absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filename}.csv'
		print('File: ' + absolute_path)
		df = pd.read_csv(absolute_path, sep=',',
						 header=None, dtype=str)

	# normal file
	times = np.transpose(df.values[0:])[0].astype(float)
	x = np.transpose(df.values[0:])[1].astype(
		float)  # Start with 0th row
	y = np.transpose(df.values[0:])[2].astype(
		float)  # Start with 0th row
	# decorrelated file
	# times		= np.transpose(df.values[0:])[0].astype(float)
	# xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
	# x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
	# y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

	# Check if data is too long or to short
	if (len(times) >= datastart+datalength):
		times = times[datastart:datastart+datalength]
		x = x[datastart:datastart+datalength]
		y = y[datastart:datastart+datalength]
	datalength = len(times)

	frontendNoise=np.copy(x)










#Downsampling of the signals
	_, x0, _ = koepplDownsampling(datalength, times, x0, y, 3, 2.7e9)
	#_, xBar8, _ = koepplDownsampling(datalength, times, xBar8, y, 3, 2.7e9)
	times, frontendNoise, _ = koepplDownsampling(datalength, times, frontendNoise, y, 3, 2.7e9)
	
	x0=np.convolve(x0, ([0,0,0,0.9,0.2,-0.1,-0.05]),mode='same')
	
	frontendNoiseA=np.random.normal(loc=0, scale=0.01, size=len(times))
	frontendNoiseB=np.random.normal(loc=0, scale=0.01, size=len(times))
	frontendNoiseA=np.convolve(frontendNoiseA, ([0,1.5,-0.5]), mode='same')
	frontendNoiseB=np.convolve(frontendNoiseB, ([0,1.5,-0.5]), mode='same')
	
	
	xBar0=x0+frontendNoiseA
	
	y=np.convolve(x0, ([0,0,0,0,0,0,0,0,0,0,0.9,0.1,0.1,0.1,-0.1,-0.05,-0.03,0.1,0.05,0.01]),mode='same')

	print(np.sqrt(np.var(xBar0)))
	print(np.sqrt(np.var(x0)))
	print(np.sqrt(np.var(frontendNoiseA)))
	
#---------------------------------------------------------------	


#As a reference: Calculate ground truth
kernelOriginal=findSystemResponse(x0,y,length1)


#Estimate the kernel from noisy input data
kernelEstimated=findSystemResponse(xBar0,y,length1)

#Apply correction
FNKernel0, iFNKernel0 = findFrontendNoiseKernel(xBar0, frontendNoiseB, length1)
kernelCorrected=np.convolve(FNKernel0,kernelEstimated, mode='same')



plt.stem(kernelOriginal, linefmt='black', label='Actual system')
plt.stem(kernelEstimated, linefmt='tomato', label='First estimate')
plt.stem(kernelCorrected, linefmt='cornflowerblue', label='Corrected estimate')
plt.legend()
plt.show()


plt.plot(5.4e9*np.arange(len(kernelOriginal))/len(kernelOriginal), 20*np.log10(np.abs(np.fft.fft(kernelOriginal))), 'r', label='Actual system', color='black')
plt.plot(5.4e9*np.arange(len(kernelOriginal))/len(kernelOriginal), 20*np.log10(np.abs(np.fft.fft(kernelEstimated))), 'b', label='First estimate', color='tomato')
plt.plot(5.4e9*np.arange(len(kernelOriginal))/len(kernelOriginal), 20*np.log10(np.abs(np.fft.fft(kernelCorrected))), 'g', label='Corrected estimate', color='cornflowerblue', linestyle='dashed')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('System response')
plt.grid()
plt.legend()
plt.show()


print(np.var(kernelOriginal))
print(np.var(kernelCorrected))
print(np.var(kernelEstimated))


# -----------------------------------------------------------
# Spectrum data
fs=1/(times[1]-times[0])

fwelch, xwelch = sg.welch(x0, fs, nperseg=100)
fwelch2, xwelch2 = sg.welch(xBar0, fs, nperseg=100)
fwelch3, xwelch3 = sg.welch(frontendNoiseB, fs, nperseg=100)
# xfft=20*np.log(abs(np.fft.fft(x)**2)/len(x))[0:round(len(x)/2)]

RBW = 10e6
Z0 = 50
Pwelch = xwelch/Z0
PwelchMean = np.mean(Pwelch)
Pwelch2 = xwelch2/Z0
Pwelch3 = xwelch3/Z0
dfwelch = fwelch[1]-fwelch[0]
dBmwelch = 10*np.log10(Pwelch/0.001)
dBmwelch2 = 10*np.log10(Pwelch2/0.001)
dBmwelch3 = 10*np.log10(Pwelch3/0.001)
dBmwelchMean = 10*np.log10(PwelchMean/0.001)


plt.plot(fwelch, dBmwelch, linestyle='dashed', label='x', color='black')
plt.plot(fwelch, dBmwelch2 ,label='xBar', color='cornflowerblue')
plt.plot(fwelch, dBmwelch3,linestyle='dashed',label='frontendNoise', color='tomato')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dBm/Hz)')
plt.title('Input Signals')
plt.grid()
plt.legend()
plt.show()



# plt.stem(kBar8, 'r')
# plt.stem(kCorrected8, 'b')
# plt.stem(kBar0, 'g')
# #plt.stem(originalFilter, 'k')


# plt.show()







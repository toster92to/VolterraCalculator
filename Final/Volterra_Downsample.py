import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D

from scipy import signal as sg
from scipy.stats import norm
from scipy.stats import gaussian_kde


import pandas as pd
import signal
import numpy as np
import csv
import math

def LPfilter(x,times,fstop):
	N=501
	
	#Use zero padding if input array is shorter than filter length
	xin=x
	if len(x)<N:
		xin=np.concatenate((np.zeros(N), xin, np.zeros(N)))
	
	datalength=len(xin)#Maximum length of the data to be used. 
	timestep=times[1]-times[0]
	fs=1/timestep#10e9
	if (fstop<fs/2):
		#First: Create Brickwall Filter
		lobewidth=(12*np.pi/(N-1))#Lobe width of blackman window
		limitIndexFIR= int(N * fstop*timestep*(1-lobewidth/2/np.pi/2))#fmax=limit/N * fsample, place Frequency limit 1/2 lobewidth below cutoff frequency
		filterF=np.zeros(N)
		filterF[0:limitIndexFIR]=np.ones(limitIndexFIR)
		filterF[N-limitIndexFIR:N]=np.ones(limitIndexFIR)#Filter in Frequency domain

		#Calculate Brickwall Filter in Time domain
		filterT=np.roll(np.real(np.fft.ifft(filterF)), int(N/2))
		#print(np.abs(filterT))
		
		#Calculate window function
		window=np.zeros(N)
		for n in range(N):
			window[n] = 0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))#Blackman Window
		
		#Apply window to filter, normalize
		filterWindowed=filterT*window
		filterWindowed=filterWindowed/np.sum(filterWindowed)

		#Apply Filter to input Signal
		xout=np.convolve(xin, filterWindowed, mode='same')
		tout=times
		
		#Remove zero padding if input array is shorter than filter length
		if len(x)<N:
			xout=xout[N:-N]
	#------------------------------------------------------------
		return(xout, tout)
	else: return(0,0)



def HPfilter(x,times,fstop):
	N=501
	
	#Use zero padding if input array is shorter than filter length
	xin=x
	if len(x)<N:
		xin=np.concatenate((np.zeros(N), xin, np.zeros(N)))
	
	datalength=len(xin)#Maximum length of the data to be used. 
	timestep=times[1]-times[0]
	fs=1/timestep#10e9
	if (fstop<fs/2):
		#First: Create Brickwall Filter
		lobewidth=(12*np.pi/(N-1))#Lobe width of blackman window
		limitIndexFIR= int(N * fstop*timestep*(1-lobewidth/2/np.pi/2))#fmax=limit/N * fsample, place Frequency limit 1/2 lobewidth below cutoff frequency
		filterF=np.ones(N)
		filterF[0:limitIndexFIR]=np.zeros(limitIndexFIR)
		filterF[N-limitIndexFIR:N]=np.zeros(limitIndexFIR)#Filter in Frequency domain

		#Calculate Brickwall Filter in Time domain
		filterT=np.roll(np.real(np.fft.ifft(filterF)), int(N/2))
		#print(np.abs(filterT))
		
		#Calculate window function
		window=np.zeros(N)
		for n in range(N):
			window[n] = 0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))#Blackman Window
		
		#Apply window to filter, normalize
		filterWindowed=filterT*window
		

		#Apply Filter to input Signal
		xout=np.convolve(xin, filterWindowed, mode='same')
		tout=times
		
		#Remove zero padding if input array is shorter than filter length
		if len(x)<N:
			xout=xout[N:-N]
	#------------------------------------------------------------
		return(xout, tout)
	else: return(0,0)




def oversample(t, x, oversampleFactor):
	timestep=t[1]-t[0]
	xfft=np.fft.fft(x)
	xfftOversampled=np.zeros(len(x)*oversampleFactor, dtype=complex)

	
	#Calculate oversampled input signal
	xfftOversampled[0:int(len(x)/2)]=xfft[0:int(len(x)/2)]
	xfftOversampled[-int(len(x)/2):]=xfft[-int(len(x)/2):]
	xOversampled=np.real(np.fft.ifft(xfftOversampled))*oversampleFactor
	
	tOversampled=t[0]+np.arange(len(t)*oversampleFactor)*timestep/oversampleFactor
	return(tOversampled, xOversampled)	


def downsample(xin,times,fstop):
	datalength=len(xin)#Maximum length of the data to be used. 

		
	timestep=times[1]-times[0]
	fs=1/timestep#10e9
	

	if (fstop<fs/2):

		limitIndex=int(fstop/fs*datalength+0.5)
		
		
		#Experimental: FIR pre-filtering-------------------------------
		
		# if(0):
			# #First: Create Brickwall Filter
			
			# N=500
			# lobewidth=(12*np.pi/(N-1))#Lobe width of blackman window
			# limitIndexFIR= int(N * fstop*timestep*(1-lobewidth/2/np.pi/2))#fmax=limit/N * fsample, place Frequency limit 1/2 lobewidth below cutoff frequency
			# filterF=np.zeros(N)
			# filterF[0:limitIndexFIR]=np.ones(limitIndexFIR)
			# filterF[N-limitIndexFIR:N]=np.ones(limitIndexFIR)#Filter in Frequency domain
	
			# #Calculate Brickwall Filter in Time domain
			# filterT=np.roll((np.fft.ifft(filterF)), int(N/2))
			# #print(np.abs(filterT))
			
			# #Calculate window function
			# window=np.zeros(N)
			# for n in range(N):
				# window[n] = 0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))#Blackman Window
			
			# #Apply window to filter
			# filterWindowed=filterT*window
	
			# #Apply Filter to input Signal
			# xin=np.convolve(xin, filterWindowed, mode='same')
		#------------------------------------------------------------

		#FFT Downsampling

		xfft=np.fft.fft(xin)/len(xin)
		xfft=np.concatenate((xfft[0:limitIndex], xfft[datalength-limitIndex:datalength]))
		xout=np.real(np.fft.ifft(xfft))*len(xfft)
		fs=2*fstop
		timestep=1/fs
		tout=np.arange(0, len(xout)*timestep, timestep)
		return(xout, tout)		
	else: 
		print('Error: Output sample rate higher than input timestep!')
		return(0,0)

def fineShift(times, x, shiftSamples=0.0, shiftTime=0.0):
	timestep=times[1]-times[0]

	shift=shiftSamples + shiftTime/timestep

	xfft=np.fft.fft(x)

	phaseShifter=np.zeros(len(xfft), dtype=complex)
	phaseShifter[:int(len(xfft)/2)+1]=np.exp(-1j*2*np.pi*shift/len(xfft)*np.arange(0, int(len(xfft)/2)+1, 1), dtype=complex)
	phaseShifter[int(len(xfft)/2)+1:]=np.exp(-1j*2*np.pi*shift/len(xfft)*np.arange(-int(len(xfft)/2)+1, 0, 1), dtype=complex)
	xfft_shifted=xfft*phaseShifter

	x_shifted=np.real(np.fft.ifft(xfft_shifted))
	#plt.plot(np.angle(phaseShifter))
	#plt.show()
	return(times, x_shifted)
	

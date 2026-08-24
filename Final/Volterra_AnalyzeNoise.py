import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D

from scipy import signal as sg
from scipy.stats import norm
from scipy.stats import gaussian_kde
from Volterra_Downsample import downsample
from scipy.special import erf
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager
from scipy.optimize import curve_fit

# from EntropyHub import ApEn

import pandas as pd
import signal
import numpy as np
import csv
import math


def calcCrossX(x, powers, tau):
	y = np.zeros(len(x))
	for n in np.arange(np.max(tau), len(x)):
		y[n] = x[n-tau[0]]**powers[0] * \
			x[n-tau[1]]**powers[1]*x[n-tau[2]]**powers[2]
	return y


np.set_printoptions(precision=4)


def modelFunction(x, *a):
	y = np.zeros(len(x))
	for i in np.arange(len(a)):
		y = y+x**i * a[i]
	return y


def analyzeDistortion(x, limStdDevs, nrBins, mode='normal', order=7):
	varX = np.var(x)
	stdDev = np.sqrt(varX)

	vals = np.linspace(-limStdDevs*stdDev, limStdDevs*stdDev, nrBins)

	probs = 1/2*(1+erf(vals/stdDev/np.sqrt(2)))
	Vout = np.quantile(x, probs)

	# Determine the polynomial describing a static nonlinearity at the output of the system
	bLow = np.ones(order+1)*-1e2
	bHigh = np.ones(order+1)*1e2

	# if mode=='ssInvariant':
	# bLow[1]=0.999
	# bHigh[1]=1.001

	initGuess = np.zeros(order+1)
	initGuess[1] = 1

	paramsFit, paramsCovFit = curve_fit(
		modelFunction, vals[60:-60], Vout[60:-60], p0=initGuess, bounds=(bLow, bHigh))
	invParamsFit, invParamsCovFit = curve_fit(
		modelFunction, Vout[60:-60], vals[60:-60], p0=initGuess, bounds=(bLow, bHigh))

	if mode == 'ssInvariant':
		params1 = paramsFit[1]
		invParams1 = invParamsFit[1]
		for i in np.arange(order+1):
			paramsFit[i] = paramsFit[i]/(params1**i)
			invParamsFit[i] = invParamsFit[i]/(invParams1**i)

	return ((vals, Vout, paramsFit, invParamsFit))

# def estimateUndistortedVariance(x, invParamsFit):
	# x=modelFunction(x, invParamsFit[0], invParamsFit[1], invParamsFit[2], invParamsFit[3])
	# return np.var(x)


if __name__ == '__main__':
	limStdDevs = 5  # xaxis limits in std deviations for diagram 1+2
	nrBins = 300  # Bins for PDF data
	datalength = 1000000  # Maximum length of the data to be used.
	datastart = 6500

	# RefAttenuations
	# 8.25 for ADA4960
	# 8.25 for ADL5565
	# 0 for LMH5401
	# 0 for LMH3401

	refAttenuation = 0  # 8.25#dB, attenuation of reference channel
	refAttenuation2 = 0

	refFactor = 10**(-refAttenuation/20)
	refFactor2 = 10**(-refAttenuation2/20)

	#wdin = 'Volterra_Input_Data/'  # Input Working directory
	wdin = ''  # Input Working directory

	# ada4960
	# ab 13: mit 8.25dB Dämpfung
	# ab 46: mit Marcels neuer Hardware

	# Original files: ([times[n], x[n], y[n]])
	# Decorrelated files: ([times[n], xmodified[n], x[n], y[n]])

	Filename = 'LMH3401XY16h_dec.csv'
	df = pd.read_csv(wdin+Filename, sep=',', header=None,
					 dtype=str)  # Oscilloscope Data
	x = np.transpose(df.values[0:])[1].astype(
		float)  # Start with 0th row, take 1st column
	inputTimes = np.transpose(df.values[0:])[0].astype(float)  # *30e-12

	Filename2 = 'LMH3401XY16h_dec.csv'
	df = pd.read_csv(wdin+Filename2, sep=',', header=None,
					 dtype=str)  # Oscilloscope Data
	x2 = np.transpose(df.values[0:])[2].astype(float)  # Start with 0th row
	inputTimes2 = np.transpose(df.values[0:])[0].astype(float)  # *30e-12

	# Filename3='LinearDeembedding/deembedding_kernel.csv'
	# df=pd.read_csv(Filename3, sep=',', header=None, dtype=str)#Oscilloscope Data
	# deembeddingKernel=np.transpose(df.values[0:])[0].astype(float)#Start with 0th row

	if (0):  # Differentiation
		x = np.convolve(x, np.array([0, 1, -1]), mode='same')
		x2 = np.convolve(x2, np.array([0, 1, -1]), mode='same')

	if (0):  # Moving average
		x = np.convolve(x, np.concatenate(
			(np.zeros(10), np.ones(11))), mode='same')
		x2 = np.convolve(x2, np.concatenate(
			(np.zeros(10), np.ones(11))), mode='same')

	if (0):  # Integrate
		x = np.cumsum(x)
		x2 = np.cumsum(x2)
		x-=np.mean(x)
		x2-=np.mean(x2)

	print('')
	print('')
	print('Analyzing '+Filename)

	# df=pd.read_csv('Volterra_Kernels/MeasurementSimIn.csv', sep='\t', header=None, dtype=str)#LTSpice sim data
	# x=np.transpose(df.values[1:])[1].astype(float)#Start with 1st row
	# times=np.transpose(df.values[1:])[0].astype(float)

	# Differentiate
	# x[1:]=x[1:]-x[0:-1]

	if (len(x) < datalength+datastart):
		datalength = len(x)-datastart
	x = x[datastart:datalength+datastart]
	inputTimes = inputTimes[datastart:datalength+datastart]
	x2 = x2[datastart:datalength+datastart]
	inputTimes2 = inputTimes2[datastart:datalength+datastart]

	inputTimestep = inputTimes[1]-inputTimes[0]
	inputFs = 1/inputTimestep  # 10e9
	fs = inputFs

	print('Input samples:			   '+str(datalength))
	print('Input timestep:			  '+'{:.2e}'.format(inputTimestep))
	# -------------------------------------------------------------------------------------------
	# Downsampling
	if (0):
		fstop = 5e9
		x, times = downsample(x, inputTimes, fstop)
		x2, times = downsample(x2, inputTimes2, fstop)
		fs = 2*fstop
		datalength = len(x)
	# ----------------------------------------------------------------------------------
	timestep = 1/fs

	print('Samples after downsampling:  '+str(datalength))
	print('Timestep after downsampling: '+'{:.2e}'.format(timestep))

	print('Coherence: '+str(np.mean(sg.coherence(x, x2))))
	# print('Entropy x1: '+str(ApEn(x[:1000],  m=2, tau=1, r=0.01, Logx=np.exp(1))))
	# print('Entropy x2: '+str(ApEn(x2[:1000], m=2, tau=1, r=0.01, Logx=np.exp(1))))
	# Statistical data-------------------------------------------
	x = x/refFactor
	x2 = x2/refFactor2

	x = x-np.mean(x)
	x2 = x2-np.mean(x2)
	varX = np.sum(x**2)/len(x)
	varX2 = np.sum(x2**2)/len(x2)
	stdDev = np.sqrt(varX)
	stdDev2 = np.sqrt(varX2)

	print('*********Noise properties*********')
	print('Var	 x: '+'{:.5f}'.format(varX))
	print('StdDev  x: '+'{:.5f}'.format(stdDev))
	print('Var	x2: '+'{:.5f}'.format(varX2))
	print('StdDev x2: '+'{:.5f}'.format(stdDev2))

	# ch=sg.coherence(x, x2, fs=2*fstop)
	# plt.plot(ch[0], ch[1])
	# plt.show()

	# x[5:]+=   (x[5:]-x[4:-1])**3 - 0.5*x[5:]**3
	# #x[5:]+=   (x[5:])**3
	# for tau1 in ([1,2,3]):
	# for tau2 in ([1,2,3]):
	# for tau3 in ([1,2,3]):
	# print(tau1-1,tau2-1,tau3-1)
	# print(np.mean((x[10-tau1:-tau1]*x[10-tau2:-tau2]*x[10-tau3:-tau3])**2))

	# -----------------------------------------------------------
	# Spectrum data
	fwelch, xwelch = sg.welch(x, fs, nperseg=1000)
	fwelch2, xwelch2 = sg.welch(x2, fs, nperseg=1000)
	# xfft=20*np.log(abs(np.fft.fft(x)**2)/len(x))[0:round(len(x)/2)]

	RBW = 10e6
	Z0 = 50
	Pwelch = xwelch/Z0
	PwelchMean = np.mean(Pwelch)
	Pwelch2 = xwelch2/Z0
	dfwelch = fwelch[1]-fwelch[0]
	dBmwelch = 10*np.log10(Pwelch/0.001)
	dBmwelch2 = 10*np.log10(Pwelch2/0.001)
	dBmwelchMean = 10*np.log10(PwelchMean/0.001)
	# ------------------------------------------------------------
	# Est. DC Transfer Data
	# stepsize=2*limStdDevs*stdDev/len(PDFExpectedX)

	# CDFExpectedx=np.cumsum(PDFExpectedX)*stepsize#Cumulative density function
	# CDFx=np.cumsum(PDFx)*stepsize

	# probVals=np.linspace(-limStdDevs*stdDev, limStdDevs*stdDev, 20)

	fig, ax = plt.subplots()
	amplitude = 3*np.sqrt(np.var(x))
	amplitude2 = 3*np.sqrt(np.var(x2))

	fitOrder = 3
	fitWidth = 5
	vals, Vout, paramsFit, invParamsFit = analyzeDistortion(
		x, fitWidth, nrBins, mode='ssInvariant', order=fitOrder)
	ax.stem(np.arange(fitOrder+1), (amplitude)**np.arange(fitOrder+1)*paramsFit /
			(amplitude), linefmt='g', markerfmt='gD', label='1st Measurement')

	vals2, Vout2, paramsFit2, invParamsFit2 = analyzeDistortion(
		x2, fitWidth, nrBins, mode='ssInvariant', order=fitOrder)
	ax.stem(np.arange(fitOrder+1), (amplitude2)**np.arange(fitOrder+1)*paramsFit2 /
			(amplitude2), linefmt='r', markerfmt='ro', label='2nd Measurement')

	# ax.plot(vals, vals, label='Ideal')
	# ax.plot(vals, Vout, label='1st Measurement')
	# ax.plot(vals2, Vout2, label='2nd Measurement')

	ax.legend()
	ax.set_xlabel('Order')
	ax.set_ylabel('Rel. contribution')

	# plt.show()

	print('X Taylor coefficients:')
	print(paramsFit)

	# print('Inverse a0, a1, a2, a3:')
	# print(invParamsFit)

	print('X2 Taylor coefficients:')
	print(paramsFit2)

	# print('Inverse a0, a1, a2, a3:')
	# print(invParamsFit)

	# print(estimateUndistortedVariance(x, invParamsFit))
	# ------------------------------------------------------------
	# Relative occurrence of values
	kernel = gaussian_kde(x)
	PDFx = kernel(vals)

	kernel2 = gaussian_kde(x2)
	PDFx2 = kernel2(vals2)

	# ------------------------------------------
	# PDFmax=vals[np.argmax(PDFx)]
	# print('Max of PDF at: ' + '{:.2e}'.format(PDFmax) +'V')

	# stdDevExpected=(np.quantile(x, 1/2*(1+erf((0.5)/np.sqrt(2))))-np.quantile(x, 1/2*(1+erf((-0.5)/np.sqrt(2)))))
	# meanExpected=(np.quantile(x, 0.5))
	# print(meanExpected)
	# print(stdDevExpected)

	# PDFExpectedX=norm.pdf(vals, loc=PDFmax, scale=1.155*np.sqrt(varX))#-3dB
	# PDFExpectedX=norm.pdf(vals, loc=0.03, scale=1.18*np.sqrt(varX))#-10dB
	PDFExpectedX = norm.pdf(vals, loc=0, scale=np.sqrt(varX))  # ideal
	PDFExpectedX2 = norm.pdf(vals2, loc=0, scale=np.sqrt(varX2))
	# PDFExpectedX=norm.pdf(vals, loc=meanExpected, scale=stdDevExpected)#expected from small-signal(+-0.5stdDev)
	# --------------------------------------------------------------

	# Calculate by which factor a pth order correlation will be affected by nonlinearly distorted noise
	# introduced after thesis was submitted

	# Integral(x^(2p)*PDF(x))dx = E[x^p * x^p]
	for order in np.arange(0, 7+1):
		print('Relative contribution of order '+str(order)+': ' +
			  str(np.sum(vals**(2*order)*PDFx)/np.sum(vals**(2*order)*PDFExpectedX)))

	# ------------------------------------------

	color7b = '#f5a300'  # medium orange
	color7c = '#d28700'  # dark orange
	color1c = '#004e8a'  # dark blue
	color10c = '#951169'  # dark violet
	color11a = '#804597'
	color10a = '#c9308e'
	color1a = '#5d85c3'

	font = FontProperties()
	# font.set_family('serif')
	# font.set_name('Times New Roman')
	# font.set_style('italic')
	# font.set_size('22')

	if (0):  # Single plot
		fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=90, constrained_layout=True)
		ax1_2 = ax1.twiny()
		ax1_2.set_xlabel("Standard Deviations", fontproperties=font)
		ax1_2.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)
		# [-stdDev, 0, stdDev])
		ax1_2.set_xticks(np.linspace(-3*stdDev, 3*stdDev, 7))
		ax1_2.set_xticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-1\sigma$',
							  r'$\mu$', r'$+1\sigma$', r'$+2\sigma$', r'$+3\sigma$'])
		ax1_2.grid(color='lightgrey', linestyle='dotted')
		ax1_2.xaxis.tick_top()
		ax1_2.xaxis.set_label_position('top')
		ax1_2.xaxis.label.set_color('grey')
		ax1_2.tick_params(axis='x', colors='grey')

		order = 1
		ax1.plot(vals, vals**(2*order)*PDFx/np.max(vals**(2*order)*PDFExpectedX),
				 color=color1a, label='x, measured, order=1', linewidth=3.5)
		order = 2
		ax1.plot(vals, vals**(2*order)*PDFx/np.max(vals**(2*order)*PDFExpectedX),
				 color=color7b, label='x, measured, order=2', linewidth=3.5)
		order = 3
		ax1.plot(vals, vals**(2*order)*PDFx/np.max(vals**(2*order)*PDFExpectedX),
				 color=color11a, label='x, measured, order=3', linewidth=3.5)
		ax1.plot(vals, vals**(2*order)*PDFExpectedX/np.max(vals**(2*order)*PDFExpectedX),
				 color='black', label='x, ideal, order=3', linestyle='dotted', linewidth=2.5)
		# ax1.plot(vals, PDFxCross, color='green', label='crossX')
		# ax1.plot(vals, PDFxCrossExpected, color='lightgreen', label='crossX expected', linestyle='--')
		ax1.grid(color='darkgrey')
		ax1.set_ylabel('relative contribution of $p$th order',
					   fontproperties=font)
		ax1.set_xlabel('Voltage $[V]$', fontproperties=font)
		ax1.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)
		ax1.legend()
		ax1.xaxis.tick_bottom()
		ax1.xaxis.set_label_position('bottom')

		plt.savefig('Plots/FigureOutput/Results_AnalyzeNoiseSingle.pdf',
					dpi=90, transparent=False, bbox_inches='tight')
		plt.show()

	if (1):  # All plots
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
			2, 2, figsize=(12, 8), dpi=90, constrained_layout=True)

		ax1_2 = ax1.twiny()
		ax1_2.set_xlabel("Standard Deviations", fontproperties=font)
		ax1_2.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)
		# [-stdDev, 0, stdDev])
		ax1_2.set_xticks(np.linspace(-3*stdDev, 3*stdDev, 7))
		ax1_2.set_xticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-1\sigma$',
							  r'$\mu$', r'$+1\sigma$', r'$+2\sigma$', r'$+3\sigma$'])
		ax1_2.grid(color='lightgrey', linestyle='dotted')
		ax1_2.xaxis.tick_top()
		ax1_2.xaxis.set_label_position('top')
		ax1_2.xaxis.label.set_color('grey')
		ax1_2.tick_params(axis='x', colors='grey')

		order = 0
		ax1.plot(vals, vals**(2*order)*PDFx, color=color7b,
				 label='x, measured', linewidth=2.5)
		ax1.plot(vals2, vals2**(2*order)*PDFx2, color=color10c,
				 label='x2, measured', linewidth=2.5, linestyle='--')
		ax1.plot(vals, vals**(2*order)*PDFExpectedX, color='black',
				 label='x, ideal', linestyle='dotted', linewidth=2.5)
		# ax1.plot(vals, PDFxCross, color='green', label='crossX')
		# ax1.plot(vals, PDFxCrossExpected, color='lightgreen', label='crossX expected', linestyle='--')
		ax1.grid(color='darkgrey')
		ax1.set_ylabel('PDF $[V^{-1}]$', fontproperties=font)
		ax1.set_xlabel('Voltage $[V]$', fontproperties=font)
		ax1.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)
		ax1.legend()
		ax1.xaxis.tick_bottom()
		ax1.xaxis.set_label_position('bottom')

		# Diagram 2-------------
		ax2_2 = ax2.twiny()
		ax2_2.set_xlabel("Standard Deviations", fontproperties=font)
		ax2_2.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)
		# [-stdDev, 0, stdDev])
		ax2_2.set_xticks(np.linspace(-3*stdDev, 3*stdDev, 7))
		ax2_2.set_xticklabels([r'$-3\sigma$', r'$-2\sigma$', r'$-1\sigma$',
							  r'$\mu$', r'$+1\sigma$', r'$+2\sigma$', r'$+3\sigma$'])
		ax2_2.grid(color='lightgrey', linestyle='dotted')
		ax2_2.xaxis.tick_top()
		ax2_2.xaxis.set_label_position('top')
		ax2_2.xaxis.label.set_color('grey')
		ax2_2.tick_params(axis='x', colors='grey')

		ax2.plot(vals, PDFx/PDFExpectedX, color=color7b,
				 label='x, measured', linewidth=2.5)
		ax2.plot(vals2, PDFx2/PDFExpectedX2, color=color10c,
				 label='x2, measured', linewidth=2.5, linestyle='--')
		ax2.plot(vals, np.ones(len(vals)), color='black',
				 label='x, ideal', linestyle='dotted', linewidth=2.5)
		ax2.legend()
		ax2.set_xlabel("Voltage $[V]$", fontproperties=font)
		ax2.set_ylabel("Relative PDF", fontproperties=font)
		ax2.set_ylim(0, 2)
		ax2.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)

		l = matplotlib.ticker.AutoLocator()
		l.create_dummy_axis()
		ax2ticks = l.tick_values(-(limStdDevs-1)*stdDev, (limStdDevs-1)*stdDev)

		ax2.set_xticks(ax2ticks)
		ax2.set_xticklabels(["{:.2f}".format(i).rstrip(
			'0').rstrip('.')+'V' for i in ax2ticks])
		ax2.xaxis.tick_bottom()
		ax2.xaxis.set_label_position('bottom')
		ax2.grid(color='darkgrey')
		# -------------------------
		# Diagram3
		# ax3.plot(xfft, color='blue', label='x')
		# ax3.plot(fwelch, dBmwelch2 - dBmwelch, color='orange', label='x2-x1', linewidth=2.5)
		ax3.plot(fwelch, dBmwelch, color=color7b,
				 label='x1, measured', linewidth=2.5)
		ax3.plot(fwelch, dBmwelch2, color=color10a,
				 label='x2, measured', linewidth=2.5, linestyle='--')
		ax3.plot(fwelch, dBmwelchMean*np.ones(len(fwelch)), color='black',
				 label='x, ideal', linestyle='dotted', linewidth=2.5)
		# ax3.semilogy(f, xwelch)
		ax3.set_ylim([-125, -80])
		# ax3.set_ylim([-30, 20])
		ax3.set_xlabel('Frequency $[Hz]$', fontproperties=font)
		ax3.set_ylabel('PSD $[dBm/Hz]$', fontproperties=font)
		ax3.grid(color='darkgrey')
		ax3.legend()

		# Diagram 4
		ax4.plot(vals, Vout, color=color7b, label='x, measured', linewidth=2.5)
		ax4.plot(vals2, Vout2, color=color10c, label='x2, measured',
				 linewidth=2.5, linestyle='--')
		ax4.plot((-limStdDevs*stdDev, limStdDevs*stdDev), (-limStdDevs*stdDev, limStdDevs *
				 stdDev), color='black', label='x, ideal', linestyle='dotted', linewidth=2.5)
		ax4.legend()
		# ax4.title.set_text('Expected DC Transfer')
		ax4.set_xlabel('Vin $[V]$', fontproperties=font)
		ax4.set_ylabel('Vout $[V]$', fontproperties=font)
		ax4.set_xlim(-limStdDevs*stdDev, limStdDevs*stdDev)
		ax4.set_ylim(-limStdDevs*stdDev, limStdDevs*stdDev)
		ax4.grid(color='darkgrey')

		plt.savefig('Plots/FigureOutput/Results_AnalyzeNoise.pdf',
					dpi=90, transparent=False, bbox_inches='tight')
		plt.show()

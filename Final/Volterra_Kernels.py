import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy import signal as sg
from scipy.stats import norm
from scipy.stats import gaussian_kde
from Volterra_Downsample import downsample, LPfilter, HPfilter
from Simulate_System import systemFunction
from Simulate_System import boostConverter
from Volterra_AnalyzeNoise import modelFunction
from Volterra_AnalyzeNoise import analyzeDistortion
from Volterra_File_Operations import writeKernels, readKernels
from Volterra_Operations import chainVolterraSystems, offDiagonalTruncation
from Volterra_In_Out_6 import VolterraResponse
from Volterra_File_Operations import invertKernelFiles
from Volterra_File_Operations import combineKernelFiles
import pyopencl as cl
import Decorrelation5_OCL
import pandas as pd
import signal
import numpy as np
import csv
import math
import time
from itertools import combinations_with_replacement
from itertools import permutations
import Volterra_PlotSystem
import os


def initializer():
	"""Ignore CTRL+C in the worker process."""
	signal.signal(signal.SIGINT, signal.SIG_IGN)


def delta(x):
	if x == 0:
		return 1
	else:
		return 0


def delta4(tau1, tau2, tau3, tau4):
	return 1 if len(set([tau1, tau2, tau3, tau4])) == 1 else 0


def doubleFactorial(n):
	result = 1

	if n % 2 == 0:  # even
		for i in np.arange(1, int(n/2)+1):
			result = result * 2*i

	if n % 2 == 1:  # odd
		for i in np.arange(1, int((n+1)/2)+1):
			result = result * (2*i-1)

	return (result)


def calcDiagonalFactor(tauList):
	dictionary = dict((i, tauList.count(i)) for i in tauList)

	# print(dictionary)
	product = 1
	for key in dictionary:
		product = product * doubleFactorial(2*dictionary[key]-1)
	return (product)


# Signal Processing--------------------------------------------------------------------------

def calc_cyx(n_start, n_stop, length1, x, y):
	c_yx = np.zeros(len(x))
	for tau1 in np.arange(length1):
		if (n_start-tau1) < 0:  # Check if negative values get out of bounds
			vecX1 = np.zeros(n_stop-n_start)
			lenFilled = n_stop-n_start-tau1
			vecX1[tau1:] = x[n_stop-lenFilled:n_stop] * \
				y[n_stop-lenFilled-tau1:n_stop-tau1]
		else:
			# Create 1st order input vector x1. This is not nice, need take care at array ends
			vecX1 = x[n_start:n_stop]*y[n_start-tau1:n_stop-tau1]
		c_yx[tau1] += np.sum(vecX1)

	c_yx = c_yx/len(x)
	return c_yx
	

def calc_cxx(n_start, n_stop, length1, x):
	c_xx=np.zeros(len(x))
	for tau1 in np.arange(length1):

		if (n_start-tau1)<0:#Check if negative values get out of bounds 
			vecX1=np.zeros(n_stop-n_start)
			lenFilled = n_stop-n_start-tau1
			vecX1[tau1:]=x[n_stop-lenFilled:n_stop]*x[n_stop-tau1-lenFilled:n_stop-tau1]
		else:
			vecX1=x[n_start:n_stop]*x[n_start-tau1:n_stop-tau1]#Create 1st order input vector x1. This is not nice, need take care at array ends		
		
		c_xx[tau1]+=np.sum(vecX1)

	c_xx=c_xx/len(x)
	return c_xx	


def koepplDownsampling(datalength, times, x, y, order, inputBandwidth):
	timesy = np.copy(times)

	print('Data length before downsampling: '+str(datalength))
	x, times = downsample(x, times, inputBandwidth)
	datalength = len(x)
	print('Data length after downsampling:  '+str(datalength))

	y, _ = downsample(y, timesy, order*inputBandwidth)
	y = y[0:order*datalength:order]
	return (times, x, y)


# Under construction: Multi-Variance characterization--------------------------------

def multivarK0(y):
	return np.mean(y)


def multivarK1(x, y, length1, A):
	n_start = length1
	n_stop = len(x)

	k_1 = np.zeros(shape=(length1))
	expValX1 = np.zeros(shape=(length1))  # Expected value of (x(n-tau1))^2
	for tau1 in np.arange(length1):
		if (n_start-tau1) < 0:  # Check if negative values get out of bounds
			vecX1 = np.zeros(n_stop-n_start)
			lenFilled = n_stop-n_start-tau1
			vecX1[tau1:] = x[n_stop-tau1-lenFilled:n_stop-tau1]
		else:
			# Create 1st order input vector x1. This is not nice, need take care at array ends
			vecX1 = x[n_start-tau1:n_stop-tau1]

		expValX1[tau1] = np.mean(vecX1**2)  # calculate expected value of x^2
		vecY = y[n_start:n_stop]
		k_1[tau1] = np.sum(vecX1*vecY) / (n_stop-n_start) / \
			math.factorial(1) / expValX1[tau1]
	return k_1


def multivarK2(x, y, length2, A, k0):
	n_start = length2
	n_stop = len(x)
	k_2 = np.zeros(shape=(length2, length2))

	# Expected value of (x(n-tau1)x(n-tau2))^2
	expValX2 = np.zeros(shape=(length2, length2))
	# Effective signal variance at [tau1, tau2] computed from expValX2
	A2eff = np.zeros(shape=(length2, length2))

	for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):

		if (n_start-tau1) < 0 or (n_start-tau2) < 0:  # Check if negative values get out of bounds
			vecX2 = np.zeros(n_stop-n_start)
			lenFilled = n_stop-n_start-max(tau1, tau2)
			vecX2[max(tau1, tau2):] = x[n_stop-tau1-lenFilled:n_stop -
										tau1]*x[n_stop-tau2-lenFilled:n_stop-tau2]
		else:
			# Create 2nd order input vector x2. This is not nice, need take care at array ends
			vecX2 = x[n_start-tau1:n_stop-tau1]*x[n_start-tau2:n_stop-tau2]
		# calculate expected value of x2^2
		expValX2[tau1, tau2] = np.mean(vecX2**2)
		A2eff[tau1, tau2] = np.sqrt(
			expValX2[tau1, tau2]/calcDiagonalFactor((tau1, tau2)))

		vecY = y[n_start:n_stop]
		k_2[tau1, tau2] = (np.sum(vecX2*vecY) / (n_stop-n_start) - A2eff[tau1, tau2]
						   * k0*delta(tau1-tau2)) / math.factorial(2) / A2eff[tau1, tau2]**2

		# set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
		perm = list(set(permutations([tau1, tau2])))
		for [t1, t2] in perm:
			# Fill eqivalent kernel positions with copied entries
			k_2[t1, t2] = k_2[tau1, tau2]

	return k_2


def multivarK3(x, y, length3, A, k1):
	n_start = length3
	n_stop = len(x)
	k_3 = np.zeros(shape=(length3, length3, length3))
	# Expected value of (x(n-tau1)x(n-tau2)x(n-tau3))^2
	expValX3 = np.zeros(shape=(length3, length3, length3))
	# Effective signal variance at [tau1, tau2, tau3] computed from expValX3
	A3eff = np.zeros(shape=(length3, length3, length3))

	for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):

		# Check if negative values get out of bounds
		if (n_start-tau1) < 0 or (n_start-tau2) < 0 or (n_start-tau3) < 0:
			vecX3 = np.zeros(n_stop-n_start)
			lenFilled = n_stop-n_start-max(tau1, tau2, tau3)
			vecX3[max(tau1, tau2, tau3):] = x[n_stop-tau1-lenFilled:n_stop-tau1] * \
				x[n_stop-tau2-lenFilled:n_stop-tau2] * \
				x[n_stop-tau3-lenFilled:n_stop-tau3]
		else:
			# Create 2nd order input vector x2. This is not nice, need take care at array ends
			vecX3 = x[n_start-tau1:n_stop-tau1] * \
				x[n_start-tau2:n_stop-tau2]*x[n_start-tau3:n_stop-tau3]
		# calculate expected value of x3^2
		expValX3[tau1, tau2, tau3] = np.mean(vecX3**2)
		A3eff[tau1, tau2, tau3] = np.cbrt(
			expValX3[tau1, tau2, tau3]/calcDiagonalFactor((tau1, tau2, tau3)))

		vecY = y[n_start:n_stop]
		k_3[tau1, tau2, tau3] = (np.sum(vecX3*vecY) / (n_stop-n_start) - A3eff[tau1, tau2, tau3]**2*(k1[tau1]*delta(
			tau2-tau3)+k1[tau2]*delta(tau1-tau3)+k1[tau3]*delta(tau1-tau2))) / math.factorial(3) / A3eff[tau1, tau2, tau3]**3

		# set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
		perm = list(set(permutations([tau1, tau2, tau3])))
		for [t1, t2, t3] in perm:
			# Fill eqivalent kernel positions with copied entries
			k_3[t1, t2, t3] = k_3[tau1, tau2, tau3]

	return k_3


def multivarK4(x, y, length4, A, k2):
	n_start = length4
	n_stop = len(x)
	k_4 = np.zeros(shape=(length4, length4, length4, length4))

	expValX4 = np.zeros(shape=(length4, length4, length4, length4))
	A4eff = np.zeros(shape=(length4, length4, length4, length4))

	for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):

		if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0):
			vecX4 = np.zeros(n_stop - n_start)
			lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4)
			vecX4[max(tau1, tau2, tau3, tau4):] = (x[n_stop - tau1 - lenFilled:n_stop - tau1] * x[n_stop - tau2 -
																								  lenFilled:n_stop - tau2] * x[n_stop - tau3 - lenFilled:n_stop - tau3] * x[n_stop - tau4 - lenFilled:n_stop - tau4])
		else:
			vecX4 = (
				x[n_start - tau1:n_stop - tau1] *
				x[n_start - tau2:n_stop - tau2] *
				x[n_start - tau3:n_stop - tau3] *
				x[n_start - tau4:n_stop - tau4]
			)

		expValX4[tau1, tau2, tau3, tau4] = np.mean(vecX4 ** 2)
		A4eff[tau1, tau2, tau3, tau4] = np.power(
			expValX4[tau1, tau2, tau3, tau4] /
			calcDiagonalFactor((tau1, tau2, tau3, tau4)), 1/4
		)

		vecY = y[n_start:n_stop]
		k_4[tau1, tau2, tau3, tau4] = (
			(np.sum(vecX4 * vecY) / (n_stop - n_start) -
			 A4eff[tau1, tau2, tau3, tau4] ** 2 *
			 (k2[tau1, tau2] * delta(tau3 - tau4) +
			  k2[tau1, tau3] * delta(tau2 - tau4) +
			  k2[tau1, tau4] * delta(tau2 - tau3) +
			  k2[tau2, tau3] * delta(tau1 - tau4) +
			  k2[tau2, tau4] * delta(tau1 - tau3) +
			  k2[tau3, tau4] * delta(tau1 - tau2))
			 ) / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4
		)

		perms = list(set(permutations([tau1, tau2, tau3, tau4])))
		for (t1, t2, t3, t4) in perms:
			k_4[t1, t2, t3, t4] = k_4[tau1, tau2, tau3, tau4]

	return k_4


def multivarK5(x, y, length5, A, k1, k3):
	n_start = length5
	n_stop = len(x)
	k_5 = np.zeros((length5, length5, length5, length5, length5))

	expValX5 = np.zeros((length5, length5, length5, length5, length5))
	A5eff = np.zeros((length5, length5, length5, length5, length5))

	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
		if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0) or (n_start - tau5 < 0):
			vecX5 = np.zeros(n_stop - n_start)
			lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4, tau5)
			vecX5[max(tau1, tau2, tau3, tau4, tau5):] = (
				x[n_stop - tau1 - lenFilled:n_stop - tau1] *
				x[n_stop - tau2 - lenFilled:n_stop - tau2] *
				x[n_stop - tau3 - lenFilled:n_stop - tau3] *
				x[n_stop - tau4 - lenFilled:n_stop - tau4] *
				x[n_stop - tau5 - lenFilled:n_stop - tau5]
			)
		else:
			vecX5 = (
				x[n_start - tau1:n_stop - tau1] *
				x[n_start - tau2:n_stop - tau2] *
				x[n_start - tau3:n_stop - tau3] *
				x[n_start - tau4:n_stop - tau4] *
				x[n_start - tau5:n_stop - tau5]
			)

		expValX5[tau1, tau2, tau3, tau4, tau5] = np.mean(vecX5 ** 2)
		A5eff[tau1, tau2, tau3, tau4, tau5] = np.power(
			expValX5[tau1, tau2, tau3, tau4, tau5] /
			calcDiagonalFactor((tau1, tau2, tau3, tau4, tau5)), 1/5
		)

		vecY = y[n_start:n_stop]
		k_5[tau1, tau2, tau3, tau4, tau5] = (
			np.sum(vecX5 * vecY) / (n_stop - n_start) -
			math.factorial(3) * A5eff[tau1, tau2, tau3, tau4, tau5]**4 * (
				k3[tau1, tau2, tau3] * delta(tau4 - tau5) +
				k3[tau1, tau2, tau4] * delta(tau3 - tau5) +
				k3[tau1, tau2, tau5] * delta(tau3 - tau4) +
				k3[tau1, tau3, tau4] * delta(tau2 - tau5) +
				k3[tau1, tau3, tau5] * delta(tau2 - tau4) +
				k3[tau1, tau4, tau5] * delta(tau2 - tau3) +
				k3[tau2, tau3, tau4] * delta(tau1 - tau5) +
				k3[tau2, tau3, tau5] * delta(tau1 - tau4) +
				k3[tau2, tau4, tau5] * delta(tau1 - tau3) +
				k3[tau3, tau4, tau5] * delta(tau1 - tau2)
			) -
			A5eff[tau1, tau2, tau3, tau4, tau5]**3 * (
				k1[tau1] * delta4(tau2, tau3, tau4, tau5) +
				k1[tau2] * delta4(tau1, tau3, tau4, tau5) +
				k1[tau3] * delta4(tau1, tau2, tau4, tau5) +
				k1[tau4] * delta4(tau1, tau2, tau3, tau5) +
				k1[tau5] * delta4(tau1, tau2, tau3, tau4)
			)
		) / math.factorial(5) / A5eff[tau1, tau2, tau3, tau4, tau5]**5

		perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for (t1, t2, t3, t4, t5) in perms:
			k_5[t1, t2, t3, t4, t5] = k_5[tau1, tau2, tau3, tau4, tau5]

	return k_5


def calcWienerMultivar(x, y, length1, length2, length3, length4, length5):
	# prepare input/output arrays measured at different variances
	order = 5
	# x=np.array([x_0,x_1,x_2,x_3])
	# y=np.array([y_0,y_1,y_2,y_3])

	A = np.var(x, axis=1)

	k0 = np.ndarray(shape=(order+1, 1))
	k1 = np.ndarray(shape=(order+1, length1))
	k2 = np.ndarray(shape=(order+1, length2, length2))
	k3 = np.ndarray(shape=(order+1, length3, length3, length3))
	k4 = np.ndarray(shape=(order+1, length4, length4, length4, length4))
	k5 = np.ndarray(shape=(order+1, length5, length5,
					length5, length5, length5))

	print('k0')
	k0[0] = multivarK0(y[0])

	print('k1')
	k1[1] = multivarK1(x[1], y[1], length1, A[1])

	print('k2')
	print(A, A[2])
	k0[2] = multivarK0(y[2])
	k2[2] = multivarK2(x[2], y[2], length2, A[2], k0[2])

	print('k3')
	k1[3] = multivarK1(x[3], y[3], length1, A[3])
	k3[3] = multivarK3(x[3], y[3], length3, A[3], k1[3])

##
	print('k4')
	print(A, A[4])
	k2[4] = multivarK2(x[4], y[4], length2, A[4], k0[4])
	k4[4] = multivarK4(x[4], y[4], length4, A[4], k2[4])

	print('k5')
	k1[5] = multivarK1(x[5], y[5], length1, A[5])
	k3[5] = multivarK3(x[5], y[5], length3, A[5], k1[5])
	k5[5] = multivarK5(x[5], y[5], length5, A[5], k1[5], k3[5])

	k4 = np.zeros(shape=(order+1, length4, length4, length4, length4))
	k5 = np.zeros(shape=(order+1, length5, length5,
						  length5, length5, length5))
##
	return (k0, k1, k2, k3, k4, k5, A)


def wiener2VolterraMultivar(k0, k1, k2, k3, k4, k5, A):

	h0 = np.copy(k0[0])[0]
	for tau1 in range(len(k2[2])):
		h0 -= A[0] * k2[2][tau1, tau1]
	for tau1 in range(len(k4[4])):
		for tau2 in range(len(k4[4])):
			h0 += 3 * A[0]**2 * k4[4][tau1, tau1, tau2, tau2]

	h1 = np.copy(k1[1])
	for tau1 in range(len(k3[3])):
		for tau2 in range(len(k3[3])):
			h1[tau1] -= 3 * A[1] * k3[3][tau1, tau2, tau2]
	for tau1 in range(len(k5[5])):
		for tau2 in range(len(k5[5])):
			for tau3 in range(len(k5[5])):
				h1[tau1] += 15 * A[1]**2 * k5[5][tau1, tau2, tau2, tau3, tau3]

	h2 = np.copy(k2[2])
	for tau1 in range(len(k4[4])):
		for tau2 in range(len(k4[4])):
			for tau3 in range(len(k4[4])):
				h2[tau1, tau2] -= 6 * A[2] * k4[4][tau1, tau2, tau3, tau3]

	h3 = np.copy(k3[3])
	for tau1 in range(len(k5[5])):
		for tau2 in range(len(k5[5])):
			for tau3 in range(len(k5[5])):
				for tau4 in range(len(k5[5])):
					h3[tau1, tau2, tau3] -= 10 * A[3] * \
						k5[5][tau1, tau2, tau3, tau4, tau4]

	h4 = np.copy(k4[4])
	h5 = np.copy(k5[5])

######### old #########
	# h0 = np.copy(k0[0])[0]
	# for tau1 in np.arange(len(k2)):
	#	 h0 -= A[0]*k2[2][tau1, tau1]

	# h1 = np.copy(k1[1])
	# for tau1 in np.arange(len(k3)):
	#	 for tau2 in np.arange(len(k3)):
	#		 h1[tau1] -= 3*A[1]*k3[3][tau1, tau2, tau2]

	# h2 = np.copy(k2[2])
	# h3 = np.copy(k3[3])
 ######################
	volterra_sys = VolterraResponse(use_opencl=True)
	volterra_sys.set_kernels(h0, h1, h2, h3, h4, h5)
	return (volterra_sys)


def leeSchetzenMultivar(x, y, length1, length2, length3, length4, length5, nr_of_workers=2):
	for i in np.arange(6):
		x[i] -= np.mean(x[i])
		y[i] -= np.mean(y[i])

	# x=np.array([x,x,x,x])
	# y=np.array([y,y,y,y])

	# Kernel calculation
	print('Starting Multi-Variance Wiener Kernel estimation:')
	# print('Processors used: '+str(nr_of_workers)+' out of '+str(cpu_count()))

	# Calculate Wiener Kernels and Gx-------------------------------------------
	k0, k1, k2, k3, k4, k5, A = calcWienerMultivar(
		x, y, length1, length2, length3, length4, length5)

	print('Wiener-Volterra Conversion:')
	# Translate from Wiener to Volterra by 'Wiener-to-Volterra formulas'----------------------------
	volterra_sys = wiener2VolterraMultivar(k0, k1, k2, k3, k4, k5, A)

	print('Finished Volterra Kernel estimation')
	print('')
	return (volterra_sys)
# ----------------------------------------------------


def identifyKernelsMultivar(xm, y, length1, length2, length3, length4, length5, nr_of_workers, iterations=5, alpha1=0.8, AntiAliasing=False, write=True, name=''):
	seconds1 = time.time()

	# for i in np.arange(6):#Optional: reduce 2-pt correlation further

	print('Starting Multiple-Variance kernel identification...')
	volterra_sys = leeSchetzenMultivar(
		xm, y, length1, length2, length3, length4, length5, nr_of_workers=nr_of_workers)

	# Off-Diagonal Truncation
	if (0):
		volterra_sys = offDiagonalTruncation(volterra_sys, distance=5)

	# Writing output
	if (write):
		print('Writing kernel files h'+name+':')
		print('')
		writeKernels(volterra_sys, name=name)

	seconds3 = time.time()
	print('Done identifying kernels. Time passed: ' +
		  '{:.1f}'.format(seconds3-seconds1)+'s')
	return (volterra_sys)


# -------------------------------------------------------------------------------------------


if __name__ == '__main__':
	print('')
	print('')
	print('')
	print('***************Volterra_Kernels***************')
	seconds0 = time.time()
	print('Beginning: ' + time.ctime(seconds0))
	print('Preparing input data...')

	np.set_printoptions(precision=4)
	nr_of_workers = 2
	# nr_of_workers=int(cpu_count()/2+1)
	wdin = 'Volterra_Input_Data'
	wdout = 'Volterra_Kernels'

	# Volterra specifications
	# size of kernels of dimension n (=1,2,3...)
	length1 = 80  # 120
	length2 = 25
	length3 = 25
	length4 = 10
	length5 = 9

	# System Parameters
	inputBandwidth = 3.333333e9  # Equals half the sample rate
	validBandwidth = 2.6e9  # Equals the AA-filter cutoff
	order = 3

	# Skew output channel this much
	skew = 0
	AntiAliasing = False

	# Input Data
	datastart = 0  # Start index of the data to be used
	datalength = 20000000  # Maximum length of the data to be used.
	times = 25e-12 * np.arange(0, datalength)
	noiseScale = 0.0

	# Hardware setup
	# RefAttenuations
	# 8.25 for ADA4960
	# 8.25 for ADL5565
	# 0 for LMH5401
	# 0 for LMH3401
	# dB, attenuation inequality between DUT and reference channel (aRef-aDUT), for characterization of ONLY the DUT
	refAttenuation = 0
	refAttenuationY = 0  # dB, attenuation inequality between measured x and DUT input x for characterization of the noise source and DUT
	refFactor = 10**(-refAttenuation/20)
	refFactorY = 10**(-refAttenuationY/20)


# Input signal preparation-------------------------------------------------------------------------------
	# np.random.seed(3495873)
	# Data format: 1st row=description, e.g. 'time,v_channel1,v_channel2'
	# Row(n) content: time(n-1), voltage(n-1),
	if (1):  # Read input/Output data from csv

		if (1):  # input data
			filename = 'bat15_2l'
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

			filename2 = 'bat15_2m'
			if os.name == 'nt':  # Windows
				print('File: '+filename2+'.csv')
				df = pd.read_csv(wdin+'/'+filename2+'.csv', sep=',',
								 header=None, dtype=str)  # Oscilloscope Data
			else:  # Other OS (Linux, MacOS)
				absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filename2}.csv'
				print('File: ' + absolute_path)
				df = pd.read_csv(absolute_path, sep=',',
								 header=None, dtype=str)

			# normal file
			times2 = np.transpose(df.values[0:])[0].astype(float)
			x2 = np.transpose(df.values[0:])[1].astype(
				float)  # Start with 0th row
			y2 = np.transpose(df.values[0:])[2].astype(
				float)  # Start with 0th row
			# decorrelated file
			# times		= np.transpose(df.values[0:])[0].astype(float)
			# xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
			# x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
			# y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

			# Check if data is too long or to short
			if (len(times2) >= datastart+datalength):
				times2 = times2[datastart:datastart+datalength]
				x2 = x2[datastart:datastart+datalength]
				y2 = y2[datastart:datastart+datalength]
			datalength = len(times2)

			filename3 = 'bat15_2h'
			if os.name == 'nt':  # Windows
				print('File: '+filename3+'.csv')
				df = pd.read_csv(wdin+'/'+filename3+'.csv', sep=',',
								 header=None, dtype=str)  # Oscilloscope Data
			else:  # Other OS (Linux, MacOS)
				absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filename3}.csv'
				print('File: ' + absolute_path)
				df = pd.read_csv(absolute_path, sep=',',
								 header=None, dtype=str)

			# normal file
			times3 = np.transpose(df.values[0:])[0].astype(float)
			x3 = np.transpose(df.values[0:])[1].astype(
				float)  # Start with 0th row
			y3 = np.transpose(df.values[0:])[2].astype(
				float)  # Start with 0th row
			# decorrelated file
			# times		= np.transpose(df.values[0:])[0].astype(float)
			# xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
			# x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
			# y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

			# Check if data is too long or to short
			if (len(times3) >= datastart+datalength):
				times3 = times3[datastart:datastart+datalength]
				x3 = x3[datastart:datastart+datalength]
				y3 = y3[datastart:datastart+datalength]
			datalength = len(times3)
			
			# #Not yet used--------------------------------
			# filenameFrontendNoise = 'noiseempty'
			# if os.name == 'nt':  # Windows
				# print('File: '+filenameFrontendNoise+'.csv')
				# df = pd.read_csv(wdin+'/'+filenameFrontendNoise+'.csv', sep=',',
								 # header=None, dtype=str)  # Oscilloscope Data
			# else:  # Other OS (Linux, MacOS)
				# absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filenameFrontendNoise}.csv'
				# print('File: ' + absolute_path)
				# df = pd.read_csv(absolute_path, sep=',',
								 # header=None, dtype=str)

			# # normal file
			# #Frontend Noise
			# timesFNoise = np.transpose(df.values[0:])[0].astype(float)
			# xFNoise = np.transpose(df.values[0:])[1].astype(
				# float)  # Start with 0th row
			# yFNoise = np.transpose(df.values[0:])[2].astype(
				# float)  # Start with 0th row
			# # decorrelated file
			# # times		= np.transpose(df.values[0:])[0].astype(float)
			# # xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
			# # x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
			# # y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

			# # Check if data is too long or to short
			# if (len(timesFNoise) >= datastart+datalength):
				# timesFNoise = timesFNoise[datastart:datastart+datalength]
				# xFNoise = xFNoise[datastart:datastart+datalength]
				# yFNoise = yFNoise[datastart:datastart+datalength]
			# datalength = len(times3)
			# #--------------------------------------
			
			x = np.nan_to_num(x)
			print(np.var(x))
			xMultivar = np.array([x, x, x2, x3, x3, x3])
			yMultivar = np.array([y, y, y2, y3, y3, y3])
			timesMultivar = np.array(
				[times, times, times2, times2, times3, times3])

	signalScale = 0.5

	Gain = 1  # Gain of the amplifier
	if (0):  # Generate simulated input data

		if (1):
			x = np.random.normal(loc=0.0, scale=signalScale, size=datalength)
			x -= np.mean(x)

		if (0):

			# filter=np.array([0.1, 0,0,0, 0.2, 0,0,0, 0.2, 0,0,0, 0.3, 0,0,0, 0, 0,0,0, 0, 0,0,0, 0])
			filter = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							  0, 0.3, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0.1])
			# Calculate filtered output
			x = np.convolve(x, filter, mode='same')

		if (0):  # Nonideal (distorted) input BEFORE linear system
			a = np.array([0,  1,  0,  -0.05])
			xoriginal = np.copy(x)
			x = a[3]*x**3 + a[2]*x**2 + a[1]*x + a[0]

		if (0):  # Nonideal (Autocorrellated) input
			timestep = times[1]-times[0]
			tau = 1200e-12
			z = np.zeros(len(x))
			for n in np.arange(len(x)):
				z[n] = z[n-1]+timestep/tau*(x[n]-z[n-1])
			x = np.copy(z)

		if (0):  # Nonideal (distorted) input AFTER linear system
			a = np.array([0,  1,  0,  -0.2])
			xoriginal = np.copy(x)
			x = a[3]*x**3 + a[2]*x**2 + a[1]*x + a[0]

		# Generate output data before downsampling
		if (1):  # Apply model Function to y
			# y=systemFunction(Gain*x, bw=1.5e9, sr=0.5e9, dt=times[1]-times[0], oversampleFactor=10, asymmetry=0.2)

			xl, _ = LPfilter(xin=x, times=times, fstop=0.9*inputBandwidth)
			y = systemFunction(Gain*xl, bw=2.3e9, sr=2e9,
							   dt=times[1]-times[0], oversampleFactor=10, asymmetry=-0.1)  # ADA4960
			# y,_,_ = boostConverter(xl, dt=times[1]-times[0], oversampleFactor=10)#Boost
			# y=y-24


# -----------------------------------------------------------------------------------------------------------

	# Downsampling of the input and output signals according to 'Volterra Kernel interpolation' by Köppl:
	if (1):
		for i in np.arange(6):#Do it for all the signal files-----
			tempTimes, tempX, tempY = koepplDownsampling(datalength, timesMultivar[i], xMultivar[i], yMultivar[i], order, inputBandwidth)
			dl = len(tempX)
			tempTimes = tempTimes[:dl]
			timesMultivar[i, 0:dl], xMultivar[i,0:dl], yMultivar[i, 0:dl] = tempTimes, tempX, tempY
		timesMultivar = timesMultivar[:, :dl]
		xMultivar = xMultivar[:, :dl]
		yMultivar = yMultivar[:, :dl]
		#-----
		
		# #Do it for the frontend noise file---
		# tempTimes, tempX, tempY = koepplDownsampling(datalength, timesFNoise, xFNoise, yFNoise, order, inputBandwidth)
		# dl = len(tempX)
		# tempTimes = tempTimes[:dl]
		# timesFNoise[0:dl], xFNoise[0:dl], yFNoise[0:dl] = tempTimes, tempX, tempY
		# timesFNoise = timesFNoise[:dl]
		# xFNoise = xMultivar[:dl]
		# yFNoise = yMultivar[:dl]
		# #---
		
		datalength = dl


# #Show the alignment between x and y, to find the correct shift
	# plt.plot(calc_cyx(1000, 10000, 2000,
			 # xMultivar[0, :-1000], xMultivar[0, 1000:]))
	# plt.show()
	
	
#Insert the frontend noise correction here#----
#empty
#---------------------------
	if(0):
		for i in np.arange(6):
			frontendKernel,invFrontendKernel=findFrontendNoiseKernel(xMultivar[i,:], xFNoise, length1)
			xMultivar[i,:]=np.convolve(xMultivar[i,:], frontendKernel, mode='same')

# -------------------------------------------------

	for i in np.arange(6):
		xMultivar[i] -= np.mean(xMultivar[i])
		yMultivar[i] -= np.mean(yMultivar[i])
		#xFNoise-=np.mean(xFNoise)

	# Calculate variances of resulting signals
	Az = np.var(xMultivar, axis=1)
	print(Az)
	varXoriginal = np.copy(Az)
	
	
	# HF Noise Injection - fills out the spectrum gap at the end of LPF bandwidth

	for i in np.arange(6):
		np.random.seed(123)
		HFnoiseX, _ = HPfilter(x=np.random.normal(loc=0.0, scale=np.sqrt(Az[i]), size=datalength), times=timesMultivar[i], fstop=validBandwidth)			

		if(1):#Use a minimum-phase DUT filter, instead of non-causal/linear phase. 
		
			#Calculate Covariances
			cxx=calc_cxx(0, datalength, length1, xMultivar[i,:])[:length1]
			cxxhf=calc_cxx(0, datalength, length1, HFnoiseX)[:length1]
			
			cxx=np.roll(np.concatenate((cxx, cxx[::-1][:-1])), length1-1)
			cxxhf=np.roll(np.concatenate((cxxhf, cxxhf[::-1][:-1])), length1-1)
			
		
			CXX=np.fft.fft(cxx)
			CXXHF=np.fft.fft(cxxhf)
			
			#Calculate transfer function of the resulting filter
			FILT=CXX/(CXX+CXXHF)
			
			filt=np.real(np.roll(np.fft.ifft(FILT), length1-1))
			
			#Build minimum phase version of the filter
			filt_mp=sg.minimum_phase(filt, method='homomorphic', n_fft=3000)
			filt_mp=np.concatenate((np.zeros(length1-1), filt_mp))
			
			filt_mp=np.convolve(filt_mp, filt_mp, mode='same')
			
			#plt.plot(filt)
			#plt.plot(filt_mp)
			#plt.plot(np.abs(np.fft.fft(filt)))
			#plt.plot(np.abs(np.fft.fft(filt_mp)))
			
			filt_mp=np.pad(filt_mp, 1000, 'constant')
			filt=np.pad(filt, 1000, 'constant')
			
			#Construct a phase filter that only contains the phase differences between minimum-phase and linear-phase filter
			PHASEFILT=np.fft.fft(filt_mp)/np.fft.fft(filt)
			
			phasefilt=np.real(np.fft.ifft(PHASEFILT))#[:length1]

			
			phasefilt=np.roll(phasefilt, 1000+length1-1)
			phasefilt=phasefilt[1000:1000+2*length1]
			
			
			#Apply phase filter to y
			yMultivar[i,:]=np.convolve(yMultivar[i,:], phasefilt, mode='same')
		
			#plt.plot(np.angle(np.fft.fft(np.roll(phasefilt, -length1-2))))
			#plt.plot(np.roll(phasefilt, -length1+2))
			#plt.show()
		xMultivar[i, :] = xMultivar[i, :]+HFnoiseX
		



	# End of input signal preparation-------------------------------------------------------------------------
	seconds1 = time.time()
	print('Done preparing data. Time passed: ' +
		  '{:.1f}'.format(seconds1-seconds0)+'s')
	print('')

# ------------------------------------------------------------------------------------------------------------
	# Decorrelate
	xModMultivar = np.zeros(shape=np.shape(xMultivar))

	# Automatic
	if (0):
		for i in np.arange(6):
			xModMultivar[i] = Decorrelation_OCL.deCorrelateFast(
				timesMultivar[i], xMultivar[i], yMultivar[i], 0.3, 0.53, 0.53, length1=61, length2=26, length3=26, write=True, name=filename[i], chunks=1024)

	# Manual selection of xModMultivar[i] from input data xMultivar[j] with variance A[j]
	if (0):
		xModMultivar[0] = Decorrelation_OCL.deCorrelateFast(timesMultivar[0], xMultivar[0], yMultivar[0], 0.5, 1, 1,
															length1=length1+1, length2=length2+1, length3=length3+1, iterations=25, write=True, name=filename, chunks=1024)
		xModMultivar[1] = np.copy(xModMultivar[0])
		xModMultivar[2] = Decorrelation_OCL.deCorrelateFast(timesMultivar[2], xMultivar[2], yMultivar[2], 0.5, 1, 1,
															length1=length1+1, length2=length2+1, length3=length3+1, iterations=25, write=True, name=filename2, chunks=1024)
		xModMultivar[4] = Decorrelation_OCL.deCorrelateFast(timesMultivar[4], xMultivar[4], yMultivar[4], 0.3, 1, 1,
															length1=length1+1, length2=length2+1, length3=length3+1, iterations=25, write=True, name=filename3, chunks=1024)
		xModMultivar[3] = np.copy(xModMultivar[2])
		xModMultivar[5] = np.copy(xModMultivar[4])

	if (0):
		xModMultivar[0] = Decorrelation_OCL.deCorrelateFast(timesMultivar[0], xMultivar[0], yMultivar[0], 0.5, 1, 1,
															length1=length1+1, length2=length2+1, length3=length3+1, iterations=25, write=True, name=filename, chunks=1024)
		xModMultivar[2] = Decorrelation_OCL.deCorrelateFast(timesMultivar[2], xMultivar[2], yMultivar[2], 0.5, 1, 1,
															length1=length1+1, length2=length2+1, length3=length3+1, iterations=25, write=True, name=filename2, chunks=1024)
		xModMultivar[3] = Decorrelation_OCL.deCorrelateFast(timesMultivar[3], xMultivar[3], yMultivar[3], 0.3, 1, 1,
															length1=length1+1, length2=length2+1, length3=length3+1, iterations=25, write=True, name=filename3, chunks=1024)
		xModMultivar[1] = np.copy(xModMultivar[0])
		xModMultivar[4] = np.copy(xModMultivar[3])
		xModMultivar[5] = np.copy(xModMultivar[3])

	if (1):
		xModMultivar[0] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[0], xMultivar[0], yMultivar[0], 0.5, 1, 1, 0.33, 0.33,
															 length1=length1+1, length2=length2+1, length3=length3+1, length4=20, length5=10, iterations=35, write=True, name=filename, chunks=1024)
		xModMultivar[2] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[2], xMultivar[2], yMultivar[2], 0.5, 1, 1, 0.33, 0.33,
															 length1=length1+1, length2=length2+1, length3=length3+1, length4=20, length5=10, iterations=35, write=True, name=filename2, chunks=1024)
		xModMultivar[3] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[3], xMultivar[3], yMultivar[3], 0.5, 1, 1, 0.33, 0.33,
															 length1=length1+1, length2=length2+1, length3=length3+1, length4=20, length5=10, iterations=35, write=True, name=filename3, chunks=1024)
		xModMultivar[1] = np.copy(xModMultivar[0])
		xModMultivar[4] = np.copy(xModMultivar[3])
		xModMultivar[5] = np.copy(xModMultivar[3])

	# Skew the data to account for differences in cable length, system delay, etc.
	if skew != 0:
		for i in np.arange(6):
			xMultivar[i, skew:] = xMultivar[i, :-skew]
			xMultivar[i, :skew] = np.zeros(skew)
			yMultivar[i, skew:] = yMultivar[i, :-skew]
			yMultivar[i, :skew] = np.zeros(skew)

	for i in np.arange(6):
		yMultivar[i, 3:] = yMultivar[i, :-3]
		yMultivar[i, :3] = np.zeros(3)

	# if skew != 0:
	#	 for i in np.arange(6):
	#		 xMultivar[i, :-skew] = xMultivar[i, skew:]
	#		 xMultivar[i, -skew:] = np.zeros(skew)
	#		 yMultivar[i, :-25] = yMultivar[i, 25:]
	#		 yMultivar[i, -25:] = np.zeros(25)

	analysisLength = 500
	cxy = calc_cyx(0, 100000, 2*analysisLength,
				   yMultivar[3, :-analysisLength], xMultivar[3, analysisLength:])[:2*analysisLength]
	print('y has '+str(np.argmax(np.abs(cxy))-analysisLength) +
		  ' samples of time shift compared to x')

	seconds2 = time.time()
	print('Done modifying data. Time passed: ' +
		  '{:.1f}'.format(seconds2-seconds1)+'s')

	plt.plot(calc_cyx(0, 100000, 2*analysisLength,
			 yMultivar[3, :-analysisLength], xMultivar[3, analysisLength:])[:2*analysisLength])

	plt.show()

# Identify the kernels--------------------------------------------------------------------

	identifyKernelsMultivar(xModMultivar, yMultivar, length1, length2, length3, length4, length5, nr_of_workers,
							iterations=5, alpha1=0.8, AntiAliasing=False, write=True, name='overall')
	identifyKernelsMultivar(xModMultivar, xMultivar/refFactorY, length1, length2, length3, length4, length5,
							nr_of_workers, iterations=5, alpha1=0.8, AntiAliasing=False, write=True, name='source')

	invertKernelFiles(nameIn='source', nameOut='invSource',
					  skew=2*skew)  # skew=2*skew
	combineKernelFiles(nameInA='invSource', nameInB='overall', nameOut='DUT')

	# plot volterra system output of the DUT system for a sin wave input
	if (0):

		f = 300e6
		A = 1
		t = np.arange(0, 10000)*150e-12
		x = A * np.sin(2 * np.pi * f * t)

		system = VolterraResponse(use_opencl=True)

		system = readKernels(name='DUT')
		system_out, p = system.Volterra_In_Out(x)
		plt.figure()
		plt.plot(x)
		plt.plot(system_out)
		plt.savefig('system_out.png')
		plt.show()


# -----------------------------------------------------------------------------------------
	seconds3 = time.time()
	print('Done calculating kernels. Calculation time: ' +
		  '{:.1f}'.format(seconds3-seconds2)+'s')
	print('')
# -----------------------------------------------------------------------------------------

	print('Overall time passed: '+'{:.1f}'.format(seconds3-seconds0)+'s')
	print("End:	  "+time.ctime(seconds3))
	print('-----------------------------------------------------------')

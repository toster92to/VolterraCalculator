import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy import signal as sg
from scipy.stats import norm
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
from opencl_util import create_context_and_queue
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
	return (delta(tau1 - tau2) * delta(tau3 - tau4) +
			delta(tau1 - tau3) * delta(tau2 - tau4) +
			delta(tau1 - tau4) * delta(tau2 - tau3))

# def delta4(tau1, tau2, tau3, tau4):
#	 return 1 if len(set([tau1, tau2, tau3, tau4])) == 1 else 0


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
################################################################################

def expValXpowN(sigma, N):
	#https://math.stackexchange.com/questions/2751719/expected-value-of-xn-for-normal-distribution
	#returns the expected value of E[X^N], where X is a Normal(0,sigma^2) distributet random variable. 
	if N%2!=0: 
		return(0)
	
	if N==0:
		return(1)
	else: 
		return((N-1)*sigma**2 * expValXpowN(sigma, N-2))

#print(expValXpowN(1, 4))


#New, more general implementation of the expected value of a product of multiple time-shifted X.
def pyExpVal(tauList, sigma):
	dictionary = dict((i, tauList.count(i)) for i in tauList)

	#print(dictionary)
	product=1
	for key in dictionary:
		product=product * expValXpowN(sigma, dictionary[key])
	return(product)
	
# test2ExpVal=pyExpVal((1,1,1,1,1,1,1,1), sigma=1)
# print('new')
# print(test2ExpVal)


###############################################################################

def calc_cxx(n_start, n_stop, length1, x):
	c_xx = np.zeros(len(x))
	for tau1 in np.arange(length1):

		if (n_start-tau1) < 0:  # Check if negative values get out of bounds
			vecX1 = np.zeros(n_stop-n_start)
			lenFilled = n_stop-n_start-tau1
			vecX1[tau1:] = x[n_stop-lenFilled:n_stop] * \
				x[n_stop-tau1-lenFilled:n_stop-tau1]
		else:
			# Create 1st order input vector x1. This is not nice, need take care at array ends
			vecX1 = x[n_start:n_stop]*x[n_start-tau1:n_stop-tau1]

		c_xx[tau1] += np.sum(vecX1)

	c_xx = c_xx/len(x)
	return c_xx


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


def koepplDownsampling(datalength, times, x, y, order, inputBandwidth):
	timesy = np.copy(times)

	print('Data length before downsampling: '+str(datalength))
	x, times = downsample(x, times, inputBandwidth)
	datalength = len(x)
	print('Data length after downsampling:  '+str(datalength))

	y, _ = downsample(y, timesy, order*inputBandwidth)
	y = y[0:order*datalength:order]
	return (times, x, y)
	
	
def findFrontendNoiseKernel(xBar, frontendNoise, length1):
	print('Calculating the frontend noise kernel...')
	#xBar2=xBar+frontendNoise

	xBar-=np.mean(xBar)
	frontendNoise-=np.mean(frontendNoise)
	#xBar2-=np.mean(xBar2)


	cxbxb=calc_cxx(0, len(xBar), length1, xBar)[:length1]
	cxbxb=np.concatenate((cxbxb[-1:0:-1],cxbxb))
	
	cn1n1=calc_cxx(0, len(xBar), length1, frontendNoise)[:length1]
	cn1n1=np.concatenate((cn1n1[-1:0:-1],cn1n1))
	
	#plt.plot(10*np.log10(np.abs(np.fft.fft(cxbxb))))
	#plt.plot(10*np.log10(np.abs(np.fft.fft(cn1n1))))
	#plt.show()
	
	frontendNoiseKernel=np.real(np.fft.ifft((np.fft.fft(cxbxb)+np.fft.fft(cn1n1))/np.fft.fft(cxbxb)))
	invFrontendNoiseKernel=np.real(np.fft.ifft(1/np.fft.fft(frontendNoiseKernel)))
	
	frontendNoiseKernel=np.roll(frontendNoiseKernel, length1-1)
	invFrontendNoiseKernel=np.roll(invFrontendNoiseKernel, length1-1)

	
	return(frontendNoiseKernel, invFrontendNoiseKernel)	


# Under construction: Multi-Variance characterization--------------------------------
ctx, queue = create_context_and_queue(device_type='GPU', profiling=True)
mf = cl.mem_flags

kernels_code = """

__kernel void compute_vecX3(
	__global float *x,
	__global float *vecX3,
	__global int *indices,
	int n_start,
	int n_stop,
	int max_tau,
	int len_x
) {
	int gid = get_global_id(0);
	if (gid >= (n_stop - n_start - max_tau)) return;

	int tau1 = indices[0];
	int tau2 = indices[1];
	int tau3 = indices[2];

	int idx1 = gid + n_start - tau1;
	int idx2 = gid + n_start - tau2;
	int idx3 = gid + n_start - tau3;

	if (idx1 < 0 || idx2 < 0 || idx3 < 0 || idx1 >= len_x || idx2 >= len_x || idx3 >= len_x) {
		vecX3[gid] = 0.0f;
	} else {
		vecX3[gid] = x[idx1] * x[idx2] * x[idx3];
	}
}

__kernel void compute_vecX4(
	__global float *x,
	__global float *vecX4,
	__global int *indices,
	int n_start,
	int n_stop,
	int max_tau,
	int len_x
) {
	int gid = get_global_id(0);
	if (gid >= (n_stop - n_start - max_tau)) return;

	int tau1 = indices[0];
	int tau2 = indices[1];
	int tau3 = indices[2];
	int tau4 = indices[3];

	int idx1 = gid + n_start - tau1;
	int idx2 = gid + n_start - tau2;
	int idx3 = gid + n_start - tau3;
	int idx4 = gid + n_start - tau4;

	if (idx1 < 0 || idx2 < 0 || idx3 < 0 || idx4 < 0 || idx1 >= len_x || idx2 >= len_x || idx3 >= len_x || idx4 >= len_x) {
		vecX4[gid] = 0.0f;
	} else {
		vecX4[gid] = x[idx1] * x[idx2] * x[idx3] * x[idx4];
	}
}

__kernel void compute_expValX4(
	__global float *vecX4,
	__global float *expValX4,
	int length
) {
	int gid = get_global_id(0);
	if (gid >= length) return;

	float sum = 0.0f;
	for (int i = 0; i < length; i++) {
		sum += vecX4[i] * vecX4[i];
	}
	expValX4[gid] = sum / length;
}

__kernel void compute_vecX5(
	__global float *x,
	__global float *vecX5,
	__global int *indices,
	int n_start,
	int n_stop,
	int max_tau,
	int len_x
) {
	int gid = get_global_id(0);
	if (gid >= (n_stop - n_start - max_tau)) return;

	int tau1 = indices[0];
	int tau2 = indices[1];
	int tau3 = indices[2];
	int tau4 = indices[3];
	int tau5 = indices[4];

	int idx1 = gid + n_start - tau1;
	int idx2 = gid + n_start - tau2;
	int idx3 = gid + n_start - tau3;
	int idx4 = gid + n_start - tau4;
	int idx5 = gid + n_start - tau5;

	if (idx1 < 0 || idx2 < 0 || idx3 < 0 || idx4 < 0 || idx5 < 0 || idx1 >= len_x || idx2 >= len_x || idx3 >= len_x || idx4 >= len_x || idx5 >= len_x) {
		vecX5[gid] = 0.0f;
	} else {
		vecX5[gid] = x[idx1] * x[idx2] * x[idx3] * x[idx4] * x[idx5];
	}
}


"""

prg = cl.Program(ctx, kernels_code).build()


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

######## Our method ##############

def multivarK4(x, y, length4, A, k0, k2):
	n_start = length4
	n_stop = len(x)
	k_4 = np.zeros(shape=(length4, length4, length4, length4))
	k_4out = np.zeros(shape=(length4, length4, length4, length4))

	expValX4 = np.zeros(shape=(length4, length4, length4, length4))
	A4eff = np.zeros(shape=(length4, length4, length4, length4))

	for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):
		if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0):
			vecX4 = np.zeros(n_stop - n_start)
			lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4)
			vecX4[max(tau1, tau2, tau3, tau4):] = (
				x[n_stop - tau1 - lenFilled:n_stop - tau1] * 
				x[n_stop - tau2 - lenFilled:n_stop - tau2] * 
				x[n_stop - tau3 - lenFilled:n_stop - tau3] * 
				x[n_stop - tau4 - lenFilled:n_stop - tau4]
			)
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
		k_4[tau1, tau2, tau3, tau4] = np.sum(vecX4 * vecY) / (n_stop - n_start) 

		k_4[tau1, tau2, tau3, tau4] -= k0 * pyExpVal([tau1, tau2, tau3, tau4], np.sqrt(A4eff[tau1, tau2, tau3, tau4]))

		for [v1,v2] in list(combinations_with_replacement(range(length4), 2)):
			perm_factor = 2 if v1 != v2 else 1

			k_4[tau1, tau2, tau3, tau4] -= perm_factor * k2[v1, v2] * pyExpVal([v1, v2, tau1, tau2, tau3, tau4], np.sqrt(A4eff[tau1, tau2, tau3, tau4])) 

		for [v1] in list(combinations_with_replacement(range(length4), 1)):   
			k_4[tau1, tau2, tau3, tau4] += A4eff[tau1, tau2, tau3, tau4] * k2[v1, v1] * pyExpVal([tau1, tau2, tau3, tau4], np.sqrt(A4eff[tau1, tau2, tau3, tau4])) 
		
		k_4out[tau1, tau2, tau3, tau4] = k_4[tau1, tau2, tau3, tau4] / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4

		perms = list(set(permutations([tau1, tau2, tau3, tau4])))
		for (t1, t2, t3, t4) in perms:
			k_4out[t1, t2, t3, t4] = k_4out[tau1, tau2, tau3, tau4]


	return k_4out


def multivarK5(x, y, length5, A, k1, k3):
	n_start = length5
	n_stop = len(x)
	k_5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	k_5out = np.zeros(shape=(length5, length5, length5, length5, length5))
	
	#print(np.max(np.abs(k3)))
	
	expValX5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	A5eff = np.zeros(shape=(length5, length5, length5, length5, length5))

	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(range(length5), 5)):

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
		#A5eff[tau1, tau2, tau3, tau4, tau5]=A
		

		#vecY = y[n_start:n_stop]
		vecY = y[n_start:n_stop]
		k_5[tau1, tau2, tau3, tau4, tau5] = np.sum(vecX5 * vecY) / (n_stop - n_start)
		#k_5[tau1, tau2, tau3, tau4, tau5] = pyExpVal([0, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5]))
		#print([tau1, tau2, tau3, tau4, tau5])
		#print(pyExpVal([0, tau1, tau2, tau3, tau4, tau5],1))
		#print((pyExpVal([0, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5])) - k_5[tau1, tau2, tau3, tau4, tau5]) / pyExpVal([0, 0, 1, 1, 2, 2], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5])))
		
		

		for [v1] in list(combinations_with_replacement(range(length5), 1)):
			
			k_5[tau1, tau2, tau3, tau4, tau5] -= k1[v1] * pyExpVal([v1, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5]))


		for [v1,v2,v3] in list(combinations_with_replacement(range(length5), 3)):
			perm_factor = 6 if len(set([v1, v2, v3])) == 3 else 3 if len(set([v1, v2, v3])) == 2 else 1

			k_5[tau1, tau2, tau3, tau4, tau5] -= perm_factor * k3[v1, v2, v3] * pyExpVal([v1, v2, v3, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5]))
				 
		for [v1, v2] in list(combinations_with_replacement(range(length5), 2)):
			perm_factor = 2 if v1 != v2 else 1

			k_5[tau1, tau2, tau3, tau4, tau5] += perm_factor * 3 * A5eff[tau1, tau2, tau3, tau4, tau5] *k3[v1, v2, v2] * pyExpVal([v1, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5]))

		k_5out[tau1, tau2, tau3, tau4, tau5] = k_5[tau1, tau2, tau3, tau4, tau5] / math.factorial(5) / A5eff[tau1, tau2, tau3, tau4, tau5] ** 5  

		perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for (t1, t2, t3, t4, t5) in perms:
			k_5out[t1, t2, t3, t4, t5] = k_5out[tau1, tau2, tau3, tau4, tau5]
		
		#print (k_5 - expVal)
		#print([tau1, tau2, tau3, tau4, tau5])
		#print(pyExpVal([0, tau1, tau2, tau3, tau4, tau5], 1))
		#print(k_5[tau1, tau2, tau3, tau4, tau5] / pyExpVal([0, 0, 1, 1, 2, 2], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5])))
	print(np.max(np.abs(k_5out)))
	print(np.sqrt(np.var(k_5out)))
	#print(k_5out)
	
	#print the indices of the 100 highest kernel entries
	# ind=np.zeros((100,5), dtype=int)#number of desired max indices, number of indexing positions
	# for i in np.arange(100):
		# ind[i,:] = np.unravel_index(np.argmax(np.abs(k_5out), axis=None), k_5out.shape)
		# k_5out[ind[i,0], ind[i,1], ind[i,2], ind[i,3], ind[i,4]]=0
	# print(ind)


	return k_5out
		

#--------------------------------------------------- Test: Calculate G_p
def calcG0(x, k0):
	return k0


def calcG1(x, k1):	
	G1 = VolterraResponse(use_opencl=False, kernelName="G1", kernelPath="", profiling_data=False)
	
	length2=1
	length3=1
	length4=1
	length5=1
	
	Gk0 = 0
	#Gk1 = np.zeros(length1)
	Gk2 = np.zeros(shape=(length2, length2))
	Gk3 = np.zeros(shape=(length3, length3, length3))
	Gk4 = np.zeros(shape=(length4, length4, length4, length4))
	Gk5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	Gk1 = k1
	
	G1.set_kernels(Gk0,Gk1,Gk2,Gk3,Gk4,Gk5)
	y=G1.H1(x, oversampleFactor=1, neg_len=0)
	return y

	
def calcG2(x, k2):
	A=np.var(x)
	
	length1=1
	length2=len(k2)
	length3=1
	length4=1
	length5=1
	
	G2 = VolterraResponse(use_opencl=False, kernelName="G2", kernelPath="", profiling_data=False)
	
	#Gk0 = 0
	Gk1 = np.zeros(length1)
	#Gk2 = np.zeros(shape=(length2, length2))
	Gk3 = np.zeros(shape=(length3, length3, length3))
	Gk4 = np.zeros(shape=(length4, length4, length4, length4))
	Gk5 = np.zeros(shape=(length5, length5, length5, length5, length5))

	Gk2 = k2
	
	Gk0 = 0
	for tau1 in np.arange(length2):
		Gk0+=k2[tau1,tau1]
	
	G2.set_kernels(Gk0,Gk1,Gk2,Gk3,Gk4,Gk5)
	y=G2.H2(x, oversampleFactor=1, neg_len=0) - A*G2.H0(x, oversampleFactor=1)
	return y

	
def calcG3(x, k3):
	A=np.var(x)
	
	length3=len(k3)
	length2=1
	length4=1
	length5=1
	
	G3 = VolterraResponse(use_opencl=False, kernelName="G3", kernelPath="", profiling_data=False)
	
	Gk0 = 0
	#Gk1 = np.zeros(length1)
	Gk2 = np.zeros(shape=(length2, length2))
	#Gk3 = np.zeros(shape=(length3, length3, length3))
	Gk4 = np.zeros(shape=(length4, length4, length4, length4))
	Gk5 = np.zeros(shape=(length5, length5, length5, length5, length5))

	Gk3 = k3
	
	Gk1 = np.zeros(length3)
	for tau1 in np.arange(length3):
		for tau2 in np.arange(length3):
			Gk1[tau1]+=k3[tau1,tau2,tau2]
	
	G3.set_kernels(Gk0,Gk1,Gk2,Gk3,Gk4,Gk5)
	y=G3.H3(x, oversampleFactor=1, neg_len=0) - 3*A*G3.H1(x, oversampleFactor=1, neg_len=0)
	return y
	
	
def calcG4(x, k4):
	A=np.var(x)
	length1=1
	length3=1
	length5=1
	
	length4=len(k4)
	
	G4 = VolterraResponse(use_opencl=False, kernelName="G4", kernelPath="", profiling_data=False)
	
	#Gk0 = 0
	Gk1 = np.zeros(length1)
	#Gk2 = np.zeros(shape=(length2, length2))
	Gk3 = np.zeros(shape=(length3, length3, length3))
	#Gk4 = np.zeros(shape=(length4, length4, length4, length4))
	Gk5 = np.zeros(shape=(length5, length5, length5, length5, length5))

	Gk4 = k4
	
	Gk2 = np.zeros(shape=(length4, length4))
	for tau1 in np.arange(length4):
		for tau2 in np.arange(length4):
			for tau3 in np.arange(length4):
				Gk2[tau1,tau2]+=k4[tau1,tau2,tau3,tau3]
				
	Gk0 = 0
	for tau1 in np.arange(length4):
		for tau2 in np.arange(length4):
			Gk0 += k4[tau1,tau1,tau2,tau2]
	
	
	G4.set_kernels(Gk0,Gk1,Gk2,Gk3,Gk4,Gk5)
	y=G4.H4(x, oversampleFactor=1, neg_len=0) - 6*A*G4.H2(x, oversampleFactor=1, neg_len=0) + 3*A**2*G4.H0(x, oversampleFactor=1)
	return y


def multivarK0G(y):
	return np.mean(y)


def multivarK1G(x, y, length1, A, k0):
	n_start = length1
	n_stop = len(x)

	k_1 = np.zeros(shape=(length1))
	k_1out = np.zeros(shape=(length1))
	
	vecG = calcG0(x, k0)
	
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
		
		k_1[tau1] = np.sum(vecX1*(vecY-vecG)) / (n_stop-n_start)
		
		k_1out[tau1] = k_1[tau1] / math.factorial(1) / expValX1[tau1]
		
	return k_1out


def multivarK2G(x, y, length2, A, k0, k1):
	n_start = length2
	n_stop = len(x)
	k_2 = np.zeros(shape=(length2, length2))
	k_2out = np.zeros(shape=(length2, length2))

	# Expected value of (x(n-tau1)x(n-tau2))^2
	expValX2 = np.zeros(shape=(length2, length2))
	# Effective signal variance at [tau1, tau2] computed from expValX2
	A2eff = np.zeros(shape=(length2, length2))
	
	vecG = (calcG1(x, k1) + calcG0(x, k0))[n_start:n_stop]
	vecY = y[n_start:n_stop]	
			
	# print(np.var(vecY))
	# print(np.var(vecY-vecG))


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


		k_2[tau1, tau2] = np.sum(vecX2*(vecY - vecG)) / (n_stop-n_start)
		
		k_2out[tau1,tau2] = k_2[tau1,tau2] / math.factorial(2) / A2eff[tau1, tau2]**2

		# set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
		perm = list(set(permutations([tau1, tau2])))
		for [t1, t2] in perm:
			# Fill eqivalent kernel positions with copied entries
			k_2out[t1, t2] = k_2out[tau1, tau2]

	return k_2out


def multivarK3G(x, y, length3, A, k0, k1, k2):
	n_start = length3
	n_stop = len(x)
	k_3 = np.zeros(shape=(length3, length3, length3))
	k_3out = np.zeros(shape=(length3, length3, length3))
	# Expected value of (x(n-tau1)x(n-tau2)x(n-tau3))^2
	expValX3 = np.zeros(shape=(length3, length3, length3))
	# Effective signal variance at [tau1, tau2, tau3] computed from expValX3
	A3eff = np.zeros(shape=(length3, length3, length3))
	
	vecG = (calcG2(x, k2) + calcG1(x, k1) + calcG0(x, k0))[n_start:n_stop]
	vecY = y[n_start:n_stop]
	
	#print(np.var(vecY))
	#print(np.var(vecY-vecG))

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

		

		k_3[tau1, tau2, tau3] = np.sum(vecX3*(vecY - vecG)) / (n_stop-n_start)

		k_3out[tau1,tau2,tau3] = k_3[tau1,tau2,tau3] / math.factorial(3) / A3eff[tau1, tau2, tau3]**3

		# set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
		perm = list(set(permutations([tau1, tau2, tau3])))
		for [t1, t2, t3] in perm:
			# Fill eqivalent kernel positions with copied entries
			k_3out[t1, t2, t3] = k_3out[tau1, tau2, tau3]

	return k_3out

######## Our method ##############

def multivarK4G(x, y, length4, A, k0, k1, k2, k3):
	n_start = length4
	n_stop = len(x)
	k_4 = np.zeros(shape=(length4, length4, length4, length4))
	k_4out = np.zeros(shape=(length4, length4, length4, length4))

	expValX4 = np.zeros(shape=(length4, length4, length4, length4))
	A4eff = np.zeros(shape=(length4, length4, length4, length4))
	
	vecG = (calcG3(x, k3) + calcG2(x, k2) + calcG1(x, k1) + calcG0(x, k0))[n_start:n_stop]
	vecY = y[n_start:n_stop]

	for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):
		if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0):
			vecX4 = np.zeros(n_stop - n_start)
			lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4)
			vecX4[max(tau1, tau2, tau3, tau4):] = (
				x[n_stop - tau1 - lenFilled:n_stop - tau1] * 
				x[n_stop - tau2 - lenFilled:n_stop - tau2] * 
				x[n_stop - tau3 - lenFilled:n_stop - tau3] * 
				x[n_stop - tau4 - lenFilled:n_stop - tau4]
			)
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

		
		k_4[tau1, tau2, tau3, tau4] = np.sum(vecX4 * (vecY - vecG)) / (n_stop - n_start) 


		
		k_4out[tau1, tau2, tau3, tau4] = k_4[tau1, tau2, tau3, tau4] / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4

		perms = list(set(permutations([tau1, tau2, tau3, tau4])))
		for (t1, t2, t3, t4) in perms:
			k_4out[t1, t2, t3, t4] = k_4out[tau1, tau2, tau3, tau4]

	return k_4out


def multivarK5G(x, y, length5, A, k0, k1, k2, k3, k4):
	n_start = length5
	n_stop = len(x)
	k_5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	k_5out = np.zeros(shape=(length5, length5, length5, length5, length5))
	
	#print(np.max(np.abs(k3)))
	
	expValX5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	A5eff = np.zeros(shape=(length5, length5, length5, length5, length5))
	
	vecG = (calcG4(x, k4) + calcG3(x, k3) + calcG2(x, k2) + calcG1(x, k1) + calcG0(x, k0))[n_start:n_stop]
	vecY = y[n_start:n_stop]

	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(range(length5), 5)):

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


		k_5[tau1, tau2, tau3, tau4, tau5] = np.sum(vecX5 * (vecY - vecG)) / (n_stop - n_start)
		
		k_5out[tau1, tau2, tau3, tau4, tau5] = k_5[tau1, tau2, tau3, tau4, tau5] / math.factorial(5) / A5eff[tau1, tau2, tau3, tau4, tau5] ** 5  

		perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for (t1, t2, t3, t4, t5) in perms:
			k_5out[t1, t2, t3, t4, t5] = k_5out[tau1, tau2, tau3, tau4, tau5]

	#print(np.max(np.abs(k_5out)))
	#print(np.sqrt(np.var(k_5out)))

	return k_5out


def calcWienerMultivarG(x, y, length1, length2, length3, length4, length5):
	print('...with explicit calculation of Gpx(n)')
	# prepare input/output arrays measured at different variances
	order = 5
	# x=np.array([x_0,x_1,x_2,x_3])
	# y=np.array([y_0,y_1,y_2,y_3])

	A = np.var(x, axis=1)
	
	print('ym variances:')
	print(np.var(y, axis=1))

	k0 = np.ndarray(shape=(order+1, 1))
	k1 = np.ndarray(shape=(order+1, length1))
	k2 = np.ndarray(shape=(order+1, length2, length2))
	k3 = np.ndarray(shape=(order+1, length3, length3, length3))
	k4 = np.ndarray(shape=(order+1, length4, length4, length4, length4))
	k5 = np.ndarray(shape=(order+1, length5, length5,
					length5, length5, length5))

	print('k0')
	k0[0] = multivarK0G(y[0])

	print('k1')
	k0[1] = multivarK0G(y[1])
	k1[1] = multivarK1G(x[1], y[1], length1, A[1], k0[1])

	print('k2')
	#print(A, A[2])
	k0[2] = multivarK0G(y[2])
	k1[2] = multivarK1G(x[2], y[2], length1, A[2], k0[2])
	k2[2] = multivarK2G(x[2], y[2], length2, A[2], k0[2], k1[2])
	#print(k2[2])

	print('k3')
	k0[3] = multivarK0G(y[3])
	k1[3] = multivarK1G(x[3], y[3], length1, A[3], k0[3])
	k2[3] = multivarK2G(x[3], y[3], length2, A[3], k0[3], k1[3])
	k3[3] = multivarK3G(x[3], y[3], length3, A[3], k0[3], k1[3], k2[3])

##
	print('k4')
	#print(A, A[4])
	k0[4] = multivarK0G(y[4])
	k1[4] = multivarK1G(x[4], y[4], length1, A[4], k0[4])
	k2[4] = multivarK2G(x[4], y[4], length2, A[4], k0[4], k1[4])
	k3[4] = multivarK3G(x[4], y[4], length3, A[4], k0[4], k1[4], k2[4])
	k4[4] = multivarK4G(x[4], y[4], length4, A[4], k0[4], k1[4], k2[4], k3[4])

	print('k5')
	k0[5] = multivarK0G(y[5])
	k1[5] = multivarK1G(x[5], y[5], length1, A[5], k0[5])
	k2[5] = multivarK2G(x[5], y[5], length2, A[5], k0[5], k1[5])
	k3[5] = multivarK3G(x[5], y[5], length3, A[5], k0[5], k1[5], k2[5])
	k4[5] = multivarK4G(x[5], y[5], length4, A[5], k0[5], k1[5], k2[5], k3[5])
	k5[5] = multivarK5G(x[5], y[5], length5, A[5], k0[5], k1[5], k2[5], k3[5], k4[5])

	#k4 = np.zeros(shape=(order+1, length4, length4, length4, length4))
	#k5 = np.zeros(shape=(order+1, length5, length5, length5, length5, length5))
##
	return (k0, k1, k2, k3, k4, k5, A)
#-------------------------------------------------


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
	#print(A, A[2])
	k0[2] = multivarK0(y[2])
	k2[2] = multivarK2(x[2], y[2], length2, A[2], k0[2])

	print('k3')
	k1[3] = multivarK1(x[3], y[3], length1, A[3])
	k3[3] = multivarK3(x[3], y[3], length3, A[3], k1[3])

##
	print('k4')
	#print(A, A[4])
	k2[4] = multivarK2(x[4], y[4], length2, A[4], k0[4])
	k4[4] = multivarK4(x[4], y[4], length4, A[4], k0[4], k2[4])

	print('k5')
	k1[5] = multivarK1(x[5], y[5], length1, A[5])
	k3[5] = multivarK3(x[5], y[5], length3, A[5], k1[5])
	k5[5] = multivarK5(x[5], y[5], length5, A[5], k1[5], k3[5])

	#k4 = np.zeros(shape=(order+1, length4, length4, length4, length4))
	#k5 = np.zeros(shape=(order+1, length5, length5,
	#					length5, length5, length5))
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

	volterra_sys = VolterraResponse(use_opencl=True)
	volterra_sys.set_kernels(h0, h1, h2, h3, h4, h5)
	return (volterra_sys)


def leeSchetzenMultivar(x, y, length1, length2, length3, length4, length5, nr_of_workers=2):
	for i in np.arange(6):
		x[i] -= np.mean(x[i])
		#y[i] -= np.mean(y[i])

	# x=np.array([x,x,x,x])
	# y=np.array([y,y,y,y])

	# Kernel calculation
	print('Starting Multi-Variance Wiener Kernel estimation:')
	# print('Processors used: '+str(nr_of_workers)+' out of '+str(cpu_count()))

	# Calculate Wiener Kernels and Gx-------------------------------------------
	k0, k1, k2, k3, k4, k5, A = calcWienerMultivarG(
		x, y, length1, length2, length3, length4, length5)
	#k0, k1, k2, k3, k4, k5, A = calcWienerMultivar(
	#	x, y, length1, length2, length3, length4, length5)

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

#---------------------------------------------------------
def multivarK1test(length1, A):
	k_1 = np.zeros(shape=(length1))
	k_1out = np.zeros(shape=(length1))


	for [tau1] in list(combinations_with_replacement(range(length1), 1)):

		k_1[[tau1]] =  pyExpVal([0, tau1], np.sqrt(A)) + 0.1*pyExpVal([0,0,0,0,0,tau1], np.sqrt(A))+ 0.1*pyExpVal([0,0,0,1,2,tau1], np.sqrt(A))#y=x(n-0) + 0.1*x(n-0)^5

		k_1out[tau1] = k_1[tau1] / math.factorial(1) / A ** 1


	return k_1out	

def multivarK3test(length3, A, k1):
	k_3 = np.zeros(shape=(length3, length3, length3))
	k_3out = np.zeros(shape=(length3, length3, length3))
	#k1=np.zeros(length3)
	#k1[0]=1

	for [tau1, tau2, tau3] in list(combinations_with_replacement(range(length3), 3)):


		k_3[[tau1, tau2, tau3]] =  pyExpVal([0, tau1, tau2, tau3], np.sqrt(A)) + 0.1*pyExpVal([0,0,0,0,0,tau1,tau2,tau3], np.sqrt(A))+ 0.1*pyExpVal([0,0,0,1,2,tau1,tau2,tau3], np.sqrt(A))#y=x(n-0)

		for [v1] in list(combinations_with_replacement(range(length3), 1)):
			
			k_3[tau1, tau2, tau3] -= k1[v1] * pyExpVal([v1, tau1, tau2, tau3], np.sqrt(A))
			

		k_3out[tau1, tau2, tau3] = k_3[tau1, tau2, tau3] / math.factorial(3) / A ** 3

		perms = list(set(permutations([tau1, tau2, tau3])))
		for (t1, t2, t3) in perms:
			k_3out[t1, t2, t3] = k_3out[tau1, tau2, tau3]
		
		#print (k_5 - expVal)
		#print([0, tau1, tau2, tau3, tau4, tau5])
		#print(pyExpVal([0, tau1, tau2, tau3, tau4, tau5], 1))
	return k_3out	


	
def multivarK5test(length5, A, k3, k1):
	k_5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	k_5out = np.zeros(shape=(length5, length5, length5, length5, length5))
	#k1=np.zeros(length5)
	#k1[0]=1
	#k3=np.zeros((length5,length5,length5))
	#k3[0,0,0]=0.1
	
	expValX5 = np.zeros(shape=(length5, length5, length5, length5, length5))
	A5eff = np.zeros(shape=(length5, length5, length5, length5, length5))

	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(range(length5), 5)):


		k_5[[tau1, tau2, tau3, tau4, tau5]] =  pyExpVal([0, tau1, tau2, tau3, tau4, tau5], np.sqrt(A))+ 0.1*pyExpVal([0,0,0,0,0,tau1,tau2,tau3,tau4,tau5], np.sqrt(A))+ 0.1*pyExpVal([0,0,0,1,2,tau1,tau2,tau3,tau4,tau5], np.sqrt(A))#y=x(n-0)

		for [v1] in list(combinations_with_replacement(range(length5), 1)):
			
			k_5[tau1, tau2, tau3, tau4, tau5] -= k1[v1] * pyExpVal([v1, tau1, tau2, tau3, tau4, tau5], np.sqrt(A))

		for [v1,v2,v3] in list(combinations_with_replacement(range(length5), 3)):
			perm_factor = 6 if len(set([v1, v2, v3])) == 3 else 3 if len(set([v1, v2, v3])) == 2 else 1

			k_5[tau1, tau2, tau3, tau4, tau5] -= perm_factor * k3[v1, v2, v3] * pyExpVal([v1, v2, v3, tau1, tau2, tau3, tau4, tau5], np.sqrt(A))
				 
		for [v1, v2] in list(combinations_with_replacement(range(length5), 2)):
			perm_factor = 2 if v1 != v2 else 1

			k_5[tau1, tau2, tau3, tau4, tau5] += perm_factor * 3 * A *k3[v1, v2, v2] * pyExpVal([v1, tau1, tau2, tau3, tau4, tau5], np.sqrt(A))

		#if(k_5[tau1, tau2, tau3, tau4, tau5]!=0):
		#	print(([tau1, tau2, tau3, tau4, tau5]))
			

		k_5out[tau1, tau2, tau3, tau4, tau5] = k_5[tau1, tau2, tau3, tau4, tau5] / math.factorial(5) / A ** 5

		perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for (t1, t2, t3, t4, t5) in perms:
			k_5out[t1, t2, t3, t4, t5] = k_5out[tau1, tau2, tau3, tau4, tau5]
		
		#print (k_5 - expVal)
		#print([0, tau1, tau2, tau3, tau4, tau5])
		#print(pyExpVal([0, tau1, tau2, tau3, tau4, tau5], 1))
	return k_5out	

if(0): #Test sensitivity of identification towards measurement errors
	#print(0.1*pyExpVal([0,0,0,0,0,0,0,0], 1) + pyExpVal([0,0,0,0], 1))

	k1=multivarK1test(length1=5, A=1)
	k3=multivarK3test(length3=5, A=1, k1=k1)
	k5=multivarK5test(length5=5, A=1, k3=k3, k1=k1)

	#k5[0,0,0,0,0]+=0.01

	k0=([0])
	k2=np.zeros((5,5))
	k4=np.zeros((5,5,5,5))

	k2[0,0]=1
	#print(np.reshape(np.repeat(k2, 6, axis=2), (6,5,5)))

	k0=np.tile(k0, (6, 1))
	k1=np.tile(k1, (6, 1))
	k2=np.tile(k2, (6, 1, 1))
	k3=np.tile(k3, (6, 1, 1, 1))
	k4=np.tile(k4, (6, 1, 1, 1, 1))
	k5=np.tile(k5, (6, 1, 1, 1, 1, 1))


	A=np.ones(6)

	system = wiener2VolterraMultivar(k0, k1, k2, k3, k4, k5, A)

	print(system.h1)
	print(system.h3)
	print(system.h5)

	#-----------------------------------------------------


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
	length2 = 20
	length3 = 20
	length4 = 8
	length5 = 8

	# System Parameters
	inputBandwidth = 3.333333e9  # Equals half the sample rate
	validBandwidth = 2.6e9  # Equals the AA-filter cutoff
	#inputBandwidth = 5.5e9  # Equals half the sample rate
	#validBandwidth = 4.6e9  # Equals the AA-filter cutoff
	order = 3

	# Skew output channel this much
	skew = 0
	AntiAliasing = False
	
	#Frontend Noise Correction
	FNCorrection=True

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
			filename = 'LMH3401XY16l'
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

			filename2 = 'LMH3401XY16m'
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

			filename3 = 'LMH3401XY16h'
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
			
			filename4 = 'LMH3401XY16hh'
			if os.name == 'nt':  # Windows
				print('File: '+filename4+'.csv')
				df = pd.read_csv(wdin+'/'+filename4+'.csv', sep=',',
								 header=None, dtype=str)  # Oscilloscope Data
			else:  # Other OS (Linux, MacOS)
				absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filename4}.csv'
				print('File: ' + absolute_path)
				df = pd.read_csv(absolute_path, sep=',',
								 header=None, dtype=str)

			# normal file
			times4 = np.transpose(df.values[0:])[0].astype(float)
			x4 = np.transpose(df.values[0:])[1].astype(
				float)  # Start with 0th row
			y4 = np.transpose(df.values[0:])[2].astype(
				float)  # Start with 0th row
			# decorrelated file
			# times		= np.transpose(df.values[0:])[0].astype(float)
			# xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
			# x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
			# y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

			# Check if data is too long or to short
			if (len(times4) >= datastart+datalength):
				times4 = times4[datastart:datastart+datalength]
				x4 = x4[datastart:datastart+datalength]
				y4 = y4[datastart:datastart+datalength]
			datalength = len(times4)
			
			#Not yet used--------------------------------
			filenameFrontendNoise = 'LMH3401XY16FN'
			if os.name == 'nt':  # Windows
				print('File: '+filenameFrontendNoise+'.csv')
				df = pd.read_csv(wdin+'/'+filenameFrontendNoise+'.csv', sep=',',
								 header=None, dtype=str)  # Oscilloscope Data
			else:  # Other OS (Linux, MacOS)
				absolute_path = f'/home/mrani/Volterra/Final/Volterra_Input_Data/{filenameFrontendNoise}.csv'
				print('File: ' + absolute_path)
				df = pd.read_csv(absolute_path, sep=',',
								 header=None, dtype=str)

			# normal file
			#Frontend Noise
			timesFNoise = np.transpose(df.values[0:])[0].astype(float)
			xFNoise = np.transpose(df.values[0:])[1].astype(
				float)  # Start with 0th row
			yFNoise = np.transpose(df.values[0:])[2].astype(
				float)  # Start with 0th row
			# decorrelated file
			# times		= np.transpose(df.values[0:])[0].astype(float)
			# xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
			# x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
			# y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

			# Check if data is too long or to short
			if (len(timesFNoise) >= datastart+datalength):
				timesFNoise = timesFNoise[datastart:datastart+datalength]
				xFNoise = xFNoise[datastart:datastart+datalength]
				yFNoise = yFNoise[datastart:datastart+datalength]
			datalength = len(times3)
			#--------------------------------------
			
			
			x = np.nan_to_num(x)
			x2 = np.nan_to_num(x2)
			x3 = np.nan_to_num(x3)
			x4 = np.nan_to_num(x4)
			
			y = np.nan_to_num(y)
			y2 = np.nan_to_num(y2)
			y3 = np.nan_to_num(y3)
			y4 = np.nan_to_num(y4)
			#y5 = np.nan_to_num(y5)
			
			print(np.var(x))
			xMultivar = np.array([x, x2, x3, x4, x4, x4])
			yMultivar = np.array([y, y2, y3, y4, y4, y4])
			timesMultivar = np.array(
				[times, times, times2, times3, times4, times4])

	signalScale = 0.3

	Gain = 1  # Gain of the amplifier
	if (0):  # Generate simulated input data
		print('sim')

		x = np.random.normal(loc=0.0, scale=0.15*signalScale, size=datalength)
		x -= np.mean(x)
		x = LPfilter(x, times=times, fstop=2.6e9)[0] + 0.1*x
		
		x2 = np.random.normal(loc=0.0, scale=0.35*signalScale, size=datalength)
		x2 -= np.mean(x2)
		x2 = LPfilter(x2, times=times, fstop=2.6e9)[0] + 0.1*x2
		
		x3 = np.random.normal(loc=0.0, scale=0.707*signalScale, size=datalength)
		x3 -= np.mean(x3)
		x3 = LPfilter(x3, times=times, fstop=2.6e9)[0] + 0.1*x3
		
		x4 = np.random.normal(loc=0.0, scale=signalScale, size=datalength)
		x4 -= np.mean(x4)
		x4 = LPfilter(x4, times=times, fstop=2.6e9)[0] + 0.1*x4
		
		times2=times
		times3=times
		times4=times
		
		filename='sim_l'
		filename2='sim_m'
		filename3='sim_h'
		filename4='sim_uh'
		
		
		y = systemFunction(x, 5000e6, 300e6, dt=25e-12, oversampleFactor=5, asymmetry=0.1, delay=150e-12)
		print('sim0 done')
		y2 = systemFunction(x2, 5000e6, 300e6, dt=25e-12, oversampleFactor=5, asymmetry=0.1, delay=150e-12)
		print('sim1 done')
		y3 = systemFunction(x3, 5000e6, 300e6, dt=25e-12, oversampleFactor=5, asymmetry=0.1, delay=150e-12)
		print('sim2 done')
		y4 = systemFunction(x4, 5000e6, 300e6, dt=25e-12, oversampleFactor=5, asymmetry=0.1, delay=150e-12)
		print('sim3 done')
		
		
		
		#y=x + 10*x**2 - 30*x**3 + 300*x**4 - 1000*x**5
		#y2=x2 + 10*x2**2 - 30*x2**3 + 300*x2**4 - 1000*x2**5
		#y3=x3 + 10*x3**2 - 30*x3**3 + 300*x3**4 - 1000*x3**5
		
		xMultivar = np.array([x, x2, x3, x4, x4, x4])
		yMultivar = np.array([y, y2, y3, y4, y4, y4])
		timesMultivar = np.array([times, times2, times3, times4, times4, times4])

		# y=y-24


# -----------------------------------------------------------------------------------------------------------

	# Downsampling of the input and output signals according to 'Volterra Kernel interpolation' by Köppl:
	if (1):
		for i in np.arange(6):
			tempTimes, tempX, tempY = koepplDownsampling(
				datalength, timesMultivar[i], xMultivar[i], yMultivar[i], order, inputBandwidth)
			dl = len(tempX)
			tempTimes = tempTimes[:dl]
			timesMultivar[i, 0:dl], xMultivar[i,0:dl], yMultivar[i, 0:dl] = tempTimes, tempX, tempY
		timesMultivar = timesMultivar[:, :dl]
		xMultivar = xMultivar[:, :dl]
		yMultivar = yMultivar[:, :dl]
		
		#Do it for the frontend noise file---
		tempTimes, tempX, tempY = koepplDownsampling(datalength, timesFNoise, xFNoise, yFNoise, order, inputBandwidth)
		dl = len(tempX)
		tempTimes = tempTimes[:dl]
		timesFNoise[0:dl], xFNoise[0:dl], yFNoise[0:dl] = tempTimes, tempX, tempY
		timesFNoise = timesFNoise[:dl]
		xFNoise = xFNoise[:dl]
		yFNoise = yFNoise[:dl]
		#---
		
		datalength = dl


#Insert the frontend noise correction here#----
#empty
#---------------------------
	if(FNCorrection):
		for i in np.arange(6):
			frontendKernel,invFrontendKernel=findFrontendNoiseKernel(xMultivar[i,:], xFNoise, length1)
			
			#plt.stem(invFrontendKernel)
			#plt.show()
			
			#plt.plot(20*np.log10(np.abs(np.fft.fft(invFrontendKernel))))
			#plt.show()
			xMultivar[i,:]=np.convolve(xMultivar[i,:], invFrontendKernel, mode='same')

# -------------------------------------------------

	for i in np.arange(6):
		xMultivar[i] -= np.mean(xMultivar[i])
		xFNoise-=np.mean(xFNoise)
		#yMultivar[i] -= np.mean(yMultivar[i])
	
	#print(np.var(xMultivar, axis=1))	
	#print(np.var(yMultivar, axis=1))
	
	# Calculate variances of resulting signals
	Az = np.var(xMultivar, axis=1)
	#print(Az)
	varXoriginal = np.copy(Az)

	# plt.plot(calc_cyx(1000, 10000, 2000,
	#		  xMultivar[0, :-1000], xMultivar[0, 1000:]))
	# plt.show()

# Add noise for testing-------------------------------------------------------------------------------
	if (0):
		SNR = 100  # dB, assumed to be present both on x/y-channels, reference = total input variance of x
		noiseScale = np.sqrt(varXoriginal)*10**(-SNR/20)
		np.random.seed(35630535)

		for i in np.arange(6):
			# print(np.sqrt(np.var(xMultivar[i,:])))
			xMultivar[i, :] = xMultivar[i, :] + \
				np.random.normal(loc=0.0, scale=noiseScale[i], size=datalength)


# -------------------------------------------------
	# HF Noise Injection - fills out the spectrum gap at the end of LPF bandwidth

	HFnoiseX=np.zeros(shape=(6,len(xMultivar[0])))
	for i in np.arange(6):
		np.random.seed(123)
		HFnoiseX[i], _ = HPfilter(x=np.random.normal(loc=0.0, scale=np.sqrt(
			Az[i]), size=datalength), times=timesMultivar[i], fstop=validBandwidth)


	# Use a minimum-phase DUT filter, instead of non-causal/linear phase.
	if (1):
		print('Replacing the low-pass filter by its minimum-phase version')
		# Calculate Covariances
		#Use the sigma_1 for determination of the allpass filter. 
		#It should be correctly scaled for this application.
		cxx = calc_cxx(0, datalength, 3*length1, xMultivar[1, :])[:3*length1]
		cxxhf = calc_cxx(0, datalength, 3*length1, HFnoiseX[1])[:3*length1]

		cxx = np.roll(np.concatenate((cxx, cxx[::-1][:-1])), 3*length1-1)
		cxxhf = np.roll(np.concatenate(
			(cxxhf, cxxhf[::-1][:-1])), 3*length1-1)

		CXX = np.fft.fft(cxx)
		CXXHF = np.fft.fft(cxxhf)

		# Calculate transfer function of the resulting filter
		FILT = CXX/(CXX+CXXHF)

		filt = np.real(np.roll(np.fft.ifft(FILT), 3*length1-1))

		# Build minimum phase version of the filter
		filt_mp = sg.minimum_phase(filt, method='homomorphic', n_fft=3000)
		filt_mp = np.concatenate((np.zeros(3*length1-1), filt_mp))

		filt_mp = np.convolve(filt_mp, filt_mp, mode='same')

		# plt.plot(filt)
		#plt.plot(filt_mp)
		#plt.show()
		#plt.plot(np.abs(np.fft.fft(filt)))
		#plt.plot(np.abs(np.fft.fft(filt_mp)))
		#plt.show()

		filt_mp = np.pad(filt_mp, 1000, 'constant')
		filt = np.pad(filt, 1000, 'constant')

		# Construct a phase filter that only contains the phase differences between minimum-phase and linear-phase filter
		PHASEFILT = np.fft.fft(filt_mp)/np.fft.fft(filt)
		
		# if max(np.abs(PHASEFILT))>1.3:
			# print('WARNING: Phase filter does not have good all-pass behaviour. This can be due to limited accuracy of sg.minimum-phase()')
			# print('Max. Gain: '+str(max(np.abs(PHASEFILT))))
			# print('Clipping the filter...')
		for j in np.arange(len(PHASEFILT)):#Do this, because the sg.minimum-phase() function is not perfect and will create amplitude errors. This line will force it to be an all-pass.
			PHASEFILT[j]=PHASEFILT[j]/np.abs(PHASEFILT[j])*np.exp(1j*np.angle(PHASEFILT[j]))

		phasefilt = np.real(np.fft.ifft(PHASEFILT))  # [:length1]
		phasefilt = np.roll(phasefilt, 1000+3*length1-1)
		phasefilt = phasefilt[1000:1000+2*3*length1]
		#apply blackman window
		phasefilt = phasefilt * (0.42 - 0.5*np.cos(2*np.pi*np.arange(2*3*length1) / (2*3*length1-1)) + 0.08*np.cos(4*np.pi*np.arange(2*3*length1) / (2*3*length1-1))  )
		
		#plt.plot(np.abs(np.fft.fft(phasefilt)))
		#plt.show()
		#plt.plot( (0.42 - 0.5*np.cos(2*np.pi*np.arange(2*3*length1) / (2*3*length1-1)) + 0.08*np.cos(4*np.pi*np.arange(2*3*length1) / (2*3*length1-1))  ))
		#plt.show()
		

	for i in np.arange(6):# Apply the phase filter to all y
		yMultivar[i, :] = np.convolve(
			yMultivar[i, :], phasefilt, mode='same')

			# plt.plot(np.angle(np.fft.fft(np.roll(phasefilt, -length1-2))))
			# plt.plot(np.roll(phasefilt, -length1+2))
			# plt.show()
# ---------------------------------------------------------------

		xMultivar[i, :] = xMultivar[i, :]+HFnoiseX[i,:]
	#print(np.var(yMultivar, axis=1))
	# HFnoiseX2,_=HPfilter(x=np.random.normal(loc=0.0, scale=np.sqrt(np.var(x2)), size=len(x2)), times=times, fstop=validBandwidth)
	# x2=x2+HFnoiseX2
	# HFnoiseY=HFnoiseX
	# HFnoiseY[8:]=HFnoiseY[:-8]
	# y=y-1*HFnoiseX

	# End of input signal preparation-------------------------------------------------------------------------
	seconds1 = time.time()
	print('Done preparing data. Time passed: ' +
		  '{:.1f}'.format(seconds1-seconds0)+'s')
	print('')
	
	#yMultivar=np.copy(xMultivar)

# ------------------------------------------------------------------------------------------------------------
	# Decorrelate
	xModMultivar = np.zeros(shape=np.shape(xMultivar))
	#print(np.var(yMultivar, axis=1))

	# Automatic
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
		alpha=np.array([0, 0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1])
		#alpha=np.array([0, 0.5, 0.4, 0.4, 0.25, 0.25, 0.15, 0.15])
		length6=9
		length7=9
		#length=np.array([length1,length2,length3,length4,length5,length6,length7])
		length=np.array([length1,length1,length1,length2,length3,length6,length7])
		
		
		# vars=np.zeros(100)
		# for i in np.arange(100):
			# vars[i]=np.var(xMultivar[1][10000*i:10000+10000*i])
		# plt.plot(vars)
		# plt.show()
		# print(np.max(xMultivar[1]))
		# print(np.min(xMultivar[1]))
			
		xMultivar[0,0:6400]=np.zeros(6400)
		xMultivar[1,0:6400]=np.zeros(6400)
		xMultivar[2,0:6400]=np.zeros(6400)
		xMultivar[3,0:6400]=np.zeros(6400)
		
		xModMultivar[0] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[0], xMultivar[0], yMultivar[0], alpha, length, iterations=70, write=True, name=filename, chunks=256)
		xModMultivar[1] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[1], xMultivar[1], yMultivar[1], alpha, length, iterations=70, write=True, name=filename2, chunks=256)
		xModMultivar[2] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[2], xMultivar[2], yMultivar[2], alpha, length, iterations=70, write=True, name=filename3, chunks=256)
		xModMultivar[3] = Decorrelation5_OCL.deCorrelateFast(timesMultivar[3], xMultivar[3], yMultivar[3], alpha, length, iterations=70, write=True, name=filename4, chunks=256)	
		
		#xModMultivar[1] = np.copy(xModMultivar[0])
		xModMultivar[4] = np.copy(xModMultivar[3])
		xModMultivar[5] = np.copy(xModMultivar[4])

	# Skew the data to account for differences in cable length, system delay, etc.
	if skew != 0:
		for i in np.arange(6):
			xMultivar[i, skew:] = xMultivar[i, :-skew]
			xMultivar[i, :skew] = np.zeros(skew)
			yMultivar[i, skew:] = yMultivar[i, :-skew]
			yMultivar[i, :skew] = np.zeros(skew)

	for i in np.arange(6):
		#Shift y back for a few samples. This makes non-causal filters causal. 
		#yMultivar[i, 1:] = yMultivar[i, :-1]
		#yMultivar[i, :1] = np.zeros(1)
		yMultivar[i, :-3] = yMultivar[i, 3:]#3
		yMultivar[i, -3:] = np.zeros(3)


	#Analyse the resulting time shift between x and y data. The shift is defined as where the peak absolute value occurs in x/y correlation.
	analysisLength = 500
	cxy = calc_cyx(0, 100000, 2*analysisLength,
				   yMultivar[3, :-analysisLength], xMultivar[3, analysisLength:])[:2*analysisLength]
	print('y has '+str(np.argmax(np.abs(cxy))-analysisLength) +
		  ' samples of time shift compared to x')

	seconds2 = time.time()
	print('Done modifying data. Time passed: ' +
		  '{:.1f}'.format(seconds2-seconds1)+'s')

	#plt.plot(cxy)
	#plt.show()

# Identify the kernels--------------------------------------------------------------------
	#print(np.var(yMultivar, axis=1))
	identifyKernelsMultivar(xModMultivar, yMultivar, length1, length2, length3, length4, length5, nr_of_workers,
							iterations=5, alpha1=0.8, AntiAliasing=False, write=True, name='overall')
	identifyKernelsMultivar(xModMultivar, xMultivar/refFactorY, length1, length2, length3, length4, length5,
							nr_of_workers, iterations=5, alpha1=0.8, AntiAliasing=False, write=True, name='source')
	#print(np.var(yMultivar, axis=1))
	invertKernelFiles(nameIn='source', nameOut='invSource',
					  skew=2*skew)  # skew=2*skew
	combineKernelFiles(nameInA='invSource', nameInB='overall', nameOut='DUT')

	# plot volterra system output of the DUT system for a sin wave input
	if (1):

		f = 10e6
		A = 0.2
		t = np.arange(0, 10000)*150e-12
		x = A * np.sin(2 * np.pi * f * t)

		system = VolterraResponse(use_opencl=True)

		system = readKernels(name='DUT')
		system_out = system.Volterra_In_Out(x)
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










###################################################################################################


# def multivarK4(x, y, length4, A, k0, k2):
#	 n_start = length4
#	 n_stop = len(x)
#	 k_4 = np.zeros(shape=(length4, length4, length4, length4))

#	 expValX4 = np.zeros(shape=(length4, length4, length4, length4))
#	 A4eff = np.zeros(shape=(length4, length4, length4, length4))

#	 for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):

#		 if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0):
#			 vecX4 = np.zeros(n_stop - n_start)
#			 lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4)
#			 vecX4[max(tau1, tau2, tau3, tau4):] = (x[n_stop - tau1 - lenFilled:n_stop - tau1] * x[n_stop - tau2 -
#																								   lenFilled:n_stop - tau2] * x[n_stop - tau3 - lenFilled:n_stop - tau3] * x[n_stop - tau4 - lenFilled:n_stop - tau4])
#		 else:
#			 vecX4 = (
#				 x[n_start - tau1:n_stop - tau1] *
#				 x[n_start - tau2:n_stop - tau2] *
#				 x[n_start - tau3:n_stop - tau3] *
#				 x[n_start - tau4:n_stop - tau4]
#			 )

#		 expValX4[tau1, tau2, tau3, tau4] = np.mean(vecX4 ** 2)
#		 A4eff[tau1, tau2, tau3, tau4] = np.power(
#			 expValX4[tau1, tau2, tau3, tau4] /
#			 calcDiagonalFactor((tau1, tau2, tau3, tau4)), 1/4
#		 )

#		 vecY = y[n_start:n_stop]
#		 k_4[tau1, tau2, tau3, tau4] = (
#			 (np.sum(vecX4 * vecY) / (n_stop - n_start) -
#			  2 * A4eff[tau1, tau2, tau3, tau4] ** 3 *
#			  (k2[tau1, tau2] * delta(tau3 - tau4) +
#			   k2[tau1, tau3] * delta(tau2 - tau4) +
#			   k2[tau1, tau4] * delta(tau2 - tau3) +
#			   k2[tau2, tau3] * delta(tau1 - tau4) +
#			   k2[tau2, tau4] * delta(tau1 - tau3) +
#			   k2[tau3, tau4] * delta(tau1 - tau2)) -

#			  A4eff[tau1, tau2, tau3, tau4] ** 2 *
#			  (k0*delta4(tau1, tau2, tau3, tau4))
			

#			  ) / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4
#		 )

#		 perms = list(set(permutations([tau1, tau2, tau3, tau4])))
#		 for (t1, t2, t3, t4) in perms:
#			 k_4[t1, t2, t3, t4] = k_4[tau1, tau2, tau3, tau4]

#	 return k_4


# ## Like in Orcioni paper ###
# def multivarK4(x, y, length4, A, k0, k2):
#	 n_start = length4
#	 n_stop = len(x)
#	 k_4 = np.zeros(shape=(length4, length4, length4, length4))

#	 expValX4 = np.zeros(shape=(length4, length4, length4, length4))
#	 A4eff = np.zeros(shape=(length4, length4, length4, length4))

#	 for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):

#		 if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0):
#			 vecX4 = np.zeros(n_stop - n_start)
#			 lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4)
#			 vecX4[max(tau1, tau2, tau3, tau4):] = (x[n_stop - tau1 - lenFilled:n_stop - tau1] * x[n_stop - tau2 -
#																								   lenFilled:n_stop - tau2] * x[n_stop - tau3 - lenFilled:n_stop - tau3] * x[n_stop - tau4 - lenFilled:n_stop - tau4])
#		 else:
#			 vecX4 = (
#				 x[n_start - tau1:n_stop - tau1] *
#				 x[n_start - tau2:n_stop - tau2] *
#				 x[n_start - tau3:n_stop - tau3] *
#				 x[n_start - tau4:n_stop - tau4]
#			 )

#		 expValX4[tau1, tau2, tau3, tau4] = np.mean(vecX4 ** 2)
#		 A4eff[tau1, tau2, tau3, tau4] = np.power(
#			 expValX4[tau1, tau2, tau3, tau4] /
#			 calcDiagonalFactor((tau1, tau2, tau3, tau4)), 1/4
#		 )

#		 vecY = y[n_start:n_stop]
#		 k_4[tau1, tau2, tau3, tau4] = (
#			 (np.sum(vecX4 * vecY) / (n_stop - n_start) -
#			  A4eff[tau1, tau2, tau3, tau4] ** 4 *
#			  (k2[tau1, tau2] * delta(tau3 - tau4) +
#			   k2[tau1, tau3] * delta(tau2 - tau4) +
#			   k2[tau1, tau4] * delta(tau2 - tau3) +
#			   k2[tau2, tau3] * delta(tau1 - tau4) +
#			   k2[tau2, tau4] * delta(tau1 - tau3) +
#			   k2[tau3, tau4] * delta(tau1 - tau2)) 

#			  ) / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4
#		 )

#		 perms = list(set(permutations([tau1, tau2, tau3, tau4])))
#		 for (t1, t2, t3, t4) in perms:
#			 k_4[t1, t2, t3, t4] = k_4[tau1, tau2, tau3, tau4]

#	 return k_4


# def multivarK5(x, y, length5, A, k1, k3):
#	 n_start = length5
#	 n_stop = len(x)
#	 k_5 = np.zeros((length5, length5, length5, length5, length5))

#	 expValX5 = np.zeros((length5, length5, length5, length5, length5))
#	 A5eff = np.zeros((length5, length5, length5, length5, length5))

#	 for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
#		 if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0) or (n_start - tau5 < 0):
#			 vecX5 = np.zeros(n_stop - n_start)
#			 lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4, tau5)
#			 vecX5[max(tau1, tau2, tau3, tau4, tau5):] = (
#				 x[n_stop - tau1 - lenFilled:n_stop - tau1] *
#				 x[n_stop - tau2 - lenFilled:n_stop - tau2] *
#				 x[n_stop - tau3 - lenFilled:n_stop - tau3] *
#				 x[n_stop - tau4 - lenFilled:n_stop - tau4] *
#				 x[n_stop - tau5 - lenFilled:n_stop - tau5]
#			 )
#		 else:
#			 vecX5 = (
#				 x[n_start - tau1:n_stop - tau1] *
#				 x[n_start - tau2:n_stop - tau2] *
#				 x[n_start - tau3:n_stop - tau3] *
#				 x[n_start - tau4:n_stop - tau4] *
#				 x[n_start - tau5:n_stop - tau5]
#			 )

#		 expValX5[tau1, tau2, tau3, tau4, tau5] = np.mean(vecX5 ** 2)
#		 A5eff[tau1, tau2, tau3, tau4, tau5] = np.power(
#			 expValX5[tau1, tau2, tau3, tau4, tau5] /
#			 calcDiagonalFactor((tau1, tau2, tau3, tau4, tau5)), 1/5
#		 )

#		 vecY = y[n_start:n_stop]
#		 k_5[tau1, tau2, tau3, tau4, tau5] = (
#			 np.sum(vecX5 * vecY) / (n_stop - n_start) -
#			 math.factorial(3) * A5eff[tau1, tau2, tau3, tau4, tau5]**4 * (
#				 k3[tau1, tau2, tau3] * delta(tau4 - tau5) +
#				 k3[tau1, tau2, tau4] * delta(tau3 - tau5) +
#				 k3[tau1, tau2, tau5] * delta(tau3 - tau4) +
#				 k3[tau1, tau3, tau4] * delta(tau2 - tau5) +
#				 k3[tau1, tau3, tau5] * delta(tau2 - tau4) +
#				 k3[tau1, tau4, tau5] * delta(tau2 - tau3) +
#				 k3[tau2, tau3, tau4] * delta(tau1 - tau5) +
#				 k3[tau2, tau3, tau5] * delta(tau1 - tau4) +
#				 k3[tau2, tau4, tau5] * delta(tau1 - tau3) +
#				 k3[tau3, tau4, tau5] * delta(tau1 - tau2)) - 
				
#				 A5eff[tau1, tau2, tau3, tau4, tau5]**3 * (
#				 k1[tau1] * delta4(tau2, tau3, tau4, tau5) +
#				 k1[tau2] * delta4(tau1, tau3, tau4, tau5) +
#				 k1[tau3] * delta4(tau1, tau2, tau4, tau5) +
#				 k1[tau4] * delta4(tau1, tau2, tau3, tau5) +
#				 k1[tau5] * delta4(tau1, tau2, tau3, tau4)
#			 ) 
#		 ) / math.factorial(5) / A5eff[tau1, tau2, tau3, tau4, tau5]**5

#		 perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
#		 for (t1, t2, t3, t4, t5) in perms:
#			 k_5[t1, t2, t3, t4, t5] = k_5[tau1, tau2, tau3, tau4, tau5]

#	 return k_5





### my derivation ########

# def multivarK4(x, y, length4, A, k0, k2):
#	 n_start = length4
#	 n_stop = len(x)
#	 k_4 = np.zeros(shape=(length4, length4, length4, length4))

#	 expValX4 = np.zeros(shape=(length4, length4, length4, length4))
#	 A4eff = np.zeros(shape=(length4, length4, length4, length4))

#	 for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):

#		 if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0):
#			 vecX4 = np.zeros(n_stop - n_start)
#			 lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4)
#			 vecX4[max(tau1, tau2, tau3, tau4):] = (x[n_stop - tau1 - lenFilled:n_stop - tau1] * x[n_stop - tau2 -
#																								   lenFilled:n_stop - tau2] * x[n_stop - tau3 - lenFilled:n_stop - tau3] * x[n_stop - tau4 - lenFilled:n_stop - tau4])
#		 else:
#			 vecX4 = (
#				 x[n_start - tau1:n_stop - tau1] *
#				 x[n_start - tau2:n_stop - tau2] *
#				 x[n_start - tau3:n_stop - tau3] *
#				 x[n_start - tau4:n_stop - tau4]
#			 )

#		 expValX4[tau1, tau2, tau3, tau4] = np.mean(vecX4 ** 2)
#		 A4eff[tau1, tau2, tau3, tau4] = np.power(
#			 expValX4[tau1, tau2, tau3, tau4] /
#			 calcDiagonalFactor((tau1, tau2, tau3, tau4)), 1/4
#		 )

#		 vecY = y[n_start:n_stop]
#		 k_4[tau1, tau2, tau3, tau4] = (
#			 (np.sum(vecX4 * vecY) / (n_stop - n_start) -
#			  A4eff[tau1, tau2, tau3, tau4] ** 3 *
#			  (k2[tau1, tau2] * delta(tau3 - tau4) +
#			   k2[tau1, tau3] * delta(tau2 - tau4) +
#			   k2[tau1, tau4] * delta(tau2 - tau3) +
#			   k2[tau2, tau3] * delta(tau1 - tau4) +
#			   k2[tau2, tau4] * delta(tau1 - tau3) +
#			   k2[tau3, tau4] * delta(tau1 - tau2)) -

#			  A4eff[tau1, tau2, tau3, tau4] ** 2 *
#			  (k0*delta4(tau1, tau2, tau3, tau4)) +
#			  A4eff[tau1, tau2, tau3, tau4] *
#				 (k2[tau1, tau1] * delta4(tau1, tau2, tau3, tau4))

#			  ) / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4
#		 )

#		 perms = list(set(permutations([tau1, tau2, tau3, tau4])))
#		 for (t1, t2, t3, t4) in perms:
#			 k_4[t1, t2, t3, t4] = k_4[tau1, tau2, tau3, tau4]

#	 return k_4

# def multivarK5(x, y, length5, A, k1, k3):
#	 n_start = length5
#	 n_stop = len(x)
#	 k_5 = np.zeros(shape=(length5, length5, length5, length5, length5))

#	 expValX5 = np.zeros(shape=(length5, length5, length5, length5, length5))
#	 A5eff = np.zeros(shape=(length5, length5, length5, length5, length5))

#	 for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(range(length5), 5)):

#		 if (n_start - tau1 < 0) or (n_start - tau2 < 0) or (n_start - tau3 < 0) or (n_start - tau4 < 0) or (n_start - tau5 < 0):
#			 vecX5 = np.zeros(n_stop - n_start)
#			 lenFilled = n_stop - n_start - max(tau1, tau2, tau3, tau4, tau5)
#			 vecX5[max(tau1, tau2, tau3, tau4, tau5):] = (
#				 x[n_stop - tau1 - lenFilled:n_stop - tau1] * 
#				 x[n_stop - tau2 - lenFilled:n_stop - tau2] * 
#				 x[n_stop - tau3 - lenFilled:n_stop - tau3] * 
#				 x[n_stop - tau4 - lenFilled:n_stop - tau4] * 
#				 x[n_stop - tau5 - lenFilled:n_stop - tau5]
#			 )
#		 else:
#			 vecX5 = (
#				 x[n_start - tau1:n_stop - tau1] *
#				 x[n_start - tau2:n_stop - tau2] *
#				 x[n_start - tau3:n_stop - tau3] *
#				 x[n_start - tau4:n_stop - tau4] *
#				 x[n_start - tau5:n_stop - tau5]
#			 )

#		 expValX5[tau1, tau2, tau3, tau4, tau5] = np.mean(vecX5 ** 2)
#		 A5eff[tau1, tau2, tau3, tau4, tau5] = np.power(
#			 expValX5[tau1, tau2, tau3, tau4, tau5] /
#			 calcDiagonalFactor((tau1, tau2, tau3, tau4, tau5)), 1/5
#		 )

#		 vecY = y[n_start:n_stop]
#		 k_5[tau1, tau2, tau3, tau4, tau5] = np.sum(vecX5 * vecY) / (n_stop - n_start)

#		 for [v1,v2,v3] in list(combinations_with_replacement(range(length5), 3)):

#			 k_5[tau1, tau2, tau3, tau4, tau5] -= k3[v1, v2, v3] * pyExpVal([v1, v2, v3, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5])) - 3 * A5eff[tau1, tau2, tau3, tau4, tau5] * k3[v1, tau1, tau2] * pyExpVal([v1, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5])) 
				 
#		 for [v1] in list(combinations_with_replacement(range(length5), 1)):
#			 k_5[tau1, tau2, tau3, tau4, tau5] -= k1[v1] * pyExpVal([v1, tau1, tau2, tau3, tau4, tau5], np.sqrt(A5eff[tau1, tau2, tau3, tau4, tau5])) 
			   
#		 perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
#		 for (t1, t2, t3, t4, t5) in perms:
#			 k_5[t1, t2, t3, t4, t5] = k_5[tau1, tau2, tau3, tau4, tau5]
		
#		 k_5[tau1, tau2, tau3, tau4, tau5] = k_5[tau1, tau2, tau3, tau4, tau5] / math.factorial(5) / A5eff[tau1, tau2, tau3, tau4, tau5] ** 5  

#	 return k_5

##########################



################## OpenCl implementation:

# def multivarK3(x, y, length3, A, k1):
#	 n_start = length3
#	 n_stop = len(x)
#	 k_3 = np.zeros(shape=(length3, length3, length3))
#	 expValX3 = np.zeros(shape=(length3, length3, length3))
#	 A3eff = np.zeros(shape=(length3, length3, length3))

#	 len_x = len(x)
#	 vecX3_length = n_stop - n_start
#	 vecX3 = np.zeros(vecX3_length, dtype=np.float32)
#	 x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
#					   hostbuf=x.astype(np.float32))
#	 vecX3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, vecX3.nbytes)
#	 indices_buf = cl.Buffer(
#		 ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(3, dtype=np.int32))

#	 for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
#		 indices = np.array([tau1, tau2, tau3], dtype=np.int32)
#		 cl.enqueue_copy(queue, indices_buf, indices)
#		 prg.compute_vecX3(queue, (vecX3_length,), None, x_buf, vecX3_buf, indices_buf, np.int32(
#			 n_start), np.int32(n_stop), np.int32(length3), np.int32(len_x))
#		 cl.enqueue_copy(queue, vecX3, vecX3_buf)
#		 vecX3 = vecX3[:n_stop - n_start]

#		 expValX3[tau1, tau2, tau3] = np.mean(vecX3 ** 2)
#		 A3eff[tau1, tau2, tau3] = np.cbrt(
#			 expValX3[tau1, tau2, tau3] / calcDiagonalFactor((tau1, tau2, tau3)))

#		 vecY = y[n_start:n_stop]
#		 k_3[tau1, tau2, tau3] = (np.sum(vecX3 * vecY) / (n_stop - n_start) - A3eff[tau1, tau2, tau3]**2 * (
#			 k1[tau1] * delta(tau2 - tau3) + k1[tau2] *
#			 delta(tau1 - tau3) + k1[tau3] * delta(tau1 - tau2)
#		 )) / math.factorial(3) / A3eff[tau1, tau2, tau3] ** 3

#		 perms = list(set(permutations([tau1, tau2, tau3])))
#		 for (t1, t2, t3) in perms:
#			 k_3[t1, t2, t3] = k_3[tau1, tau2, tau3]

#	 return k_3


# def multivarK4(x, y, length4, A, k2):
#	 n_start = length4
#	 n_stop = len(x)
#	 k_4 = np.zeros(shape=(length4, length4, length4, length4))
#	 expValX4 = np.zeros(shape=(length4, length4, length4, length4))
#	 A4eff = np.zeros(shape=(length4, length4, length4, length4))

#	 # Allocate buffers
#	 len_x = len(x)
#	 vecX4_length = n_stop - n_start
#	 vecX4 = np.zeros(vecX4_length, dtype=np.float32)
#	 x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
#					   hostbuf=x.astype(np.float32))
#	 vecX4_buf = cl.Buffer(ctx, mf.WRITE_ONLY, vecX4.nbytes)
#	 indices_buf = cl.Buffer(
#		 ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(4, dtype=np.int32))
#	 expValX4_buf = cl.Buffer(ctx, mf.WRITE_ONLY, expValX4.nbytes)

#	 for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(range(length4), 4)):
#		 indices = np.array([tau1, tau2, tau3, tau4], dtype=np.int32)

#		 # Update indices buffer
#		 cl.enqueue_copy(queue, indices_buf, indices)

#		 # Execute vecX4 computation kernel
#		 prg.compute_vecX4(queue, (vecX4_length,), None, x_buf, vecX4_buf, indices_buf, np.int32(
#			 n_start), np.int32(n_stop), np.int32(length4), np.int32(len_x))
#		 cl.enqueue_copy(queue, vecX4, vecX4_buf)

#		 vecX4 = vecX4[:n_stop - n_start]

#		 # Calculate expValX4 on the CPU
#		 expValX4[tau1, tau2, tau3, tau4] = np.mean(vecX4 ** 2)
#		 A4eff[tau1, tau2, tau3, tau4] = np.power(
#			 expValX4[tau1, tau2, tau3, tau4] / calcDiagonalFactor((tau1, tau2, tau3, tau4)), 1/4)

#		 vecY = y[n_start:n_stop]
#		 k_4[tau1, tau2, tau3, tau4] = (
#			 (np.sum(vecX4 * vecY) / (n_stop - n_start) -
#			  A4eff[tau1, tau2, tau3, tau4] ** 2 *
#			  (k2[tau1, tau2] * delta(tau3 - tau4) +
#			   k2[tau1, tau3] * delta(tau2 - tau4) +
#			   k2[tau1, tau4] * delta(tau2 - tau3) +
#			   k2[tau2, tau3] * delta(tau1 - tau4) +
#			   k2[tau2, tau4] * delta(tau1 - tau3) +
#			   k2[tau3, tau4] * delta(tau1 - tau2))
#			  ) / math.factorial(4) / A4eff[tau1, tau2, tau3, tau4] ** 4
#		 )

#		 perms = list(set(permutations([tau1, tau2, tau3, tau4])))
#		 for (t1, t2, t3, t4) in perms:
#			 k_4[t1, t2, t3, t4] = k_4[tau1, tau2, tau3, tau4]

#	 return k_4


# def multivarK5(x, y, length5, A, k1, k3):
#	 n_start = length5
#	 n_stop = len(x)
#	 k_5 = np.zeros((length5, length5, length5, length5, length5))
#	 expValX5 = np.zeros((length5, length5, length5, length5, length5))
#	 A5eff = np.zeros((length5, length5, length5, length5, length5))

#	 len_x = len(x)
#	 vecX5_length = n_stop - n_start
#	 vecX5 = np.zeros(vecX5_length, dtype=np.float32)
#	 x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
#					   hostbuf=x.astype(np.float32))
#	 vecX5_buf = cl.Buffer(ctx, mf.WRITE_ONLY, vecX5.nbytes)
#	 indices_buf = cl.Buffer(
#		 ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(5, dtype=np.int32))

#	 for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
#		 indices = np.array([tau1, tau2, tau3, tau4, tau5], dtype=np.int32)
#		 cl.enqueue_copy(queue, indices_buf, indices)
#		 prg.compute_vecX5(queue, (vecX5_length,), None, x_buf, vecX5_buf, indices_buf, np.int32(
#			 n_start), np.int32(n_stop), np.int32(length5), np.int32(len_x))
#		 cl.enqueue_copy(queue, vecX5, vecX5_buf)
#		 vecX5 = vecX5[:n_stop - n_start]

#		 expValX5[tau1, tau2, tau3, tau4, tau5] = np.mean(vecX5 ** 2)
#		 A5eff[tau1, tau2, tau3, tau4, tau5] = np.power(
#			 expValX5[tau1, tau2, tau3, tau4, tau5] / calcDiagonalFactor((tau1, tau2, tau3, tau4, tau5)), 1/5)

#		 vecY = y[n_start:n_stop]
#		 k_5[tau1, tau2, tau3, tau4, tau5] = (
#			 np.sum(vecX5 * vecY) / (n_stop - n_start) -
#			 math.factorial(3) * A5eff[tau1, tau2, tau3, tau4, tau5]**4 * (
#				 k3[tau1, tau2, tau3] * delta(tau4 - tau5) +
#				 k3[tau1, tau2, tau4] * delta(tau3 - tau5) +
#				 k3[tau1, tau2, tau5] * delta(tau3 - tau4) +
#				 k3[tau1, tau3, tau4] * delta(tau2 - tau5) +
#				 k3[tau1, tau3, tau5] * delta(tau2 - tau4) +
#				 k3[tau1, tau4, tau5] * delta(tau2 - tau3) +
#				 k3[tau2, tau3, tau4] * delta(tau1 - tau5) +
#				 k3[tau2, tau3, tau5] * delta(tau1 - tau4) +
#				 k3[tau2, tau4, tau5] * delta(tau1 - tau3) +
#				 k3[tau3, tau4, tau5] * delta(tau1 - tau2)
#			 ) -
#			 A5eff[tau1, tau2, tau3, tau4, tau5]**3 * (
#				 k1[tau1] * delta4(tau2, tau3, tau4, tau5) +
#				 k1[tau2] * delta4(tau1, tau3, tau4, tau5) +
#				 k1[tau3] * delta4(tau1, tau2, tau4, tau5) +
#				 k1[tau4] * delta4(tau1, tau2, tau3, tau5) +
#				 k1[tau5] * delta4(tau1, tau2, tau3, tau4)
#			 )
#		 ) / math.factorial(5) / A5eff[tau1, tau2, tau3, tau4, tau5]**5

#		 perms = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
#		 for (t1, t2, t3, t4, t5) in perms:
#			 k_5[t1, t2, t3, t4, t5] = k_5[tau1, tau2, tau3, tau4, tau5]

#	 return k_5
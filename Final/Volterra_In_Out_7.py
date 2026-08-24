"""In this script the permutations are correct """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from itertools import combinations_with_replacement
import pyopencl as cl
import time
import json
import os
from datetime import datetime
from opencl_util import create_context_and_queue
import pandas as pd
import math
import scipy.signal as sg


from Volterra_Downsample import downsample
from Volterra_Downsample import oversample
from Volterra_Downsample import LPfilter
from Volterra_Downsample import HPfilter
from Volterra_Downsample import fineShift


from matplotlib.font_manager import FontProperties
import matplotlib.font_manager
# import Volterra_PlotSystem as Volterra_PlotSystem
# from Volterra_File_Operations import readKernels, writeKernels

# Volterra_In_Out_6.py
ctx, queue = create_context_and_queue(device_type='GPU', profiling=True)
mf = cl.mem_flags

kernels_code = """
//Auxiliary functions

//Multi-input min function
int min3(int a1, int a2, int a3) {
	return min(min(a1,a2),a3);
}
int min4(int a1, int a2, int a3, int a4) {
	return min(min3(a1,a2,a3),a4);
}
int min5(int a1, int a2, int a3, int a4, int a5) {
	return min(min4(a1,a2,a3,a4),a5);
}
int min6(int a1, int a2, int a3, int a4, int a5, int a6) {
	return min(min5(a1,a2,a3,a4,a5),a6);
}

//Multi-input max function
int max3(int a1, int a2, int a3) {
	return max(max(a1,a2),a3);
}
int max4(int a1, int a2, int a3, int a4) {
	return max(max3(a1,a2,a3),a4);
}
int max5(int a1, int a2, int a3, int a4, int a5) {
	return max(max4(a1,a2,a3,a4),a5);
}
int max6(int a1, int a2, int a3, int a4, int a5, int a6) {
	return max(max5(a1,a2,a3,a4,a5),a6);
}

/*
//convert negative tau to positive (wraparound) tau
int pos_idx(int tau, int lTotal){
	return tau+(tau<0)*lTotal;
}
*/

//convert negative tau to positive (wraparound) tau
int pos_idx(int tau, int lTotal){
	return tau;
}


//take in taus, and return corresponding array index
int index1(int tau1,int l1Total){
	return(pos_idx(tau1,l1Total));
}
int index2(int tau1,int tau2,int l2Total){
	return(pos_idx(tau1,l2Total)*l2Total + pos_idx(tau2,l2Total));
}
int index3(int tau1,int tau2,int tau3,int l3Total){
	return((pos_idx(tau1,l3Total) * l3Total + pos_idx(tau2,l3Total)) * l3Total + pos_idx(tau3,l3Total));
}
int index4(int tau1,int tau2,int tau3,int tau4,int l4Total){
	return(((pos_idx(tau1,l4Total) * l4Total + pos_idx(tau2,l4Total)) * l4Total + pos_idx(tau3,l4Total)) *l4Total + pos_idx(tau4,l4Total));
}
int index5(int tau1,int tau2,int tau3,int tau4,int tau5,int l5Total){
	return((((pos_idx(tau1,l5Total) * l5Total + pos_idx(tau2,l5Total)) * l5Total + pos_idx(tau3,l5Total)) *l5Total + pos_idx(tau4,l5Total))*l5Total +pos_idx(tau5,l5Total));
}



__kernel void volterraH1(
	__global float *x,
	__global float *h1,
	__global float *H1x,
	uint len_x,
	int l1Total,
	int neg_len,
	uint oversampleFactor	
) {
	int index = get_global_id(0);
	if (index >= len_x) return;

	float sum = 0.0f;
	for (int tau1 = -neg_len; tau1 < l1Total - neg_len; tau1++) {
		int offset1 = l1Total - tau1 * oversampleFactor;	

		if (index + offset1 < len_x + l1Total * oversampleFactor) {	
			float h1_val = h1[index1(tau1+neg_len, l1Total)];
			float x_val1 = x[index + offset1];

			sum += h1_val * x_val1;
		}
	}
	H1x[index] = sum;
}


__kernel void volterraH2(
	__global float *x,
	__global float *h2,
	__global float *H2x,
	uint len_x,
	int l2Total,
	int neg_len,
	uint oversampleFactor	
) {
	int index = get_global_id(0);
	if (index >= len_x) return;

	float sum = 0.0f;
	for (int tau1 = -neg_len; tau1 < l2Total - neg_len; tau1++) {
		for (int tau2 = tau1; tau2 < l2Total - neg_len; tau2++) {
			int offset1 = l2Total - tau1 * oversampleFactor;	
			int offset2 = l2Total - tau2 * oversampleFactor;	

			if ((index + offset1) < (len_x + l2Total * oversampleFactor) && (index + offset2) < (len_x + l2Total * oversampleFactor)) {	
				float h2_val = h2[index2(tau1+neg_len, tau2+neg_len, l2Total)];
				float x_val1 = x[index + offset1];
				float x_val2 = x[index + offset2];
				int num_permutations = (tau1 == tau2) ? 1 : 2;

				sum += num_permutations * h2_val * x_val1 * x_val2;
			}
		}
	}
	H2x[index] = sum;
}

__kernel void volterraH3(
	__global float *x,
	__global float *h3,
	__global float *H3x,
	uint len_x,
	int l3Total,
	int neg_len,
	uint oversampleFactor	
) {
	int index = get_global_id(0);
	if (index >= len_x) return;

	float sum = 0.0f;
	for (int tau1 = -neg_len; tau1 < l3Total - neg_len; tau1++) {
		for (int tau2 = tau1; tau2 < l3Total - neg_len; tau2++) {
			for (int tau3 = tau2; tau3 < l3Total - neg_len; tau3++) {
				int offset1 = l3Total - tau1 * oversampleFactor;	
				int offset2 = l3Total - tau2 * oversampleFactor;	
				int offset3 = l3Total - tau3 * oversampleFactor;	

				if (index + offset1 < len_x + l3Total * oversampleFactor && index + offset2 < len_x + l3Total * oversampleFactor && index + offset3 < len_x + l3Total * oversampleFactor) {	
					float h3_val = h3[index3(tau1+neg_len, tau2+neg_len, tau3+neg_len, l3Total)];
					float x_val1 = x[index + offset1];
					float x_val2 = x[index + offset2];
					float x_val3 = x[index + offset3];
					int num_permutations = (tau1 == tau2 && tau2 == tau3) ? 1 : ((tau1 == tau2 || tau2 == tau3) ? 3 : 6);

					sum += num_permutations * h3_val * x_val1 * x_val2 * x_val3;
				}
			}
		}
	}
	H3x[index] = sum;
}


__kernel void volterraH4(
	__global float *x,
	__global float *h4,
	__global float *H4x,
	uint len_x,
	int l4Total,
	int neg_len,
	uint oversampleFactor	
) {
	int index = get_global_id(0);
	if (index >= len_x) return;

	float sum = 0.0f;
	int len4 = l4Total;
	int x_len_padded = len_x + 2 * len4 * oversampleFactor;	

	for (int tau1 = -neg_len; tau1 < l4Total - neg_len; tau1++) {
		for (int tau2 = tau1; tau2 < l4Total - neg_len; tau2++) {
			for (int tau3 = tau2; tau3 < l4Total - neg_len; tau3++) {
				for (int tau4 = tau3; tau4 < l4Total - neg_len; tau4++) {
					int offset1 = len4 - tau1 * oversampleFactor;	
					int offset2 = len4 - tau2 * oversampleFactor;	
					int offset3 = len4 - tau3 * oversampleFactor;	
					int offset4 = len4 - tau4 * oversampleFactor;	

					if (index + offset1 < x_len_padded && index + offset2 < x_len_padded && index + offset3 < x_len_padded && index + offset4 < x_len_padded) {	
						float h4_val = h4[index4(tau1+neg_len, tau2+neg_len, tau3+neg_len, tau4+neg_len, l4Total)];
						float x_val1 = x[index + offset1];
						float x_val2 = x[index + offset2];
						float x_val3 = x[index + offset3];
						float x_val4 = x[index + offset4];

						int num_permutations = 24;
						if ((tau1 == tau2) && (tau2 == tau3) && (tau3 == tau4)) {
							num_permutations = 1;
						} else if (((tau1 == tau2) && (tau2 == tau3)) || ((tau2 == tau3) && (tau3 == tau4)) || ((tau1 == tau3) && (tau3 == tau4)) || ((tau1 == tau2) && (tau2 == tau4))) {
							num_permutations = 4;
						} else if (((tau1 == tau2) && (tau3 == tau4)) || ((tau1 == tau3) && (tau2 == tau4)) || ((tau1 == tau4) && (tau2 == tau3))) {
							num_permutations = 6;
						} else if ((tau1 == tau2) || (tau1 == tau3) || (tau1 == tau4) || (tau2 == tau3) || (tau2 == tau4) || (tau3 == tau4)) {
							num_permutations = 12;
						}

						sum += num_permutations * h4_val * x_val1 * x_val2 * x_val3 * x_val4;
					}
				}
			}
		}
	}
	H4x[index] = sum;
}


__kernel void volterraH5(
	__global float *x,
	__global float *h5,
	__global float *H5x,
	uint len_x,
	int l5Total,
	int neg_len,
	uint oversampleFactor	
) {
	int index = get_global_id(0);
	if (index >= len_x) return;

	float sum = 0.0f;
	int len5 = l5Total;
	int x_len_padded = len_x + 2 * len5 * oversampleFactor;	

	for (int tau1 = -neg_len; tau1 < l5Total - neg_len; tau1++) {
		for (int tau2 = tau1; tau2 < l5Total - neg_len; tau2++) {
			for (int tau3 = tau2; tau3 < l5Total - neg_len; tau3++) {
				for (int tau4 = tau3; tau4 < l5Total - neg_len; tau4++) {
					for (int tau5 = tau4; tau5 < l5Total - neg_len; tau5++) {
						int offset1 = len5 - tau1 * oversampleFactor;	
						int offset2 = len5 - tau2 * oversampleFactor;	
						int offset3 = len5 - tau3 * oversampleFactor;	
						int offset4 = len5 - tau4 * oversampleFactor;	
						int offset5 = len5 - tau5 * oversampleFactor;	

						if (index + offset1 < x_len_padded && index + offset2 < x_len_padded && index + offset3 < x_len_padded && index + offset4 < x_len_padded && index + offset5 < x_len_padded) {	
							float h5_val = h5[index5(tau1+neg_len, tau2+neg_len, tau3+neg_len, tau4+neg_len, tau5+neg_len, l5Total)];
							float x_val1 = x[index + offset1];
							float x_val2 = x[index + offset2];
							float x_val3 = x[index + offset3];
							float x_val4 = x[index + offset4];
							float x_val5 = x[index + offset5];

							int num_permutations = 120; // Default value for all indices distinct

							if (tau1 == tau2 && tau2 == tau3 && tau3 == tau4 && tau4 == tau5) {
								num_permutations = 1; // All five indices are equal
							} else if ((tau1 == tau2 && tau2 == tau3 && tau3 == tau4) || 
									(tau2 == tau3 && tau3 == tau4 && tau4 == tau5) || 
									(tau1 == tau2 && tau2 == tau3 && tau3 == tau5) || 
									(tau1 == tau2 && tau2 == tau4 && tau4 == tau5) || 
									(tau1 == tau3 && tau3 == tau4 && tau4 == tau5)) {
								num_permutations = 5; // Four indices are equal
							} else if ((tau1 == tau2 && tau2 == tau3 && tau4 == tau5) || 
									(tau1 == tau2 && tau3 == tau4 && tau4 == tau5) || 
									(tau1 == tau3 && tau2 == tau4 && tau4 == tau5) || 
									(tau1 == tau2 && tau3 == tau5 && tau4 == tau5) || 
									(tau1 == tau3 && tau2 == tau5 && tau4 == tau5) || 
									(tau1 == tau4 && tau2 == tau3 && tau3 == tau5) || 
									(tau1 == tau4 && tau2 == tau5 && tau3 == tau5) || 
									(tau1 == tau5 && tau2 == tau3 && tau3 == tau4) || 
									(tau1 == tau5 && tau2 == tau4 && tau3 == tau4) || 
									(tau2 == tau3 && tau4 == tau5 && tau1 == tau4)) {
								num_permutations = 10; // Three indices are equal and the other two are equal
							} else if ((tau1 == tau2 && tau2 == tau3) || 
									(tau2 == tau3 && tau3 == tau4) || 
									(tau3 == tau4 && tau4 == tau5) || 
									(tau1 == tau2 && tau2 == tau4) || 
									(tau1 == tau3 && tau2 == tau4) || 
									(tau1 == tau3 && tau3 == tau4) || 
									(tau1 == tau4 && tau2 == tau3) || 
									(tau1 == tau4 && tau2 == tau5) || 
									(tau2 == tau4 && tau3 == tau5) || 
									(tau2 == tau4 && tau4 == tau5) || 
									(tau3 == tau4 && tau2 == tau5) || 
									(tau1 == tau2 && tau2 == tau5) || 
									(tau1 == tau3 && tau3 == tau5) || 
									(tau1 == tau4 && tau4 == tau5) || 
									(tau2 == tau3 && tau3 == tau5)) {
								num_permutations = 20; // Three indices are equal and the remaining two are distinct
							} else if ((tau1 == tau2 && tau3 == tau4) || 
									(tau1 == tau3 && tau2 == tau4) || 
									(tau1 == tau4 && tau2 == tau3) || 
									(tau1 == tau2 && tau3 == tau5) || 
									(tau1 == tau2 && tau4 == tau5) || 
									(tau1 == tau3 && tau2 == tau5) || 
									(tau1 == tau3 && tau4 == tau5) || 
									(tau1 == tau4 && tau2 == tau5) || 
									(tau1 == tau5 && tau2 == tau3) || 
									(tau1 == tau5 && tau3 == tau4) || 
									(tau1 == tau5 && tau4 == tau3) || 
									(tau2 == tau3 && tau4 == tau5) || 
									(tau2 == tau4 && tau3 == tau5)) {
								num_permutations = 30; // Two pairs of indices are equal and the remaining one is distinct
							} else if ((tau1 == tau2 && tau3 == tau4 && tau4 == tau5) || 
									(tau1 == tau3 && tau2 == tau4 && tau4 == tau5) || 
									(tau1 == tau4 && tau2 == tau3 && tau3 == tau5) || 
									(tau1 == tau5 && tau2 == tau3 && tau3 == tau4) || 
									(tau1 == tau2 && tau2 == tau5 && tau3 == tau4)) {
								num_permutations = 60; // Two pairs of indices are equal
							} else if ((tau1 == tau2) || (tau1 == tau3) || 
									(tau1 == tau4) || (tau1 == tau5) || 
									(tau2 == tau3) || (tau2 == tau4) || 
									(tau2 == tau5) || (tau3 == tau4) || 
									(tau3 == tau5) || (tau4 == tau5)) {
								num_permutations = 60; // One pair of indices is equal and the remaining three are distinct
							}

							sum += num_permutations * h5_val * x_val1 * x_val2 * x_val3 * x_val4 * x_val5;
						}
					}
				}
			}
		}
	}
	H5x[index] = sum;
}


"""
prg = cl.Program(ctx, kernels_code).build()


class VolterraResponse:
	def __init__(self, use_opencl, kernelName="DUT", kernelPath="", profiling_data=False):
		self.kernelName = kernelName
		self.kernelPath = kernelPath
		self.use_opencl = use_opencl
		self.profiling_data = profiling_data

		self.h0 = 0
		self.h1 = np.zeros(80)
		self.h2 = np.zeros(shape=(20, 20))
		self.h3 = np.zeros(shape=(20, 20, 20))
		self.h4 = np.zeros(shape=(20, 20, 20, 20))
		self.h5 = np.zeros(shape=(10, 10, 10, 10, 10))

		#Kernel length in the positive direction (causal)
		self.len1 = len(self.h1)
		self.len2 = len(self.h2)
		self.len3 = len(self.h3)
		self.len4 = len(self.h4)
		self.len5 = len(self.h5)
		
		#Kernel length in the negative direction (non-causal)
		self.nLen1=0
		self.nLen2=0
		self.nLen3=0
		self.nLen4=0
		self.nLen5=0

	def set_kernels(self, h0, h1, h2, h3, h4, h5, nlens=([0,0,0,0,0])):
		self.h0 = h0
		self.h1 = h1
		self.h2 = h2
		self.h3 = h3
		self.h4 = h4
		self.h5 = h5
		
		self.len1 = len(self.h1)-nlens[0]
		self.len2 = len(self.h2)-nlens[1]
		self.len3 = len(self.h3)-nlens[2]
		self.len4 = len(self.h4)-nlens[3]
		self.len5 = len(self.h5)-nlens[4]
		
		self.nLen1 = nlens[0]
		self.nLen2 = nlens[1]
		self.nLen3 = nlens[2]
		self.nLen4 = nlens[3]
		self.nLen5 = nlens[4]

	def H0(self, x, oversampleFactor=1):
		H0x = np.zeros(len(x))
		for n in np.arange(len(x)):
			H0x[n] = self.h0
		return H0x

	def H1(self, x, oversampleFactor=1, neg_len=0):
		H1x = np.zeros(len(x))
		len1 = len(self.h1)
		x1 = np.concatenate((np.zeros(len1*oversampleFactor), x, np.zeros(len1*oversampleFactor)))

		
		for tau1 in range(-neg_len, len1 - neg_len):
			# print(tau1)
			# print(len(H1x))
			# print(len(x1[len1 - tau1*oversampleFactor: len(x) + len1 - tau1*oversampleFactor]))
			# print(str(len1 - tau1*oversampleFactor) +':'+ str(len(x) + len1 - tau1*oversampleFactor))
			H1x += self.h1[tau1+neg_len] * x1[(len1 - tau1)*oversampleFactor: len(x) + (len1 - tau1)*oversampleFactor]
		return H1x

	def H2(self, x, oversampleFactor=1, neg_len=0):
		H2x = np.zeros(len(x))
		len2 = len(self.h2)
		x2 = np.concatenate((np.zeros(len2*oversampleFactor), x, np.zeros(len2*oversampleFactor)))
		for tau1, tau2 in combinations_with_replacement(range(-neg_len, len2 - neg_len), 2):
			unique_indices = [tau1, tau2]
			num_permutations = np.math.factorial(
				2) // np.prod([np.math.factorial(unique_indices.count(tau)) for tau in set(unique_indices)])

			H2x += num_permutations * self.h2[tau1+neg_len, tau2+neg_len] * \
				x2[len2*oversampleFactor - tau1*oversampleFactor: len(x) + len2*oversampleFactor - tau1*oversampleFactor] * \
				x2[len2*oversampleFactor - tau2*oversampleFactor: len(x) + len2*oversampleFactor - tau2*oversampleFactor]

		return H2x

	def H3(self, x, oversampleFactor=1, neg_len=0):
		H3x = np.zeros(len(x))
		len3 = len(self.h3)
		x3 = np.concatenate((np.zeros(len3*oversampleFactor), x, np.zeros(len3*oversampleFactor)))

		for tau1, tau2, tau3 in combinations_with_replacement(range(-neg_len, len3 - neg_len), 3):
			unique_indices = [tau1, tau2, tau3]
			num_permutations = np.math.factorial(
				3) // np.prod([np.math.factorial(unique_indices.count(tau)) for tau in set(unique_indices)])

			H3x += num_permutations * self.h3[tau1+neg_len, tau2+neg_len, tau3+neg_len] * \
				x3[len3*oversampleFactor - tau1*oversampleFactor: len(x) + len3*oversampleFactor - tau1*oversampleFactor] * \
				x3[len3*oversampleFactor - tau2*oversampleFactor: len(x) + len3*oversampleFactor - tau2*oversampleFactor] * \
				x3[len3*oversampleFactor - tau3*oversampleFactor: len(x) + len3*oversampleFactor - tau3*oversampleFactor]

		return H3x

	def H4(self, x, oversampleFactor=1, neg_len=0):
		H4x = np.zeros(len(x))
		len4 = len(self.h4)
		x4 = np.concatenate((np.zeros(len4*oversampleFactor), x, np.zeros(len4*oversampleFactor)))

		for tau1, tau2, tau3, tau4 in combinations_with_replacement(range(-neg_len, len4 - neg_len), 4):
			unique_indices = [tau1, tau2, tau3, tau4]
			num_permutations = np.math.factorial(
				4) // np.prod([np.math.factorial(unique_indices.count(tau)) for tau in set(unique_indices)])

			H4x += num_permutations * self.h4[tau1+neg_len, tau2+neg_len, tau3+neg_len, tau4+neg_len] * \
				x4[len4*oversampleFactor - tau1*oversampleFactor: len(x) + len4*oversampleFactor - tau1*oversampleFactor] * \
				x4[len4*oversampleFactor - tau2*oversampleFactor: len(x) + len4*oversampleFactor - tau2*oversampleFactor] * \
				x4[len4*oversampleFactor - tau3*oversampleFactor: len(x) + len4*oversampleFactor - tau3*oversampleFactor] * \
				x4[len4*oversampleFactor - tau4*oversampleFactor: len(x) + len4*oversampleFactor - tau4*oversampleFactor]

		return H4x

	def H5(self, x, oversampleFactor=1, neg_len=0):
		H5x = np.zeros(len(x))
		len5 = len(self.h5)
		x5 = np.concatenate((np.zeros(len5*oversampleFactor), x, np.zeros(len5*oversampleFactor)))

		for tau1, tau2, tau3, tau4, tau5 in combinations_with_replacement(range(-neg_len, len5 - neg_len), 5):
			unique_indices = [tau1, tau2, tau3, tau4, tau5]
			num_permutations = np.math.factorial(
				5) // np.prod([np.math.factorial(unique_indices.count(tau)) for tau in set(unique_indices)])

			H5x += num_permutations * self.h5[tau1+neg_len, tau2+neg_len, tau3+neg_len, tau4+neg_len, tau5+neg_len] * \
				x5[len5*oversampleFactor - tau1*oversampleFactor: len(x) + len5*oversampleFactor - tau1*oversampleFactor] * \
				x5[len5*oversampleFactor - tau2*oversampleFactor: len(x) + len5*oversampleFactor - tau2*oversampleFactor] * \
				x5[len5*oversampleFactor - tau3*oversampleFactor: len(x) + len5*oversampleFactor - tau3*oversampleFactor] * \
				x5[len5*oversampleFactor - tau4*oversampleFactor: len(x) + len5*oversampleFactor - tau4*oversampleFactor] * \
				x5[len5*oversampleFactor - tau5*oversampleFactor: len(x) + len5*oversampleFactor - tau5*oversampleFactor]

		return H5x
	def H1_opencl(self, x, oversampleFactor=1, neg_len=0):
		len_x = len(x)
		len_h1 = len(self.h1)
		x1 = np.concatenate(
			(np.zeros(len_h1*oversampleFactor), x, np.zeros(len_h1*oversampleFactor))).astype(np.float32)
		result = np.zeros(len_x, dtype=np.float32)
		h1 = self.h1.flatten().astype(np.float32)

		# Create buffers
		x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x1)
		h1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h1)
		result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

		# Execute kernel
		event = prg.volterraH1(queue, (len_x,), None, x_buf, h1_buf, result_buf, np.uint32(
			len_x), np.uint32(len_h1), np.int32(neg_len), np.uint32(oversampleFactor))   
		event.wait()
		cl.enqueue_copy(queue, result, result_buf, wait_for=[event])
		queue.finish()

		profiling_data = {
			'queuing_overhead (us)': round((event.profile.submit - event.profile.queued) * 1e-3, 10),
			'device_latency (us)': round((event.profile.start - event.profile.submit) * 1e-3, 10),
			'execution_time (s)': round((event.profile.end - event.profile.start) * 1e-9, 4)
		}

		return result, profiling_data

	def H2_opencl(self, x, oversampleFactor=1, neg_len=0):
		len_x = len(x)
		len_h2 = len(self.h2)
		x2 = np.concatenate(
			(np.zeros(len_h2 * oversampleFactor), x, np.zeros(len_h2 * oversampleFactor))).astype(np.float32)	
		result = np.zeros(len_x, dtype=np.float32)
		h2 = self.h2.flatten().astype(np.float32)

		# Create buffers
		x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2)
		h2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h2)
		result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

		# Execute kernel
		event = prg.volterraH2(queue, (len_x,), None, x_buf, h2_buf, result_buf, np.uint32(
			len_x), np.uint32(len_h2), np.int32(neg_len), np.uint32(oversampleFactor))   
		event.wait()
		cl.enqueue_copy(queue, result, result_buf, wait_for=[event])
		queue.finish()

		profiling_data = {
			'queuing_overhead (us)': round((event.profile.submit - event.profile.queued) * 1e-3, 10),
			'device_latency (us)': round((event.profile.start - event.profile.submit) * 1e-3, 10),
			'execution_time (s)': round((event.profile.end - event.profile.start) * 1e-9, 4)
		}

		return result, profiling_data
	
	def H3_opencl(self, x, oversampleFactor=1, neg_len=0):
		len_x = len(x)
		len_h3 = len(self.h3)
		x3 = np.concatenate(
			(np.zeros(len_h3 * oversampleFactor), x, np.zeros(len_h3 * oversampleFactor))).astype(np.float32)	
		result = np.zeros(len_x, dtype=np.float32)
		h3 = self.h3.flatten().astype(np.float32)

		# Create buffers
		x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x3)
		h3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h3)
		result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

		# Execute kernel
		event = prg.volterraH3(queue, (len_x,), None, x_buf, h3_buf, result_buf, np.uint32(
			len_x), np.uint32(len_h3), np.int32(neg_len), np.uint32(oversampleFactor))   
		event.wait()
		cl.enqueue_copy(queue, result, result_buf, wait_for=[event])
		queue.finish()

		profiling_data = {
			'queuing_overhead (us)': round((event.profile.submit - event.profile.queued) * 1e-3, 10),
			'device_latency (us)': round((event.profile.start - event.profile.submit) * 1e-3, 10),
			'execution_time (s)': round((event.profile.end - event.profile.start) * 1e-9, 4)
		}

		return result, profiling_data
	
	def H4_opencl(self, x, oversampleFactor=1, neg_len=0):
		len_x = len(x)
		len_h4 = len(self.h4)
		x4 = np.concatenate(
			(np.zeros(len_h4 * oversampleFactor), x, np.zeros(len_h4 * oversampleFactor))).astype(np.float32)	
		result = np.zeros(len_x, dtype=np.float32)
		h4 = self.h4.flatten().astype(np.float32)

		# Create buffers
		x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x4)
		h4_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h4)
		result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

		# Execute kernel
		event = prg.volterraH4(queue, (len_x,), None, x_buf, h4_buf, result_buf, np.uint32(
			len_x), np.uint32(len_h4), np.int32(neg_len), np.uint32(oversampleFactor))   
		event.wait()
		cl.enqueue_copy(queue, result, result_buf, wait_for=[event])
		queue.finish()

		profiling_data = {
			'queuing_overhead (us)': round((event.profile.submit - event.profile.queued) * 1e-3, 10),
			'device_latency (us)': round((event.profile.start - event.profile.submit) * 1e-3, 10),
			'execution_time (s)': round((event.profile.end - event.profile.start) * 1e-9, 4)
		}

		return result, profiling_data

	def H5_opencl(self, x, oversampleFactor=1, neg_len=0):
		len_x = len(x)
		len_h5 = len(self.h5)
		x5 = np.concatenate(
			(np.zeros(len_h5 * oversampleFactor), x, np.zeros(len_h5 * oversampleFactor))).astype(np.float32) 
		result = np.zeros(len_x, dtype=np.float32)
		h5 = self.h5.flatten().astype(np.float32)

		# Create buffers
		x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x5)
		h5_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h5)
		result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

		# Execute kernel
		event = prg.volterraH5(queue, (len_x,), None, x_buf, h5_buf, result_buf, np.uint32(
			len_x), np.uint32(len_h5), np.int32(neg_len), np.uint32(oversampleFactor))   
		event.wait()
		cl.enqueue_copy(queue, result, result_buf, wait_for=[event])
		queue.finish()

		profiling_data = {
			'queuing_overhead (us)': round((event.profile.submit - event.profile.queued) * 1e-3, 10),
			'device_latency (us)': round((event.profile.start - event.profile.submit) * 1e-3, 10),
			'execution_time (s)': round((event.profile.end - event.profile.start) * 1e-9, 4)
		}

		return result, profiling_data



	def H(self, x, oversampleFactor=1, neg_len=np.array([0,0,0,0,0])):
		if self.use_opencl:
			h1_res, h1_prof = self.H1_opencl(x, oversampleFactor, neg_len[0])
			h2_res, h2_prof = self.H2_opencl(x, oversampleFactor, neg_len[1])
			h3_res, h3_prof = self.H3_opencl(x, oversampleFactor, neg_len[2])
			h4_res, h4_prof = self.H4_opencl(x, oversampleFactor, neg_len[3])
			h5_res, h5_prof = self.H5_opencl(x, oversampleFactor, neg_len[4])

			Hx = h1_res + h2_res + h3_res + h4_res + h5_res

			profiling_data = {
				'H1': h1_prof,
				'H2': h2_prof,
				'H3': h3_prof,
				'H4': h4_prof,
				'H5': h5_prof,
			} if self.profiling_data else None
		else:
			Hx = (
				self.H1(x, oversampleFactor, neg_len[0])
				+ self.H2(x, oversampleFactor, neg_len[1])
				+ self.H3(x, oversampleFactor, neg_len[2])
				+ self.H4(x, oversampleFactor, neg_len[3])
				+ self.H5(x, oversampleFactor, neg_len[4])
			)
		if self.profiling_data:
			return Hx, profiling_data
		else:
			return Hx

	def Volterra_In_Out(self, x, oversampleFactor=1, neg_len=np.array([0,0,0,0,0])):
		result = self.H(x, oversampleFactor, neg_len)
		if self.profiling_data:
			y, profiling_data = result
			return y, profiling_data
		else:
			y = result
			return y


############################################################################################################

# Function to update the report file


def update_report(report_file, data):
	# Get the directory of the current script
	script_dir = os.path.dirname(os.path.realpath(__file__))

	# Create the full path to the report file
	report_file_path = os.path.join(script_dir, report_file)

	# Initialize an empty report
	report = []

	# Check if the report file exists and is not empty
	if os.path.exists(report_file_path) and os.path.getsize(report_file_path) > 0:
		try:
			with open(report_file_path, 'r') as file:
				report = json.load(file)
		except json.JSONDecodeError as e:
			print(f"Error reading JSON file: {e}")
			print("Creating a new report file.")

	# Append the new data
	report.append(data)

	# Write the updated report to the file
	with open(report_file_path, 'w') as file:
		json.dump(report, file, indent=2, separators=(',', ': '))

############################################################################################################


if (1):
	if __name__ == '__main__':
		from Volterra_File_Operations import readKernels, writeKernels
		import Volterra_PlotSystem as Volterra_PlotSystem

		system = VolterraResponse(kernelName='system', kernelPath='', use_opencl=True, profiling_data=False)
		#system = readKernels(name='overall', folderName='/thru_corr')
		system = readKernels(name='G', folderName='')
		system.use_opencl=False
		#print(system.h2)
		order = 5
		inptBandwidth = 500e6

		Amplitude = -15
		timestep = 1/(2*inptBandwidth*order)


		# Volterra_PlotSystem.plotOverview(system, Amplitude, order*timestep)

		# Volterra_PlotSystem.plot1(system, 0, system.len1)
		# Volterra_PlotSystem.plot2(system, 0, system.len2)
		# Volterra_PlotSystem.plot3(system, 0, system.len3)
		# Volterra_PlotSystem.interactive_plot_h4(system)
		# Volterra_PlotSystem.interactive_plot_h5(system)
		# #Volterra_PlotSystem.plot4_pca(system, 0, 10)
		# #Volterra_PlotSystem.plot5_pca(system, 0, 10)

		plt.show()

#-------------------------------------------------------

			
			
			
			
		#system=VolterraResponse(kernelName='DUT', use_opencl=True, kernelPath='')#kernelPath=''
		#system = readKernels(name='DUT', folderName='')
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
		

		inptlength=360*order#240*order
		
		



		#Generate an inpt array for testing the system. Select one of the test functions here------------------------------
		times=np.arange(inptlength)*timestep



		#RefAttenuations
		#8.25	for ADA4960
		#8.25 for ADL5565
		#0 for LMH5401
		#0 for LMH3401
		refAttenuation=20#dB, attenuation inequality between DUT and reference channel (aRef-aDUT)

		skew=0#-83#41.2[Samples] Skew output channel this much
		skewMismatch=0#-30e-12#[s] Skew mismatch between both measured curves. Happens due to inconsistent triggering

		ssAttenuation=20#dB, attenuation difference between both step responses
		
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

			#Impulse
			if(0):
				inpt=np.zeros(inptlength)
				inpt[int(inptlength/2)]=1.0*Amplitude
				
			#Step
			if(0):
				inpt[0]=0
				inpt[int(inptlength/2):]=np.ones(int(inptlength/2))*Amplitude
				
			#WGN		
			if(0):
				np.random.seed(98798)
				inpt=np.random.normal(loc=0.0, scale=Amplitude, size=inptlength)

			#Sinewave	
			if(0):
				for n in np.arange(100,len(inpt)-100):
					inpt[n]= 0.3*math.sin(2*np.pi*3*n/(128*order)) + 0.02*math.sin(2*np.pi*34.5*n/(128*order)) 

				
			#Other functions	
			#vals=np.array([0,0,0,0,0,0,0,0,0,0,0,1,-1,2,-2,3,-3,0,0,0,0,0,0,0,0,0,0,0,0])/3*Amplitude#Zoom=3
			vals=np.array([0,0,1,2,3,0,0,0,-1,-2,-3,0,0])/3*Amplitude#Zoom=1
			if(0):
				for n in np.arange(len(inpt)):
					#inpt[n]=Amplitude*np.sign(math.sin(2*np.pi*4*n/inptlength))#Squarewave
					#inpt[n]=Amplitude*math.sin(2*np.pi*3*n/inptlength)#Sinewave
					inpt[n]=vals[int(len(vals)*n/inptlength)]#Arbitrary


			#Random bits
			#Bit length in Volterra timesteps
			bitLength=20
			if(0):
				for n in np.arange(len(inpt)):
					if n%bitLength==0:
						inpt[n]=2*np.random.randint(2)*Amplitude - Amplitude
					else:
						inpt[n]=inpt[n-1]

					
			# inpt2=np.zeros(inptlength)
			# inpt2=inpt*ssFactor
		
			#Filter inpt data for better depiction
			if(0):
				#Apply Gaussian Filter to inpt data
				gaussian=sg.windows.gaussian(20,2.0,sym=True)#0.7 seems to be a good value for std deviation
				gaussian=gaussian/np.sum(gaussian)

				#Calculate filtered output
				inpt=np.convolve(inpt, gaussian, mode='same')
				inpt2=np.convolve(inpt2, gaussian, mode='same')
			
			if(0):
				#Apply LP Filter to inpt data
				inpt=np.pad(inpt, (1000, 1000), 'constant', constant_values=(0, 0))
				inpt2=np.pad(inpt2, (1000, 1000), 'constant', constant_values=(0, 0))
				
				inpt,_=LPfilter(inpt, times, fstop=500e6)
				inpt2,_=LPfilter(inpt2, times, fstop=500e6)
				
				#print(np.sum(inpt)*timestep)
				#print(np.sum(inpt)*timestep)
				
				inpt=inpt[1000:-1000]
				inpt2=inpt2[1000:-1000]
			
			
			#Import simulated LTSpice data
			if(1):
				#load csv data from LTSpice for comparison
				yMeasured=np.ones(inptlength)*1e-12
				yMeasured2=np.ones(inptlength)*1e-12
				
								
				
				Filename1='SPICE.csv'
				mdin='Volterra_input_Data'
					#File1	
				#inpt to the system
				df=pd.read_csv(mdin+'/'+Filename1, sep='\t', header=1, dtype=str)#Oscilloscope Data
				inpt=np.transpose(df.values[0:])[3].astype(float)#Start with 0th row, take 1st column
				inptTimes=np.transpose(df.values[0:])[0].astype(float)
				
				inpt2=inpt*ssFactor
				
				#Output of the system
				yMeasured=np.transpose(df.values[0:])[1].astype(float)#Start with 0th row, take 2nd column		
				yMeasured2=ssFactor*np.transpose(df.values[0:])[2].astype(float)#Start with 0th row, take 2nd column
				
				
				inpt=np.concatenate((np.zeros(401),inpt,np.zeros(375)))
				inpt2=np.concatenate((np.zeros(401),inpt2,np.zeros(375)))
				yMeasured=np.concatenate((np.zeros(401),yMeasured,np.zeros(375)))#388
				yMeasured2=np.concatenate((np.zeros(401),yMeasured2,np.zeros(375)))

				reference=yMeasured
				reference2=yMeasured2

				
				
			else:
				yMeasured=np.ones(inptlength)*1e-12
				yMeasured2=np.ones(inptlength)*1e-12

				
				reference=yMeasured
				reference2=yMeasured2
			
		#Measured System:
		#Read a measured trace for comparison, and use measured inpt as inpt to volterra system--------------------------------------------------------
		if(0):
			Filename1=Device+'step-6.csv'
			Filename2=Device+'step-10.csv'
			mdin='Volterra_input_Data'
			
		#File1	
			#inpt to the system
			df=pd.read_csv(mdin+'/'+Filename1, sep=',', header=None, dtype=str)#Oscilloscope Data
			inpt=np.transpose(df.values[0:])[1].astype(float)#Start with 0th row, take 1st column
			inptTimes=np.transpose(df.values[0:])[0].astype(float)
			#inpt2=inpt*ssFactor
			
			#Output of the system
			#df=pd.read_csv(mdin+'/'+Filename1, sep=',', header=None, dtype=str)#Oscilloscope Data
			yMeasured=np.transpose(df.values[0:])[2].astype(float)#Start with 0th row, take 2nd column		
			
		#File2
			#inpt to the system
			df=pd.read_csv(mdin+'/'+Filename2, sep=',', header=None, dtype=str)#Oscilloscope Data
			inpt2=np.transpose(df.values[0:])[1].astype(float)#Start with 0th row, take 1st column
			#inpt2[1:]=inpt2[:-1]
			
			
			#Output of the system
			yMeasured2=np.transpose(df.values[0:])[2].astype(float)#Start with 0th row, take 2nd column
			#yMeasured2[1:]=yMeasured2[:-1]

			
			
			if(1):#Downsample
				fstop=inptBandwidth
			
				yMeasured,timesMeasured =downsample(yMeasured, inptTimes,fstop*order)
				yMeasured2,timesMeasured =downsample(yMeasured2, inptTimes,fstop*order)
				
				inpt,__ =downsample(inpt, inptTimes,fstop*order)
				inpt2,inptTimes =downsample(inpt2, inptTimes,fstop*order)


			#Remove attenuation difference
			inpt=inpt/refFactor
			inpt2=inpt2/refFactor
			

			#Remove mean
			yMeasured-=np.mean(yMeasured)
			yMeasured2-=np.mean(yMeasured2)
			
			inpt-=np.mean(inpt)
			inpt2-=np.mean(inpt2)
			
			
			
			_, inpt=fineShift(inptTimes, inpt, shiftSamples=0, shiftTime=skewMismatch)
			_, yMeasured=fineShift(inptTimes, yMeasured, shiftSamples=skew, shiftTime=skewMismatch)

			_, inpt2=fineShift(inptTimes, inpt2, shiftSamples=0, shiftTime=0e-12)
			_, yMeasured2=fineShift(inptTimes, yMeasured2, shiftSamples=skew, shiftTime=0e-12)

			#Limit length of the measured inpt/output
			yMeasured=yMeasured[int(len(yMeasured)/2-inptlength/2):int(len(yMeasured)/2+inptlength/2)]
			yMeasured2=yMeasured2[int(len(yMeasured2)/2-inptlength/2):int(len(yMeasured2)/2+inptlength/2)]
			inpt=inpt[int(len(inpt)/2-inptlength/2):int(len(inpt)/2+inptlength/2)]
			inpt2=inpt2[int(len(inpt2)/2-inptlength/2):int(len(inpt2)/2+inptlength/2)]

			if(0):
				#Remove constant value
				yMeasured-=yMeasured[50]
				yMeasured2-=yMeasured2[50]
				
				inpt-=inpt[50]
				inpt2-=inpt2[50]
			
			# #Smoothen measured inpt/output
			if(0):
				gaussian=sg.windows.gaussian(21,0.55,sym=True)#0.55 seems to be a good value for std deviation
				#gaussian=np.array([0.5, 1, 0.5])
				gaussian=gaussian/np.sum(gaussian)
				
				yMeasured=np.convolve(yMeasured, gaussian, mode='same')#Smoothen it for better depiction
				yMeasured2=np.convolve(yMeasured2, gaussian, mode='same')#Smoothen it for better depiction
				
				inpt=np.convolve(inpt, gaussian, mode='same')
				inpt2=np.convolve(inpt2, gaussian, mode='same')
			

		reference=yMeasured*1
		reference2=yMeasured2*1



	#-------------------------------------
		#Calculate output of Volterra System
		system.use_opencl=False
		output=system.H(x=inpt, oversampleFactor=order, neg_len=([8,8,8,8,8]))
		output2=system.H(x=inpt2, oversampleFactor=order, neg_len=([8,8,8,8,8]))
		output21=system.H(x=inpt/100, oversampleFactor=order, neg_len=([8,8,8,8,8]))*100

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
		visualOversample=1
		
		inptlength = inptlength*visualOversample
		_, inpt=oversample(times, inpt, visualOversample)
		_, inpt2=oversample(times, inpt2, visualOversample)
		_, output=oversample(times, output, visualOversample)
		_, output2=oversample(times, output2, visualOversample)
		_, yMeasured = oversample(times, yMeasured, visualOversample)
		_, yMeasured2 = oversample(times, yMeasured2, visualOversample)
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
			indexShift=int(0*inptlength/zoomFactor)#float value determines shift to the left
			#1*
			indexZero=int(inptlength/2+indexShift)
			indexLow=int(inptlength/2-inptlength/2/zoomFactor + indexShift)
			indexHigh=int(inptlength/2+inptlength/2/zoomFactor + indexShift)

			


			fig, ax = plt.subplots(1, 1, figsize=(6,3), dpi=90, constrained_layout=True)#output 6,3; input 3,3
			
			fontproperties=font
			

			#ax.set_xticks(np.arange(0, 1+len(inpt), len(inpt)/4))
			#ax.set_yticks(np.arange(-0.6, 0.8, 0.2))
			#ax.set_ylim(-0.7, 0.7)

			markertype='o'
			#markertype=''
			
			
			if(0):#Input
				ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, inpt[indexLow:indexHigh], color='grey', linewidth=2.5, marker=markertype, markevery=visualOversample*order, ms=4)
				ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, inpt[indexLow:indexHigh], color='#81549b', linewidth=2.5, label='inpt, LS')
				
				#ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, inpt2[indexLow:indexHigh]/ssFactor, color='lightgrey', linewidth=2.5, marker=markertype, markevery=visualOversample*order, ms=4)
				#ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, inpt2[indexLow:indexHigh]/ssFactor, color='#5c3c6e', linewidth=2.5, label='inpt, SS')
			
			if(1):
				#Volterra outputs
				#ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, output[indexLow:indexHigh], color='lightgrey', linewidth=2.5, marker=markertype, markevery=visualOversample, ms=4)
				ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, output[indexLow:indexHigh], color='#7bc2f1', linewidth=2.5, label='Volterra LS', marker=markertype, markevery=visualOversample*order, ms=6)
				#ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, output2[indexLow:indexHigh]/ssFactor, color='lightgrey', linewidth=2.5, marker=markertype, markevery=visualOversample, ms=4)
				ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, output2[indexLow:indexHigh]/ssFactor, color='#f2a144', linewidth=2.5, label='Volterra ss', marker=markertype, markevery=visualOversample*order, ms=6)

			if(1):#Measured/SPICE Simulated
				
				ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, yMeasured[indexLow:indexHigh], color='#2d6cb3', linewidth=1.5, label='SPICE LS')
				ax.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, yMeasured2[indexLow:indexHigh]/ssFactor, color='#b3582d', linewidth=1.5, label='SPICE ss')

			# if(1):#Simulated 
				# ax.plot(times[indexLow:indexHigh]-times[indexZero], reference[indexLow:indexHigh], color='lightblue', linewidth=1.5, linestyle='--', label='Actual System, LS', marker=markertype, markevery=visualOversample)
				# ax.plot(times[indexLow:indexHigh]-times[indexZero], reference2[indexLow:indexHigh]/ssFactor, color='orange', linewidth=1.5, linestyle='--', label='Actual System, SS +'+str(ssAttenuation)+r'$\, dB$', marker=markertype, markevery=visualOversample)

			

			ax.set_xlabel(r'Time $(ns)$', fontproperties=font)
			#ax.set_ylabel(r'Voltage $(V)$', fontproperties=font)
			
			
			
		#----------------------------------------------	
			#Paper Plot - output
			handles, labels = plt.gca().get_legend_handles_labels()
			order = [2,0,3,1]
			ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left')

			
			ax.set_xlim((-3, 20))
			ax.set_ylim((-3.5, 2.75))
			ax.set(yticklabels=[])
			
			
			# Major ticks = base sample grid
			ax.xaxis.set_major_locator(MultipleLocator(2/visualOversample))

			# Minor ticks = oversampled grid
			#ax.xaxis.set_minor_locator(MultipleLocator(1/(visualOversample*order)))
			
			ax.grid(visible=True, axis='x', linestyle = 'dotted')
			ax.grid(visible=True, axis='y', linestyle = 'dotted')
			
			
			
			if(1):
				# inset Axes....
				x1, x2, y1, y2 = 13.25, 15, -1.4, -0.6  # subregion of the original image
				axins = ax.inset_axes(
					[0.8, 0.03, 0.185, 0.22],
					xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
				
				axins.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, output[indexLow:indexHigh], color='#7bc2f1', linewidth=2.5, label='Volterra LS', marker=markertype, markevery=visualOversample*order, ms=6)
				axins.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, output2[indexLow:indexHigh]/ssFactor, color='#f2a144', linewidth=2.5, label='Volterra ss', marker=markertype, markevery=visualOversample*order, ms=6)

				
				axins.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, yMeasured[indexLow:indexHigh], color='#2d6cb3', linewidth=1.5, label='SPICE LS')
				axins.plot((times[indexLow:indexHigh]-times[indexZero])/1e-9, yMeasured2[indexLow:indexHigh]/ssFactor, color='#b3582d', linewidth=1.5, label='SPICE ss')

					
				axins.tick_params(bottom=False, top=False, left=False, right=False)
				
				ax.indicate_inset_zoom(axins, edgecolor="black")
		#---------------------------------------------------
		
		
		
		
		# #input
			# # Major ticks = base sample grid
			# ax.xaxis.set_major_locator(MultipleLocator(2/visualOversample))

			# # Minor ticks = oversampled grid
			# #ax.xaxis.set_minor_locator(MultipleLocator(1/(visualOversample*order)))
			
			# ax.grid(visible=True, axis='x', linestyle = 'dotted')
			# ax.grid(visible=True, axis='y', linestyle = 'dotted')
		
			# ax.set_xlim((-5, 6.5))
			# ax.set_ylim((-3.5, 2.75))
			
			# ax.legend()


			
			
			plt.savefig('Plots/FigureOutput/Results_InOut.pdf', dpi=90, transparent=False, bbox_inches='tight')
			plt.show()



























#------------------------------------------


	if(0):
		A = 0.25
		t = np.arange(0, 10000) * 150e-12
		frequency = 10e6  # 300 MHz
		duty_cycle = 0.5  # 50% duty cycle

		T = 1 / frequency
		x = A * ((t % T) < (duty_cycle * T)).astype(float)
		x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')

		

		system = VolterraResponse(use_opencl=True)

		system = readKernels(name='DUT')
		system_out = system.Volterra_In_Out(x)
		system_out2 = 2*system.Volterra_In_Out(x/2)
		system_out3 = 10*system.Volterra_In_Out(x/10)
		plt.figure()
		plt.plot(t, x, label='inpt Square Wave')
		plt.plot(t, system_out, label='System Output')
		plt.plot(t, system_out2, label='System Output/2')
		plt.plot(t, system_out3, label='System Output/10')
		plt.xlabel('Time (s)')
		plt.ylabel('Amplitude')
		plt.legend()
		plt.title('Volterra System Response to Square Wave')
		plt.savefig('volterra_square_wave.png')
		plt.show()


# Experiment script to run simulations for different kernel lengths and inpt sizes
if (0):
	if __name__ == '__main__':
		report_file = "volterra_report.json"

		# Initialize Volterra system with random values
		np.random.seed(0)

		kernel_lengths = range(10, 110, 10)
		# 1000, 2000, 4000, ...
		inpt_lengths = [1000 * 2**i for i in range(0, 14)]

		for kernel_len in kernel_lengths:
			for inpt_len in inpt_lengths:
				# Create random kernels of the current length
				h0 = 1
				h1 = np.random.rand(kernel_len).astype(np.float32)
				h2 = np.random.rand(kernel_len, kernel_len).astype(np.float32)
				h3 = np.random.rand(kernel_len, kernel_len,
									kernel_len).astype(np.float32)
				h4 = np.random.rand(kernel_len, kernel_len,
									kernel_len, kernel_len).astype(np.float32)
				h5 = np.random.rand(
					kernel_len, kernel_len, kernel_len, kernel_len, kernel_len).astype(np.float32)

				# Create inpt signal of the current length
				# x = np.random.rand(inpt_len).astype(np.float32)
				x = np.sin(np.linspace(0, 2*np.pi, inpt_len)
						   ).astype(np.float32)

				# Create Volterra system instances for CPU and GPU
				volterra_cpu = VolterraResponse(use_opencl=False)
				volterra_gpu = VolterraResponse(use_opencl=True)

				volterra_cpu.set_kernels(h0, h1, h2, h3, h4, h5)
				volterra_gpu.set_kernels(h0, h1, h2, h3, h4, h5)

				# Measure computation time for CPU
				start_time_cpu = time.time()
				y_cpu = volterra_cpu.Volterra_In_Out(x)
				cpu_time = time.time() - start_time_cpu

				# Measure computation time for GPU
				start_time_gpu = time.time()
				y_gpu, profiling_data = volterra_gpu.Volterra_In_Out(x)
				gpu_time = time.time() - start_time_gpu

				diff = y_gpu - y_cpu

				# Calculate speedup
				speedup = cpu_time / gpu_time

				# Plot the output of the Volterra system for both CPU and GPU
				plt.figure(figsize=(12, 6))
				plt.plot(y_cpu, label='CPU Output', linestyle='--')
				plt.plot(y_gpu, label='GPU Output', linestyle='-.')
				plt.plot(diff, label='Difference', linestyle=':')
				plt.title(
					f'Volterra System Output Comparison\nKernel Length: {kernel_len}, inpt Length: {inpt_len}')
				plt.xlabel('Sample Index')
				plt.ylabel('Output')
				plt.legend()
				plt.grid(True)

				# Save the plot as a PNG file
				plot_filename = f'volterra_output_kernel_{kernel_len}_inpt_{inpt_len}.png'
				plt.savefig(plot_filename)

				# Display the plot
				plt.show()
				plt.close()

				# Prepare data for report
				report_data = {
					"timestamp": datetime.now().isoformat(),
					"inpt_size": len(x),
					"kernel_lengths": {
						"h1": len(h1),
						"h2": h2.shape,
						"h3": h3.shape,
						"h4": h4.shape,
						"h5": h5.shape,
					},
					"cpu_time": cpu_time,
					"gpu_time": gpu_time,
					"speedup": speedup,
					"gpu_profiling_data": profiling_data,
				}

				# Update the report file
				update_report(report_file, report_data)

				# Print computation times
				print(
					f"Kernel Length: {kernel_len}, inpt Length: {inpt_len}")
				print(f"CPU computation time: {cpu_time:.6f} seconds")
				print(f"GPU computation time: {gpu_time:.6f} seconds")
				print(f"Speedup: {speedup:.2f}x")

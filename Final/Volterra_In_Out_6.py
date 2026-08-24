"""In this script the permutations are correct """
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import pyopencl as cl
import time
import json
import os
from datetime import datetime
from opencl_util import create_context_and_queue
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
			H1x += (self.h1[tau1+neg_len] * x1[(len1 - tau1)*oversampleFactor: len(x) + (len1 - tau1)*oversampleFactor])

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
			(np.zeros(len_h1), x, np.zeros(len_h1))).astype(np.float32)
		result = np.zeros(len_x, dtype=np.float32)
		h1 = self.h1.flatten().astype(np.float32)

		# Create buffers
		x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x1)
		h1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h1)
		result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

		# Execute kernel
		event = prg.volterraH1(queue, (len_x,), None, x_buf, h1_buf, result_buf, np.uint32(
			len_x), np.int32(len_h1), np.int32(neg_len), np.uint32(oversampleFactor))   
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
			len_x), np.int32(len_h2), np.int32(neg_len), np.uint32(oversampleFactor))   
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
			len_x), np.int32(len_h3), np.int32(neg_len), np.uint32(oversampleFactor))   
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
			len_x), np.int32(len_h4), np.int32(neg_len), np.uint32(oversampleFactor))   
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
			len_x), np.int32(len_h5), np.int32(neg_len), np.uint32(oversampleFactor))   
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

		system = VolterraResponse(
			kernelName='system', kernelPath='', use_opencl=True, profiling_data=False)
		system = readKernels(name='BVD', folderName='')
		print(system.h2)
		order = 5
		inputBandwidth = 2.5e9

		Amplitude = 0.35
		timestep = 1/(2*inputBandwidth*order)

		# Volterra_PlotSystem.plotOverview(system, Amplitude, order*timestep)

		#Volterra_PlotSystem.plotOverview(system, Amplitude, order*timestep)

		#Volterra_PlotSystem.plot1(system, 0, system.len1)
		#Volterra_PlotSystem.plot2(system, 0, system.len2)
		#Volterra_PlotSystem.plot3(system, 0, system.len3)
		#Volterra_PlotSystem.interactive_plot_h4(system)
		#Volterra_PlotSystem.interactive_plot_h5(system)
		#Volterra_PlotSystem.plot4_pca(system, 0, 10)
		#Volterra_PlotSystem.plot5_pca(system, 0, 10)

		plt.show()

		A = 0.15
		t = np.arange(0, 100000) * 200e-12
		
		
		
		system = VolterraResponse(use_opencl=True)
		system = readKernels(name='BVD')
		
		freqs=np.linspace(0.7, 1.4, 100)*1e9
		
		system_out=np.zeros(len(freqs))
		system_out2=np.zeros(len(freqs))
		system_out3=np.zeros(len(freqs))
		
		for i in np.arange(len(freqs)):
		
			frequency = freqs[i]  # 300 MHz
			
			x=np.sin(2*np.pi*freqs[i]*t)*A
			
			index=np.argmax(np.abs(np.fft.fft(x[1000:])))#Find the FFT index of the test frequency. Remove first 1000 samples => steady state
			system_out[i] = np.abs(np.fft.fft(system.Volterra_In_Out(x)[1000:]))[index]
			system_out2[i] = np.abs(np.fft.fft(system.Volterra_In_Out(x*0.5)[1000:]/0.5))[index]
			system_out3[i] = np.abs(np.fft.fft(system.Volterra_In_Out(x*0.25)[1000:]/0.25))[index]
			
			
		plt.plot(freqs, system_out, label='A=0.2V')
		plt.plot(freqs, system_out2, label='A=0.1V')
		plt.plot(freqs, system_out3, label='A=0.05V')	
		plt.grid()
		plt.legend()
		plt.show()

		# T = 1 / frequency
		# x = A * ((t % T) < (duty_cycle * T)).astype(float)
		# x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')
		#x=np.convolve(x, ([0.1,0.2,0.4,0.2,0.1]), mode='same')


		# system_out = system.Volterra_In_Out(x)
		# system_out2 = 2*system.Volterra_In_Out(x/2)
		# system_out3 = 10*system.Volterra_In_Out(x/10)
		# plt.figure()
		# plt.plot(t, x, label='Input Square Wave')
		# plt.plot(t, system_out, label='System Output')
		# plt.plot(t, system_out2, label='System Output/2')
		# plt.plot(t, system_out3, label='System Output/10')
		# plt.xlabel('Time (s)')
		# plt.ylabel('Amplitude')
		# plt.legend()
		# plt.title('Volterra System Response to Square Wave')
		# plt.savefig('volterra_square_wave.png')
		# plt.show()


# Experiment script to run simulations for different kernel lengths and input sizes
if (0):
	if __name__ == '__main__':
		report_file = "volterra_report.json"

		# Initialize Volterra system with random values
		np.random.seed(0)

		kernel_lengths = range(10, 110, 10)
		# 1000, 2000, 4000, ...
		input_lengths = [1000 * 2**i for i in range(0, 14)]

		for kernel_len in kernel_lengths:
			for input_len in input_lengths:
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

				# Create input signal of the current length
				# x = np.random.rand(input_len).astype(np.float32)
				x = np.sin(np.linspace(0, 2*np.pi, input_len)
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
					f'Volterra System Output Comparison\nKernel Length: {kernel_len}, Input Length: {input_len}')
				plt.xlabel('Sample Index')
				plt.ylabel('Output')
				plt.legend()
				plt.grid(True)

				# Save the plot as a PNG file
				plot_filename = f'volterra_output_kernel_{kernel_len}_input_{input_len}.png'
				plt.savefig(plot_filename)

				# Display the plot
				plt.show()
				plt.close()

				# Prepare data for report
				report_data = {
					"timestamp": datetime.now().isoformat(),
					"input_size": len(x),
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
					f"Kernel Length: {kernel_len}, Input Length: {input_len}")
				print(f"CPU computation time: {cpu_time:.6f} seconds")
				print(f"GPU computation time: {gpu_time:.6f} seconds")
				print(f"Speedup: {speedup:.2f}x")

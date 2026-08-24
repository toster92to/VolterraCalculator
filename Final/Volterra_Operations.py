""" this file contains functions for Volterra operations, such as addition, subtraction, multiplication, and inversion of Volterra systems."""
from itertools import combinations_with_replacement, permutations, product
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, permutations
from Volterra_In_Out_6 import VolterraResponse
import copy
import pyopencl as cl
from opencl_util import create_context_and_queue


def addVolterraSystems(*volterra_systems):
	if not volterra_systems:
		raise ValueError("At least one Volterra system must be provided")

	combined_system = copy.deepcopy(volterra_systems[0])

	for system in volterra_systems[1:]:
		combined_system.h0 += system.h0
		combined_system.h1 += system.h1
		combined_system.h2 += system.h2
		combined_system.h3 += system.h3
		combined_system.h4 += system.h4
		combined_system.h5 += system.h5

	return combined_system


def subtractVolterraSystems(system1, system2):
	combined_system = copy.deepcopy(system1)

	combined_system.h0 -= system2.h0
	combined_system.h1 -= system2.h1
	combined_system.h2 -= system2.h2
	combined_system.h3 -= system2.h3
	combined_system.h4 -= system2.h4
	combined_system.h5 -= system2.h5

	return combined_system


def multiplyVolterraSystem(system, factor):
	combined_system = copy.deepcopy(system)

	combined_system.h0 *= factor
	combined_system.h1 *= factor
	combined_system.h2 *= factor
	combined_system.h3 *= factor
	combined_system.h4 *= factor
	combined_system.h5 *= factor

	return combined_system


def chainVolterraSystems(g, k, skew=0):
	res_sys = VolterraResponse(use_opencl=False, kernelName="resultSystem")

	length1 = len(g.h1)
	length2 = len(g.h2)
	length3 = len(g.h3)
	length4 = len(g.h4)
	length5 = len(g.h5)

	res_sys.h0 = k.h0 + g.h0*np.sum(k.h1) + g.h0**2*np.sum(k.h2) + g.h0**3 * \
		np.sum(k.h3) + g.h0**4*np.sum(k.h4) + g.h0**5*np.sum(k.h5)

	res_sys.h1 = np.zeros(length1)
	res_sys.h2 = np.zeros(shape=(length2, length2))
	res_sys.h3 = np.zeros(shape=(length3, length3, length3))
	res_sys.h4 = np.zeros(shape=(length4, length4, length4, length4))
	res_sys.h5 = np.zeros(shape=(length5, length5, length5, length5, length5))

	# order1. Tau is sigma in the paper
	for tau1 in np.arange(length1):
		temp = 0
		for v1 in np.arange(tau1+1):
			temp += k.h1[v1]*g.h1[tau1-v1]

		res_sys.h1[tau1] = temp

	# order2
	for tau1 in range(length2):
		for tau2 in range(length2):
			temp = 0
			for v1 in np.arange(min(tau1, tau2)+1):
				temp += k.h1[v1]*g.h2[tau1-v1, tau2-v1]

			for v1 in np.arange(tau1+1):
				for v2 in np.arange(tau2+1):
					temp += k.h2[v1, v2]*g.h1[tau1-v1]*g.h1[tau2-v2]

			res_sys.h2[tau1, tau2] = temp

	# order3
	for tau1 in range(length3):
		for tau2 in range(length3):
			for tau3 in range(length3):
				temp = 0

				for v1 in np.arange(min(tau1, tau2, tau3)+1):
					temp += k.h1[v1]*g.h3[tau1-v1, tau2-v1, tau3-v1]
					
				#added permutations
				for v1 in np.arange(tau1+1):
					for v2 in np.arange(min(tau2, tau3)+1):
						temp += k.h2[v1, v2]*g.h1[tau1-v1] * \
							g.h2[tau2-v2, tau3-v2]
				for v1 in np.arange(min(tau1,tau2)+1):
					for v2 in np.arange(tau3+1):			
						temp += k.h2[v1, v2]*g.h2[tau1-v1,tau2-v1] * \
							g.h1[tau3-v2]	

				for v1 in np.arange(tau1+1):
					for v2 in np.arange(tau2+1):
						for v3 in np.arange(tau3+1):
							temp += k.h3[v1, v2, v3]*g.h1[tau1-v1] * \
								g.h1[tau2-v2]*g.h1[tau3-v3]

				res_sys.h3[tau1, tau2, tau3] = temp

	# order4
	for tau1 in range(length4):
		for tau2 in range(length4):
			for tau3 in range(length4):
				for tau4 in range(length4):
					temp = 0
					for v1 in np.arange(min(tau1, tau2, tau3, tau4)+1):
						temp += k.h1[v1]*g.h4[tau1-v1,
											  tau2-v1, tau3-v1, tau4-v1]

					#added permutations
					for v1 in np.arange(tau1+1):
						for v2 in np.arange(min(tau2, tau3, tau4)+1):
							temp += k.h2[v1, v2]*g.h1[tau1-v1] * \
								g.h3[tau2-v2, tau3-v2, tau4-v2]
					for v1 in np.arange(min(tau1,tau2,tau3)+1):
						for v2 in np.arange(tau4+1):			
							temp += k.h2[v1, v2]*g.h3[tau1-v1,tau2-v1,tau3-v1] * \
								g.h1[tau4-v2]	

					for v1 in np.arange(min(tau1, tau2)+1):
						for v2 in np.arange(min(tau3, tau4)+1):
							temp += k.h2[v1, v2]*g.h2[tau1-v1, tau2-v1] * \
								g.h2[tau3-v2, tau4-v2]

					#added permutations
					for v1 in np.arange(tau1+1):
						for v2 in np.arange(tau2+1):
							for v3 in np.arange(min(tau3, tau4)+1):
								temp += k.h3[v1, v2, v3]*g.h1[tau1-v1] * \
									g.h1[tau2-v2]*g.h2[tau3-v3, tau4-v3]
					for v1 in np.arange(tau1+1):
						for v2 in np.arange(min(tau2,tau3)+1):
							for v3 in np.arange(tau4+1):				
								temp += k.h3[v1, v2, v3]*g.h1[tau1-v1] * \
									g.h2[tau2-v2,tau3-v2]*g.h1[tau4-v3]
					for v1 in np.arange(min(tau1,tau2)+1):
						for v2 in np.arange(tau3+1):
							for v3 in np.arange(tau4+1):				
								temp += k.h3[v1, v2, v3]*g.h2[tau1-v1,tau2-v1] * \
									g.h1[tau3-v2]*g.h1[tau4-v3]	

					for v1 in np.arange(tau1+1):
						for v2 in np.arange(tau2+1):
							for v3 in np.arange(tau3+1):
								for v4 in np.arange(tau4+1):
									temp += k.h4[v1, v2, v3, v4] * \
										g.h1[tau1-v1]*g.h1[tau2-v2] * \
										g.h1[tau3-v3]*g.h1[tau4-v4]

					res_sys.h4[tau1, tau2, tau3, tau4] = temp

	# order5
	for tau1 in range(length5):
		for tau2 in range(length5):
			for tau3 in range(length5):
				for tau4 in range(length5):
					for tau5 in range(length5):
						temp = 0
						for v1 in np.arange(min(tau1, tau2, tau3, tau4, tau5)+1):
							temp += k.h1[v1]*g.h5[tau1-v1, tau2 -
												  v1, tau3-v1, tau4-v1, tau5-v1]

						#added permutations
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(min(tau2, tau3, tau4, tau5)+1):
								temp += k.h2[v1, v2]*g.h1[tau1-v1] * \
									g.h4[tau2-v2, tau3-v2, tau4-v2, tau5-v2]
						for v1 in np.arange(min(tau1,tau2,tau3,tau4)+1):
							for v2 in np.arange(tau5+1):			
								temp += k.h2[v1, v2]*g.h4[tau1-v1,tau2-v1,tau3-v1,tau4-v1] * \
									g.h1[tau5-v2]

						#added permutations
						for v1 in np.arange(min(tau1,tau2)+1):
							for v2 in np.arange(min(tau3, tau4, tau5)+1):
								temp += k.h2[v1, v2]*g.h2[tau1-v1, tau2-v1] * \
									g.h3[tau3-v2, tau4-v2, tau5-v2]
						for v1 in np.arange(min(tau1,tau2,tau3)+1):
							for v2 in np.arange(min(tau4, tau5)+1):			
								temp += k.h2[v1, v2]*g.h3[tau1-v1, tau2-v1, tau3-v1] * \
									g.h2[tau4-v2, tau5-v2]

						#added permutations
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(tau2+1):
								for v3 in np.arange(min(tau3, tau4, tau5)+1):
									temp += k.h3[v1, v2, v3]*g.h1[tau1-v1] * \
										g.h1[tau2-v2] * \
										g.h3[tau3-v3, tau4-v3, tau5-v3]
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(min(tau2,tau3,tau4)+1):
								for v3 in np.arange(tau5+1):				
									temp += k.h3[v1, v2, v3]*g.h1[tau1-v1] * \
										g.h3[tau2-v2,tau3-v2,tau4-v2] * \
										g.h1[tau5-v3]
						for v1 in np.arange(min(tau1,tau2,tau3)+1):
							for v2 in np.arange(tau4+1):
								for v3 in np.arange(tau5+1):				
									temp += k.h3[v1, v2, v3]*g.h3[tau1-v1,tau2-v1,tau3-v1] * \
										g.h1[tau4-v2] * \
										g.h1[tau5-v3]

						#added permutations
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(min(tau2, tau3)+1):
								for v3 in np.arange(min(tau4, tau5)+1):
									temp += k.h3[v1, v2, v3]*g.h1[tau1-v1] * \
										g.h2[tau2-v2, tau3-v2] * \
										g.h2[tau4-v3, tau5-v3]
						for v1 in np.arange(min(tau1,tau2)+1):
							for v2 in np.arange(tau3+1):
								for v3 in np.arange(min(tau4, tau5)+1):				
									temp += k.h3[v1, v2, v3]*g.h2[tau1-v1,tau2-v1] * \
										g.h1[tau3-v2] * \
										g.h2[tau4-v3, tau5-v3]
						for v1 in np.arange(min(tau1,tau2)+1):
							for v2 in np.arange(min(tau3, tau4)+1):
								for v3 in np.arange(tau5+1):				
									temp += k.h3[v1, v2, v3]*g.h2[tau1-v1,tau2-v1] * \
										g.h2[tau3-v2, tau4-v2] * \
										g.h1[tau5-v3]										
										
						#added permutations				
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(tau2+1):
								for v3 in np.arange(tau3+1):
									for v4 in np.arange(min(tau4, tau5)+1):
										temp += k.h4[v1, v2, v3, v4]*g.h1[tau1-v1] * \
											g.h1[tau2-v2]*g.h1[tau3-v3] * \
											g.h2[tau4-v4, tau5-v4]
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(tau2+1):
								for v3 in np.arange(min(tau3,tau4)+1):
									for v4 in np.arange(tau5+1):					
										temp += k.h4[v1, v2, v3, v4]*g.h1[tau1-v1] * \
											g.h1[tau2-v2]*g.h2[tau3-v3,tau4-v3] * \
											g.h1[tau5-v4]
						for v1 in np.arange(tau1+1):
							for v2 in np.arange(min(tau2,tau3)+1):
								for v3 in np.arange(tau4+1):
									for v4 in np.arange(tau5+1):						
										temp += k.h4[v1, v2, v3, v4]*g.h1[tau1-v1] * \
											g.h2[tau2-v2,tau3-v2]*g.h1[tau4-v3] * \
											g.h1[tau5-v4]	
						for v1 in np.arange(min(tau1,tau2)+1):
							for v2 in np.arange(tau3+1):
								for v3 in np.arange(tau4+1):
									for v4 in np.arange(tau5+1):						
										temp += k.h4[v1, v2, v3, v4]*g.h2[tau1-v1,tau2-v1] * \
											g.h1[tau3-v2]*g.h1[tau4-v3] * \
											g.h1[tau5-v4]	

						for v1 in np.arange(tau1+1):
							for v2 in np.arange(tau2+1):
								for v3 in np.arange(tau3+1):
									for v4 in np.arange(tau4+1):
										for v5 in np.arange(tau5+1):
											temp += k.h5[v1, v2, v3, v4, v5] * \
												g.h1[tau1-v1]*g.h1[tau2-v2] * \
												g.h1[tau3-v3] * \
												g.h1[tau4-v4]*g.h1[tau5-v5]

						res_sys.h5[tau1, tau2, tau3, tau4, tau5] = temp

	# Calculate a skewed version of the output
	h0e = 0
	h1e = np.zeros(length1)
	h2e = np.zeros(shape=(length2, length2))
	h3e = np.zeros(shape=(length3, length3, length3))
	h4e = np.zeros(shape=(length4, length4, length4, length4))
	h5e = np.zeros(shape=(length5, length5, length5, length5, length5))

	h0e = res_sys.h0
	h1e[:length1-skew] = res_sys.h1[skew:]
	h2e[:length2-skew, :length2-skew] = res_sys.h2[skew:, skew:]
	h3e[:length3-skew, :length3-skew, :length3 -
		skew] = res_sys.h3[skew:, skew:, skew:]
	h4e[:length4-skew, :length4-skew, :length4-skew,
		:length4-skew] = res_sys.h4[skew:, skew:, skew:, skew:]
	h5e[:length5-skew, :length5-skew, :length5-skew, :length5 -
		skew, :length5-skew] = res_sys.h5[skew:, skew:, skew:, skew:, skew:]

	res_sys_skewed = VolterraResponse(kernelName="resultSystemSkewed")
	res_sys_skewed.set_kernels(h0e, h1e, h2e, h3e, h4e, h5e)

	return (res_sys_skewed)


kernel_code = """
__kernel void volterraH1(
	__global const float *g_h1, __global const float *k_h1, __global float *res_h1, const uint length1) {
	int index = get_global_id(0);
	if (index >= length1) return;

	float temp = 0.0f;
	for (int v1 = 0; v1 <= index; ++v1) {
		temp += k_h1[v1] * g_h1[index - v1];
	}
	res_h1[index] = temp;
}

__kernel void volterraH2(
	__global const float *g_h1, __global const float *g_h2, 
	__global const float *k_h1, __global const float *k_h2, 
	__global float *res_h2, const uint length2) {

	int index = get_global_id(0);
	if (index >= length2 * length2) return;

	int tau1 = index / length2;
	int tau2 = index % length2;

	float temp = 0.0f;

	for (int v1 = 0; v1 <= min(tau1, tau2); ++v1) {
		temp += k_h1[v1] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)];
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h1[tau1 - v1] * g_h1[tau2 - v2];
		}
	}

	res_h2[tau1 * length2 + tau2] = temp;
}


__kernel void volterraH1test(
	__global const float *g_h1, __global const float *k_h1, __global float *res_h1, const uint length1) {
	int index = get_global_id(0);
	if (index >= length1) return;
	
	int zero1 = 3;
	
	int tau1 = index - zero1;

	float temp = 0.0f;
	for (int v1 = -zero1; v1 <= tau1; ++v1) {
		temp += k_h1[v1] * g_h1[tau1 - v1];
	}
	res_h1[tau1] = temp;
}

__kernel void volterraH2test(
	__global const float *g_h1, __global const float *g_h2, 
	__global const float *k_h1, __global const float *k_h2, 
	__global float *res_h2, const uint length2) {

	int index = get_global_id(0);
	if (index >= length2 * length2) return;
	
	int zero1=3;
	int zero2=3;

	int tau1 = (index / length2) - zero2;
	int tau2 = (index % length2) - zero2;

	float temp = 0.0f;

	for (int v1 = -zero1; v1 <= min(tau1, tau2); ++v1) {
		temp += k_h1[v1] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)];
	}

	for (int v1 = -zero2; v1 <= tau1; ++v1) {
		for (int v2 = -zero2; v2 <= tau2; ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h1[tau1 - v1] * g_h1[tau2 - v2];
		}
	}

	res_h2[tau1 * length2 + tau2] = temp;
}


__kernel void volterraH3(
	__global const float *g_h1, __global const float *g_h2, __global const float *g_h3,
	__global const float *k_h1, __global const float *k_h2, __global const float *k_h3,
	__global float *res_h3, const uint length3, const uint length2) {

	int index = get_global_id(0);
	if (index >= length3 * length3 * length3) return;

	int tau1 = index / (length3 * length3);
	int tau2 = (index / length3) % length3;
	int tau3 = index % length3;

	float temp = 0.0f;

	for (int v1 = 0; v1 <= min(min(tau1, tau2), tau3); ++v1) {
		temp += k_h1[v1] * g_h3[((tau1 - v1) * length3 + (tau2 - v1)) * length3 + (tau3 - v1)];
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(tau2, tau3); ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h1[tau1 - v1] * g_h2[(tau2 - v2) * length2 + (tau3 - v2)];
		}
	}
	for (int v1 = 0; v1 <= min(tau1, tau2); ++v1) {
		for (int v2 = 0; v2 <= tau3; ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h1[tau3 - v2];
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= tau3; ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h1[tau3 - v3];
			}
		}
	}

	res_h3[(tau1 * length3 + tau2) * length3 + tau3] = temp;
}


__kernel void volterraH4(
	__global const float *g_h1, __global const float *g_h2, __global const float *g_h3, __global const float *g_h4,
	__global const float *k_h1, __global const float *k_h2, __global const float *k_h3, __global const float *k_h4,
	__global float *res_h4, const uint length4, const uint length3, const uint length2) {

	int index = get_global_id(0);
	if (index >= length4 * length4 * length4 * length4) return;

	int tau1 = index / (length4 * length4 * length4);
	int tau2 = (index / (length4 * length4)) % length4;
	int tau3 = (index / length4) % length4;
	int tau4 = index % length4;

	float temp = 0.0f;

	for (int v1 = 0; v1 <= min(min(min(tau1, tau2), tau3), tau4); ++v1) {
		temp += k_h1[v1] * g_h4[(((tau1 - v1) * length4 + (tau2 - v1)) * length4 + (tau3 - v1)) * length4 + (tau4 - v1)];
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(min(tau2, tau3), tau4); ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h1[tau1 - v1] * g_h3[((tau2 - v2) * length3 + (tau3 - v2)) * length3 + (tau4 - v2)];
		}
	}
	for (int v1 = 0; v1 <= min(min(tau1,tau2),tau3); ++v1) {
		for (int v2 = 0; v2 <= tau4; ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h3[((tau1 - v1) * length3 + (tau2 - v1)) * length3 + (tau3 - v1)] * g_h1[tau4 - v2];
		}
	}


	for (int v1 = 0; v1 <= min(tau1, tau2); ++v1) {
		for (int v2 = 0; v2 <= min(tau3, tau4); ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h2[(tau3 - v2) * length2 + (tau4 - v2)];
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= min(tau3, tau4); ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h2[(tau3 - v3) * length2 + (tau4 - v3)];
			}
		}
	}
	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(tau2,tau3); ++v2) {
			for (int v3 = 0; v3 <= tau4; ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h1[tau1 - v1] * g_h2[(tau2 - v2) * length2 + (tau3 - v2)] * g_h1[tau4 - v3];
			}
		}
	}
	for (int v1 = 0; v1 <= min(tau1,tau2); ++v1) {
		for (int v2 = 0; v2 <= tau3; ++v2) {
			for (int v3 = 0; v3 <= tau4; ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h1[tau3 - v2] * g_h1[tau4 - v3];
			}
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= tau3; ++v3) {
				for (int v4 = 0; v4 <= tau4; ++v4) {
					temp += k_h4[((v1 * length4 + v2) * length4 + v3) * length4 + v4] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h1[tau3 - v3] * g_h1[tau4 - v4];
				}
			}
		}
	}

	res_h4[((tau1 * length4 + tau2) * length4 + tau3) * length4 + tau4] = temp;
}

__kernel void volterraH5(
	__global const float *g_h1, __global const float *g_h2, __global const float *g_h3, __global const float *g_h4, __global const float *g_h5,
	__global const float *k_h1, __global const float *k_h2, __global const float *k_h3, __global const float *k_h4, __global const float *k_h5,
	__global float *res_h5, const uint length5, const uint length4, const uint length3, const uint length2) {

	int index = get_global_id(0);
	if (index >= length5 * length5 * length5 * length5 * length5) return;

	int tau1 = index / (length5 * length5 * length5 * length5);
	int tau2 = (index / (length5 * length5 * length5)) % length5;
	int tau3 = (index / (length5 * length5)) % length5;
	int tau4 = (index / length5) % length5;
	int tau5 = index % length5;

	float temp = 0.0f;

	for (int v1 = 0; v1 <= min(min(min(min(tau1, tau2), tau3), tau4), tau5); ++v1) {
		temp += k_h1[v1] * g_h5[((((tau1 - v1) * length5 + (tau2 - v1)) * length5 + (tau3 - v1)) * length5 + (tau4 - v1)) * length5 + (tau5 - v1)];
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(min(min(tau2, tau3), tau4), tau5); ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h1[tau1 - v1] * g_h4[(((tau2 - v2) * length4 + (tau3 - v2)) * length4 + (tau4 - v2)) * length4 + (tau5 - v2)];
		}
	}
	for (int v1 = 0; v1 <= min(min(min(tau1,tau2),tau3),tau4); ++v1) {
		for (int v2 = 0; v2 <= tau5; ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h4[(((tau1 - v1) * length4 + (tau2 - v1)) * length4 + (tau3 - v1)) * length4 + (tau4 - v1)] * g_h1[tau5 - v2];
		}
	}

	for (int v1 = 0; v1 <= min(tau1,tau2); ++v1) {
		for (int v2 = 0; v2 <= min(min(tau3, tau4), tau5); ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h3[((tau3 - v2) * length3 + (tau4 - v2)) * length3 + (tau5 - v2)];
		}
	}
	for (int v1 = 0; v1 <= min(min(tau1,tau2),tau3); ++v1) {
		for (int v2 = 0; v2 <= min(tau4, tau5); ++v2) {
			temp += k_h2[v1 * length2 + v2] * g_h3[((tau1 - v1) * length3 + (tau2 - v1)) * length3 + (tau3 - v1)] * g_h2[(tau4 - v2) * length2 + (tau5 - v2)];
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= min(min(tau3, tau4), tau5); ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h3[((tau3 - v3) * length3 + (tau4 - v3)) * length3 + (tau5 - v3)];
			}
		}
	}
	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(min(tau2, tau3), tau4); ++v2) {
			for (int v3 = 0; v3 <= tau5; ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h1[tau1 - v1] * g_h3[((tau2 - v2) * length3 + (tau3 - v2)) * length3 + (tau4 - v2)] * g_h1[tau5 - v3];
			}
		}
	}
	for (int v1 = 0; v1 <= min(min(tau1, tau2), tau3); ++v1) {
		for (int v2 = 0; v2 <= tau4; ++v2) {
			for (int v3 = 0; v3 <= tau5; ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h3[((tau1 - v1) * length3 + (tau2 - v1)) * length3 + (tau3 - v1)] * g_h1[tau4 - v2] * g_h1[tau5 - v3];
			}
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(tau2, tau3); ++v2) {
			for (int v3 = 0; v3 <= min(tau4, tau5); ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h1[tau1 - v1] * g_h2[(tau2 - v2) * length2 + (tau3 - v2)] * g_h2[(tau4 - v3) * length2 + (tau5 - v3)];
			}
		}
	}
	for (int v1 = 0; v1 <= min(tau1,tau2); ++v1) {
		for (int v2 = 0; v2 <= tau3; ++v2) {
			for (int v3 = 0; v3 <= min(tau4, tau5); ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h1[tau3 - v2] * g_h2[(tau4 - v3) * length2 + (tau5 - v3)];
			}
		}
	}
	for (int v1 = 0; v1 <= min(tau1, tau2); ++v1) {
		for (int v2 = 0; v2 <= min(tau3, tau4); ++v2) {
			for (int v3 = 0; v3 <= tau5; ++v3) {
				temp += k_h3[(v1 * length3 + v2) * length3 + v3] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h2[(tau3 - v2) * length2 + (tau4 - v2)] * g_h1[tau5 - v3];
			}
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= tau3; ++v3) {
				for (int v4 = 0; v4 <= min(tau4,tau5); ++v4) {
					temp += k_h4[((v1 * length4 + v2) * length4 + v3) * length4 + v4] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h1[tau3 - v3] * g_h2[(tau4 - v4) * length2 + (tau5 - v4)];
				}
			}
		}
	}
	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= min(tau3,tau4); ++v3) {
				for (int v4 = 0; v4 <= tau5; ++v4) {
					temp += k_h4[((v1 * length4 + v2) * length4 + v3) * length4 + v4] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h2[(tau3 - v3) * length2 + (tau4 - v3)] * g_h1[tau5 - v4];
				}
			}
		}
	}
	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= min(tau2,tau3); ++v2) {
			for (int v3 = 0; v3 <= tau4; ++v3) {
				for (int v4 = 0; v4 <= tau5; ++v4) {
					temp += k_h4[((v1 * length5 + v2) * length4 + v3) * length4 + v4] * g_h1[tau1 - v1] * g_h2[(tau2 - v2) * length2 + (tau3 - v2)] * g_h1[tau4 - v3] * g_h1[tau5 - v4];
				}
			}
		}
	}
	for (int v1 = 0; v1 <= min(tau1,tau2); ++v1) {
		for (int v2 = 0; v2 <= tau3; ++v2) {
			for (int v3 = 0; v3 <= tau4; ++v3) {
				for (int v4 = 0; v4 <= tau5; ++v4) {
					temp += k_h4[((v1 * length4 + v2) * length4 + v3) * length4 + v4] * g_h2[(tau1 - v1) * length2 + (tau2 - v1)] * g_h1[tau3 - v2] * g_h1[tau4 - v3] * g_h1[tau5 - v4];
				}
			}
		}
	}

	for (int v1 = 0; v1 <= tau1; ++v1) {
		for (int v2 = 0; v2 <= tau2; ++v2) {
			for (int v3 = 0; v3 <= tau3; ++v3) {
				for (int v4 = 0; v4 <= tau4; ++v4) {
					for (int v5 = 0; v5 <= tau5; ++v5) {
						temp += k_h5[(((v1 * length5 + v2) * length5 + v3) * length5 + v4) * length5 + v5] * g_h1[tau1 - v1] * g_h1[tau2 - v2] * g_h1[tau3 - v3] * g_h1[tau4 - v4] * g_h1[tau5 - v5];
					}
				}
			}
		}
	}

	res_h5[((((tau1 * length5 + tau2) * length5 + tau3) * length5 + tau4) * length5 + tau5)] = temp;
}

"""


def chainVolterraSystems_opencl(g, k, skew=0):
	res_sys = VolterraResponse(use_opencl=True, kernelName="resultSystem")

	ctx, queue = create_context_and_queue(device_type='GPU', profiling=True)
	mf = cl.mem_flags

	length1 = len(g.h1)
	length2 = len(g.h2)
	length3 = len(g.h3)
	length4 = len(g.h4)
	length5 = len(g.h5)
	
	res_sys.length1=length1
	res_sys.length2=length2
	res_sys.length3=length3
	res_sys.length4=length4
	res_sys.length5=length5
	
	
	# Flatten and convert data to float32
	g_h1 = g.h1.astype(np.float32)
	g_h2 = g.h2.flatten().astype(np.float32)
	g_h3 = g.h3.flatten().astype(np.float32)
	g_h4 = g.h4.flatten().astype(np.float32)
	g_h5 = g.h5.flatten().astype(np.float32)
	k_h1 = k.h1.astype(np.float32)
	k_h2 = k.h2.flatten().astype(np.float32)
	k_h3 = k.h3.flatten().astype(np.float32)
	k_h4 = k.h4.flatten().astype(np.float32)
	k_h5 = k.h5.flatten().astype(np.float32)
	res_h1 = np.zeros(length1, dtype=np.float32)
	res_h2 = np.zeros(length2 * length2, dtype=np.float32)
	res_h3 = np.zeros(length3 * length3 * length3, dtype=np.float32)
	res_h4 = np.zeros(length4 * length4 * length4 * length4, dtype=np.float32)
	res_h5 = np.zeros(length5 * length5 * length5 *
					  length5 * length5, dtype=np.float32)
		


	# Create buffers
	g_h1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_h1)
	g_h2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_h2)
	g_h3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_h3)
	g_h4_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_h4)
	g_h5_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_h5)
	k_h1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_h1)
	k_h2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_h2)
	k_h3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_h3)
	k_h4_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_h4)
	k_h5_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_h5)
	res_h1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res_h1.nbytes)
	res_h2_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res_h2.nbytes)
	res_h3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res_h3.nbytes)
	res_h4_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res_h4.nbytes)
	res_h5_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res_h5.nbytes)

	# Compile and execute the kernel
	prg = cl.Program(ctx, kernel_code).build()

	prg.volterraH1(queue, (length1,), None, g_h1_buf, k_h1_buf,
				   res_h1_buf, np.uint32(length1)).wait()
	prg.volterraH2(queue, (length2 * length2,), None, g_h1_buf, g_h2_buf,
				   k_h1_buf, k_h2_buf, res_h2_buf, np.uint32(length2)).wait()
	prg.volterraH3(queue, (length3 * length3 * length3,), None, g_h1_buf, g_h2_buf,
				   g_h3_buf, k_h1_buf, k_h2_buf, k_h3_buf, res_h3_buf, np.uint32(length3), np.uint32(length2)).wait()
	prg.volterraH4(queue, (length4 * length4 * length4 * length4,), None, g_h1_buf, g_h2_buf, g_h3_buf,
				   g_h4_buf, k_h1_buf, k_h2_buf, k_h3_buf, k_h4_buf, res_h4_buf, np.uint32(length4), np.uint32(length3), np.uint32(length2)).wait()
	prg.volterraH5(queue, (length5 * length5 * length5 * length5 * length5,), None, g_h1_buf, g_h2_buf, g_h3_buf,
				   g_h4_buf, g_h5_buf, k_h1_buf, k_h2_buf, k_h3_buf, k_h4_buf, k_h5_buf, res_h5_buf, np.uint32(length5), np.uint32(length4), np.uint32(length3), np.uint32(length2)).wait()

	# Copy results back to host
	cl.enqueue_copy(queue, res_h1, res_h1_buf).wait()
	cl.enqueue_copy(queue, res_h2, res_h2_buf).wait()
	cl.enqueue_copy(queue, res_h3, res_h3_buf).wait()
	cl.enqueue_copy(queue, res_h4, res_h4_buf).wait()
	cl.enqueue_copy(queue, res_h5, res_h5_buf).wait()

	# Assign results to res_sys

	res_sys.set_kernels(
		k.h0 + g.h0*np.sum(k.h1) + g.h0**2*np.sum(k.h2) + g.h0**3 * np.sum(k.h3) + g.h0**4*np.sum(k.h4) + g.h0**5*np.sum(k.h5),
		res_h1,
		res_h2.reshape((length2, length2)),
		res_h3.reshape((length3, length3, length3)),
		res_h4.reshape((length4, length4, length4, length4)),
		res_h5.reshape((length5, length5, length5, length5, length5))
	)


	# Calculate a skewed version of the output
	h0e = 0
	h1e = np.zeros(length1)
	h2e = np.zeros(shape=(length2, length2))
	h3e = np.zeros(shape=(length3, length3, length3))
	h4e = np.zeros(shape=(length4, length4, length4, length4))
	h5e = np.zeros(shape=(length5, length5, length5, length5, length5))
	
	

	h0e = res_sys.h0
	h1e[:length1-skew] = res_sys.h1[skew:]
	h2e[:length2-skew, :length2-skew] = res_sys.h2[skew:, skew:]
	h3e[:length3-skew, :length3-skew, :length3 -
		skew] = res_sys.h3[skew:, skew:, skew:]
	h4e[:length4-skew, :length4-skew, :length4-skew,
		:length4-skew] = res_sys.h4[skew:, skew:, skew:, skew:]
	h5e[:length5-skew, :length5-skew, :length5-skew, :length5 -
		skew, :length5-skew] = res_sys.h5[skew:, skew:, skew:, skew:, skew:]
		
	res_sys_skewed = VolterraResponse(kernelName="resultSystemSkewed", use_opencl=True)
	res_sys_skewed.set_kernels(h0e, h1e, h2e, h3e, h4e, h5e)

	# return (res_sys_skewed)
	result = make_symmetrical(res_sys_skewed)
	return (result)


# Similar to Zhu/Brazil, Behavioral Modeling of RF Power Amplifiers Based on Pruned Volterra Series (2004)
# Truncates the volterra series at strongly off-diagonal kernel entries.

def offDiagonalTruncation(input_system, distance):
	system = VolterraResponse(use_opencl=False, kernelName="truncatedSystem")
	system.h0 = input_system.h0
	system.h1 = input_system.h1
	system.h2 = input_system.h2
	system.h3 = input_system.h3
	system.h4 = input_system.h4
	system.h5 = input_system.h5

	length1 = len(system.h1)
	length2 = len(system.h2)
	length3 = len(system.h3)
	length4 = len(system.h4)
	length5 = len(system.h5)

	for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):
		if max(tau1, tau2)-min(tau1, tau2) > distance:
			system.h2[tau1, tau2] = 0

		perm = list(set(permutations([tau1, tau2])))
		for [t1, t2] in perm:
			system.h2[t1, t2] = system.h2[tau1, tau2]

	for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
		if max(tau1, tau2, tau3)-min(tau1, tau2, tau3) > distance:
			system.h3[tau1, tau2, tau3] = 0

		perm = list(set(permutations([tau1, tau2, tau3])))
		for [t1, t2, t3] in perm:
			system.h3[t1, t2, t3] = system.h3[tau1, tau2, tau3]

	for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(np.arange(length4), 4)):
		if max(tau1, tau2, tau3, tau4)-min(tau1, tau2, tau3, tau4) > distance:
			system.h4[tau1, tau2, tau3, tau4] = 0

		perm = list(set(permutations([tau1, tau2, tau3, tau4])))
		for [t1, t2, t3, t4] in perm:
			system.h4[t1, t2, t3, t4] = system.h4[tau1, tau2, tau3, tau4]

	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
		if max(tau1, tau2, tau3, tau4, tau5)-min(tau1, tau2, tau3, tau4, tau5) > distance:
			system.h5[tau1, tau2, tau3, tau4, tau5] = 0

		perm = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for [t1, t2, t3, t4, t5] in perm:
			system.h5[t1, t2, t3, t4, t5] = system.h5[tau1,
													  tau2, tau3, tau4, tau5]

		return system


# From M.Schetzen, Theory of pth-order inverses of nonlinear systems.
# The Kernels can be used for calculating the 3rd order inverse system.
# For the pth (p>>3) order inverse use Volterra_Inverse.py - this will not directly yield a system, however, but only its response

# Skew shifts the inverse response to the right. Can be used to make a non-causal system causal, at the cost of delay.

# DUT Skew:
# xm->y skew
# -
# xm->x skew
# +
# invertSkew

# Note that with bigger skews, nonlinear kernel values at the end of the array begin to get lost.

def invertKernels(inp_sys, use_opencl, skew=0):
	inv_sys = VolterraResponse(
		use_opencl=use_opencl, kernelName="inverse_system")


	H0 = VolterraResponse(kernelName="H0", use_opencl=use_opencl)
	H1 = VolterraResponse(kernelName="H1", use_opencl=use_opencl)
	H2 = VolterraResponse(kernelName="H2", use_opencl=use_opencl)
	H3 = VolterraResponse(kernelName="H3", use_opencl=use_opencl)
	H4 = VolterraResponse(kernelName="H4", use_opencl=use_opencl)
	H5 = VolterraResponse(kernelName="H5", use_opencl=use_opencl)
	I0 = VolterraResponse(kernelName="I0", use_opencl=use_opencl)
	I1 = VolterraResponse(kernelName="I1", use_opencl=use_opencl)
	I2 = VolterraResponse(kernelName="I2", use_opencl=use_opencl)
	I3 = VolterraResponse(kernelName="I3", use_opencl=use_opencl)
	I4 = VolterraResponse(kernelName="I4", use_opencl=use_opencl)
	I5 = VolterraResponse(kernelName="I5", use_opencl=use_opencl)

	length1 = len(inp_sys.h1)
	length2 = len(inp_sys.h2)
	length3 = len(inp_sys.h3)
	length4 = len(inp_sys.h4)
	length5 = len(inp_sys.h5)

	inv_sys.set_kernels(
		0, 
		np.zeros(length1), 
		np.zeros(shape=(length2, length2)), 
		np.zeros(shape=(length3, length3, length3)), 
		np.zeros(shape=(length4, length4, length4, length4)), 
		np.zeros(shape=(length5, length5, length5, length5, length5))
	)

		
	I0.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	I1.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	I2.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	I3.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	I4.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	I5.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))	

	H0.set_kernels(inp_sys.h0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	H1.set_kernels(0, inp_sys.h1, np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	H2.set_kernels(0, np.zeros(length1), inp_sys.h2, np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	H3.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), inp_sys.h3, np.zeros(
		shape=(length4, length4, length4, length4)), np.zeros(shape=(length5, length5, length5, length5, length5)))
	H4.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(
		length3, length3, length3)), inp_sys.h4, np.zeros(shape=(length5, length5, length5, length5, length5)))
	H5.set_kernels(0, np.zeros(length1), np.zeros(shape=(length2, length2)), np.zeros(shape=(length3, length3, length3)), np.zeros(
		shape=(length4, length4, length4, length4)), inp_sys.h5)


	# I0:------------------------------

	I0.h0 = -inp_sys.h0

	# I1:-----------------------------------------
	#FFT-based 1st order inversion, with 1000 padding length
	padding = 10000
	impulse = np.concatenate((np.zeros(padding), np.zeros(
		length1), np.zeros(length1-1), np.zeros(padding)))
	impulse[0] = 1
	h1con = np.concatenate((np.zeros(padding), np.zeros(
		length1-1), inp_sys.h1, np.zeros(padding)))
	I1.h1 = np.real(np.fft.ifft(np.fft.fft(impulse)/np.fft.fft(h1con)))

	if skew != 0:
		I1.h1 = I1.h1[padding+length1-skew:padding+2*length1-skew]
	else:
		I1.h1 = I1.h1[padding+length1:padding+2*length1]

	# I2:----------------------------------------
	#Calculate H'2
	Hp2=H2
	I2 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(chainVolterraSystems_opencl(I1, Hp2, skew), I1, skew))
	
	# I3:----------------------------------------
	Hp3=addVolterraSystems(H2, H3)
	I3 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(chainVolterraSystems_opencl(I2, Hp3, skew), I1, skew))

	# I4:----------------------------------------
	Hp4=addVolterraSystems(H2, H3, H4)
	I4 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(chainVolterraSystems_opencl(I3, Hp4, skew), I1, skew))
	
	# I5:----------------------------------------
	Hp5=addVolterraSystems(H2, H3, H4, H5)
	I5 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(chainVolterraSystems_opencl(I4, Hp5, skew), I1, skew))


	inv_sys	= I5
	return inv_sys


def make_symmetrical(volterra_system):
	symmetrical_system = copy.deepcopy(volterra_system)

	for order in range(1, 6):
		kernel = getattr(symmetrical_system, f'h{order}').astype(np.float64)
		permuted_kernels = [np.transpose(kernel, perm)
							for perm in permutations(range(order))]
		mean_kernel = np.mean(permuted_kernels, axis=0)
		setattr(symmetrical_system, f'h{order}', mean_kernel)

	return symmetrical_system


def is_symmetrical(volterra_system):
	for order in range(1, 6):
		kernel = getattr(volterra_system, f'h{order}')
		for i, j in combinations_with_replacement(range(order), 2):
			if np.any(kernel != np.swapaxes(kernel, i, j)):
				return False
	return True

# def make_symmetrical(volterra_system):
#	 symmetrical_system = copy.deepcopy(volterra_system)

#	 for order in range(1, 6):
#		 kernel = getattr(symmetrical_system, f'h{order}').astype(np.float64)
#		 for perm in permutations(range(order)):
#			 permuted_kernel = np.transpose(kernel, perm)
#			 if np.any(kernel != permuted_kernel):
#				 kernel = np.mean([kernel, permuted_kernel], axis=0)
#		 setattr(symmetrical_system, f'h{order}', kernel)

#	 return symmetrical_system

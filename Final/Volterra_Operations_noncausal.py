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


def chainVolterraSystems(g, k):
	res_sys = VolterraResponse(use_opencl=False, kernelName="resultSystem")

	length1 = g.len1
	length2 = g.len2
	length3 = g.len3
	length4 = g.len4
	length5 = g.len5

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


	return (res_sys)


kernel_code = """
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



//take in taus, and return corresponding array index

int index1(int tau1,int l1Total){
	return(tau1);
}
int index2(int tau1,int tau2,int l2Total){
	return(tau1 * l2Total + tau2);
}
int index3(int tau1,int tau2,int tau3,int l3Total){
	return((tau1 * l3Total + tau2) * l3Total + tau3);
}
int index4(int tau1,int tau2,int tau3,int tau4,int l4Total){
	return(((tau1 * l4Total + tau2) * l4Total + tau3) * l4Total + tau4);
}
int index5(int tau1,int tau2,int tau3,int tau4,int tau5,int l5Total){
	return((((tau1 * l5Total + tau2) * l5Total + tau3) * l5Total + tau4) * l5Total +  tau5);
}





//Kernel functions
__kernel void volterraH1(
	__global const float *g_h1, __global const float *k_h1, __global float *res_h1, const int l1, const int l1neg) {
	
	int l1Total=l1+l1neg;
	
	int index = get_global_id(0);
	if (index >= l1Total) return;
	
	private int tau1=index-l1neg;

	float temp = 0.0f;
	
	int lowv1=max(-l1neg,tau1-l1+1);
	int hghv1=min(l1, tau1+l1neg+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		temp += k_h1[index1(v1 + l1neg, l1Total)] * g_h1[index1(tau1 - v1 +l1neg, l1Total)];
	}
	res_h1[index1(tau1 + l1neg, l1Total)] = temp;
}

__kernel void volterraH2(
	__global const float *g_h1, __global const float *g_h2, 
	__global const float *k_h1, __global const float *k_h2, 
	__global float *res_h2, const int l1, const int l1neg, const int l2, const int l2neg) {
	
	int l1Total=l1+l1neg;
	int l2Total=l2+l2neg;
	
	private float g1=0;

	int index = get_global_id(0);
	if (index >= l2Total * l2Total) return;

	private int tau1 = index / l2Total - l2neg;
	private int tau2 = index % l2Total - l2neg;

	float temp = 0.0f;
	
	int lowv1= max3(-l1neg, tau1-l2+1, tau2-l2+1);
	int hghv1= min3(l1, tau1+l2neg+1, tau2+l2neg+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		temp += k_h1[index1(v1 + l1neg, l1Total)] * g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
	}

	//Pre-compute lower/upper bounds for v1,v2,...
	lowv1= max(-l2neg, tau1-l1+1);
	hghv1= min(l2, tau1+l1neg+1);
	int lowv2 = max(-l2neg, tau2-l1+1);
	int hghv2 = min(l2, tau2+l1neg+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h1[index1(tau2 - v2 + l1neg, l1Total)];
		}
	}

	res_h2[index2(tau1 + l2neg, tau2 + l2neg, l2Total)] = temp;
}


__kernel void volterraH3(
	__global const float *g_h1, __global const float *g_h2, __global const float *g_h3,
	__global const float *k_h1, __global const float *k_h2, __global const float *k_h3,
	__global float *res_h3, const int l1, const int l1neg, const int l2, const int l2neg, const int l3, const int l3neg) {

	int l1Total=l1+l1neg;
	int l2Total=l2+l2neg;
	int l3Total=l3+l3neg;
	
	private float g1=0;
	private float g2=0;
	
	int index = get_global_id(0);
	if (index >= l3Total * l3Total * l3Total) return;

	private int tau1 = index / (l3Total * l3Total) - l3neg;
	private int tau2 = (index / l3Total) % l3Total - l3neg;
	private int tau3 = index % l3Total - l3neg;

	float temp = 0.0f;

	int lowv1=max4(-l1neg, tau1-l3+1, tau2-l3+1, tau3-l3+1);
	int hghv1=min4(l1, l3neg+tau1+1, l3neg+tau2+1, l3neg+tau3+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		temp += k_h1[index1(v1 + l1neg, l1Total)] * g_h3[index3(tau1 - v1 + l3neg, tau2 - v1 + l3neg, tau3 - v1 + l3neg, l3Total)];
	}

	lowv1=max(-l2neg, tau1-l1+1);
	hghv1=min(l2, l1neg+tau1+1);
	int lowv2=max3(-l2neg, tau2-l2+1, tau3-l2+1);
	int hghv2=min3(l2, l2neg+tau2+1, l2neg+tau3+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h2[index2(tau2 - v2 + l2neg, tau3 - v2 + l2neg, l2Total)];
		}
	}
	lowv1= max3(-l2neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l2, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max(-l2neg, tau3-l1+1);
	hghv2= min(l2, l1neg+tau3+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h1[index1(tau3 - v2 + l1neg, l1Total)];
		}
	}

	lowv1= max(-l3neg, tau1-l1+1);
	hghv1= min(l3, l1neg+tau1+1);
	lowv2= max(-l3neg, tau2-l1+1); 
	hghv2= min(l3, l1neg+tau2+1);
	int lowv3= max(-l3neg, tau3-l1+1);
	int hghv3= min(l3, l1neg+tau3+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h1[index1(tau3 - v3 + l1neg, l1Total)];
			}
		}
	}

	res_h3[index3(tau1 + l3neg, tau2 + l3neg, tau3 + l3neg, l3Total)] = temp;
}


__kernel void volterraH4(
	__global const float *g_h1, __global const float *g_h2, __global const float *g_h3, __global const float *g_h4,
	__global const float *k_h1, __global const float *k_h2, __global const float *k_h3, __global const float *k_h4,
	__global float *res_h4, const int l1, const int l1neg, const int l2, const int l2neg, const int l3, const int l3neg, const int l4, const int l4neg) {

	int l1Total=l1+l1neg;
	int l2Total=l2+l2neg;
	int l3Total=l3+l3neg;
	int l4Total=l4+l4neg;
	
	private float g1=0;
	private float g2=0;
	private float g3=0;
	
	int index = get_global_id(0);
	if (index >= l4Total * l4Total * l4Total * l4Total) return;

	private int tau1 = index / (l4Total * l4Total * l4Total) -l4neg;
	private int tau2 = (index / (l4Total * l4Total)) % l4Total -l4neg;
	private int tau3 = (index / l4Total) % l4Total -l4neg;
	private int tau4 = index % l4Total - l4neg;

	float temp = 0.0f;
	
	int lowv1= max5(-l1neg, tau1-l4, tau2-l4, tau3-l4, tau4-l4);
	int hghv1= min5(l1, l4neg+tau1+1, l4neg+tau2+1, l4neg+tau3+1, l4neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		temp += k_h1[index1(v1 + l1neg, l1Total)] * g_h4[index4(tau1 - v1 + l4neg, tau2 - v1 + l4neg, tau3 - v1 + l4neg, tau4 - v1 + l4neg, l4Total)];
	}

	lowv1= max(-l2neg, tau1-l1+1);
	hghv1= min(l2, l1neg+tau1+1);
	int lowv2= max4(-l2neg, tau2-l3+1, tau3-l3+1, tau4-l3+1);
	int hghv2= min4(l2, l3neg+tau2+1, l3neg+tau3+1, l3neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h3[index3(tau2 - v2 + l3neg, tau3 - v2 + l3neg, tau4 - v2 + l3neg, l3Total)];
		}
	}
	lowv1= max4(-l2neg, tau1-l3+1,tau2-l3+1,tau3-l3+1);
	hghv1= min4(l2, l3neg+tau1+1, l3neg+tau2+1, l3neg+tau3+1);
	lowv2= max(-l2neg, tau4-l1+1); 
	hghv2= min(l2, l1neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h3[index3(tau1 - v1 + l3neg, tau2 - v1 + l3neg, tau3 - v1 + l3neg, l3Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h1[index1(tau4 - v2 + l1neg, l1Total)];
		}
	}

	lowv1= max3(-l2neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l2, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max3(-l2neg, tau3-l2+1, tau4-l2+1);
	hghv2= min3(l2, l2neg+tau3+1, l2neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h2[index2(tau3 - v2 + l2neg, tau4 - v2 + l2neg, l2Total)];
		}
	}

	lowv1= max(-l3neg, tau1-l1+1);
	hghv1= min(l3, l1neg+tau1+1);
	lowv2= max(-l3neg, tau2-l1+1);
	hghv2= min(l3, l1neg+tau2+1); 
	int lowv3= max3(-l3neg, tau3-l2+1, tau4-l2+1);
	int hghv3= min3(l3, l2neg+tau3+1, l2neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h2[index2(tau3 - v3 + l2neg, tau4 - v3 + l2neg, l2Total)];
			}
		}
	}
	lowv1= max(-l3neg, tau1-l1+1);
	hghv1= min(l3, l1neg+tau1+1);
	lowv2= max3(-l3neg, tau2-l2+1, tau3-l2+1);
	hghv2= min3(l3, l2neg+tau2+1, l2neg+tau3+1);
	lowv3= max(-l3neg, tau4-l1+1);
	hghv3= min(l3, l1neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h2[index2(tau2 - v2 + l2neg, tau3 - v2 + l2neg, l2Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h1[index1(tau4 - v3 + l1neg, l1Total)];
			}
		}
	}
	lowv1= max3(-l3neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l3, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max(-l3neg, tau3-l1+1);
	hghv2= min(l3, l1neg+tau3+1); 
	lowv3= max(-l3neg, tau4-l1+1);
	hghv3= min(l3, l1neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau3 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h1[index1(tau4 - v3 + l1neg, l1Total)];
			}
		}
	}

	lowv1= max(-l4neg, tau1-l1+1);
	hghv1= min(l4, l1neg+tau1+1);
	lowv2= max(-l4neg, tau2-l1+1);
	hghv2= min(l4, l1neg+tau2+1);
	lowv3= max(-l4neg, tau3-l1+1); 
	hghv3= min(l4, l1neg+tau3+1);
	int lowv4= max(-l4neg, tau4-l1+1);
	int hghv4= min(l4, l1neg+tau4+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				g3 = g_h1[index1(tau3 - v3 + l1neg, l1Total)];
				for (int v4 = lowv4; v4 <  hghv4; ++v4) {
					temp += k_h4[index4(v1 + l4neg, v2 + l4neg, v3 + l4neg, v4 + l4neg, l4Total)] * g1 * g2 * g3 * g_h1[index1(tau4 - v4 + l1neg, l1Total)];
				}
			}
		}
	}

	res_h4[index4(tau1 + l4neg, tau2 + l4neg, tau3 + l4neg, tau4 + l4neg, l4Total)] = temp;
}

__kernel void volterraH5(
	__global const float *g_h1, __global const float *g_h2, __global const float *g_h3, __global const float *g_h4, __global const float *g_h5,
	__global const float *k_h1, __global const float *k_h2, __global const float *k_h3, __global const float *k_h4, __global const float *k_h5,
	__global float *res_h5, const int l1, const int l1neg, const int l2, const int l2neg, const int l3, const int l3neg, const int l4, const int l4neg, const int l5, const int l5neg) {

	int l1Total=l1+l1neg;
	int l2Total=l2+l2neg;
	int l3Total=l3+l3neg;
	int l4Total=l4+l4neg;
	int l5Total=l5+l5neg;
	
	int index = get_global_id(0);
	if (index >= l5Total * l5Total * l5Total * l5Total * l5Total) return;

	private int tau1 = index / (l5Total * l5Total * l5Total * l5Total) - l5neg;
	private int tau2 = (index / (l5Total * l5Total * l5Total)) % l5Total - l5neg;
	private int tau3 = (index / (l5Total * l5Total)) % l5Total - l5neg;
	private int tau4 = (index / l5Total) % l5Total - l5neg;
	private int tau5 = index % l5Total - l5neg;
	
	private float g1=0;
	private float g2=0;
	private float g3=0;
	private float g4=0;

	float temp = 0.0f;

	int lowv1= max6(-l1neg, tau1-l5+1, tau2-l5+1, tau3-l5+1, tau4-l5+1, tau5-l5+1);
	int hghv1= min6(l1, l5neg+tau1+1, l5neg+tau2+1, l5neg+tau3+1, l5neg+tau4+1, l5neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		temp += k_h1[index1(v1 + l1neg,l1Total)] * g_h5[index5(tau1 - v1 + l5neg, tau2 - v1 + l5neg, tau3 - v1 + l5neg, tau4 - v1 + l5neg, tau5 - v1 + l5neg, l5Total)];
	}

	lowv1= max(-l2neg, tau1-l1+1);
	hghv1= min(l2, l1neg+tau1+1);
	int lowv2= max5(-l2neg, tau2-l4+1, tau3-l4+1, tau4-l4+1, tau5-l4+1); 
	int hghv2= min5(l2, l4neg+tau2+1, l4neg+tau3+1, l4neg+tau4+1, l4neg+tau5+1);
	for (int v1 = lowv1; v1 <  hghv1; ++v1) {
		g1=g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1*  g_h4[index4(tau2 - v2 + l4neg, tau3 - v2 + l4neg, tau4 - v2 + l4neg, tau5 - v2 + l4neg, l4Total)];
		}
	}
	lowv1= max5(-l2neg, tau1-l4+1, tau2-l4+1, tau3-l4+1, tau4-l4+1);
	hghv1= min5(l2, l4neg+tau1+1, l4neg+tau2+1, l4neg+tau3+1, l4neg+tau4+1);
	lowv2= max(-l2neg, tau5-l1+1);
	hghv2= min(l2, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1= g_h4[index4(tau1 - v1 + l4neg, tau2 - v1 + l4neg, tau3 - v1 + l4neg, tau4 - v1 + l4neg, l4Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h1[index1(tau5 - v2 + l1neg, l1Total)];
		}
	}

	lowv1= max3(-l2neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l2, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max4(-l2neg, tau3-l3+1, tau4-l3+1, tau5-l3+1);
	hghv2= min4(l2, l3neg+tau3+1, l3neg+tau4+1, l3neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h3[index3(tau3 - v2 + l3neg, tau4 - v2 + l3neg, tau5 - v2 + l3neg, l3Total)];
		}
	}
	lowv1= max4(-l2neg, tau1-l3+1, tau2-l3+1, tau3-l3+1);
	hghv1= min4(l2, l3neg+tau1+1, l3neg+tau2+1, l3neg+tau3+1);
	lowv2= max3(-l2neg, tau4-l2+1, tau5-l2+1); 
	hghv2= min3(l2, l2neg+tau4+1, l2neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h3[index3(tau1 - v1 + l3neg, tau2 - v1 + l3neg, tau3 - v1 + l3neg, l3Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			temp += k_h2[index2(v1 + l2neg, v2 + l2neg, l2Total)] * g1 * g_h2[index2(tau4 - v2 + l2neg, tau5 - v2 + l2neg, l2Total)];
		}
	}

	lowv1= max(-l3neg, tau1-l1+1);
	hghv1= min(l3, l1neg+tau1+1);
	lowv2= max(-l3neg, tau2-l1+1);
	hghv2= min(l3, l1neg+tau2+1); 
	int lowv3= max4(-l3neg, tau3-l3+1, tau4-l3+1, tau5-l3+1);
	int hghv3= min4(l3, l3neg+tau3+1, l3neg+tau4+1, l3neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 <  hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h3[index3(tau3 - v3 + l3neg, tau4 - v3 + l3neg, tau5 - v3 + l3neg, l3Total)];
			}
		}
	}
	lowv1= max(-l3neg, tau1-l1+1);
	hghv1= min(l3, l1neg+tau1+1);
	lowv2= max4(-l3neg, tau2-l3+1, tau3-l3+1, tau4-l3+1);
	hghv2= min4(l3, l3neg+tau2+1, l3neg+tau3+1, l3neg+tau4+1);
	lowv3= max(-l3neg, tau5-l1+1);
	hghv3= min(l3, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h3[index3(tau2 - v2 + l3neg, tau3 - v2 + l3neg, tau4 - v2 + l3neg, l3Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h1[index1(tau5 - v3 + l1neg, l1Total)];
			}
		}
	}
	lowv1= max4(-l3neg, tau1-l3+1, tau2-l3+1, tau3-l3+1);
	hghv1= min4(l3, l3neg+tau1+1, l3neg+tau2+1, l3neg+tau3+1);
	lowv2= max(-l3neg, tau4-l1+1);
	hghv2= min(l3, l1neg+tau4+1);
	lowv3= max(-l3neg, tau5-l1+1);
	hghv3= min(l3, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h3[index3(tau1 - v1 + l3neg, tau2 - v1 + l3neg, tau3 - v1 + l3neg, l3Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau4 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h1[index1(tau5 - v3 + l1neg, l1Total)];
			}
		}
	}

	lowv1= max(-l3neg, tau1-l1+1);
	hghv1= min(l3, l1neg+tau1+1);
	lowv2= max3(-l3neg, tau2-l2+1, tau3-l2+1);
	hghv2= min3(l3, l2neg+tau2+1, l2neg+tau3+1);
	lowv3= max3(-l3neg, tau4-l2+1, tau5-l2+1);
	hghv3= min3(l3, l2neg+tau4+1, l2neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h2[index2(tau2 - v2 + l2neg, tau3 - v2 + l2neg, l2Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h2[index2(tau4 - v3 + l2neg, tau5 - v3 + l2neg, l2Total)];
			}
		}
	}
	lowv1= max3(-l3neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l3, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max(-l3neg, tau3-l1+1); 
	hghv2= min(l3, l1neg+tau3+1);
	lowv3= max3(-l3neg, tau4-l2+1, tau5-l2+1);
	hghv3= min3(l3, l2neg+tau4+1, l2neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 <  hghv2; ++v2) {
			g2 = g_h1[index1(tau3 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h2[index2(tau4 - v3 + l2neg, tau5 - v3 + l2neg, l2Total)];
			}
		}
	}
	lowv1= max3(-l3neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l3, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max3(-l3neg, tau3-l2+1, tau4-l2+1);
	hghv2= min3(l3, l2neg+tau3+1, l2neg+tau4+1);
	lowv3= max(-l3neg, tau5-l1+1);
	hghv3= min(l3, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h2[index2(tau3 - v2 + l2neg, tau4 - v2 + l2neg, l2Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				temp += k_h3[index3(v1 + l3neg, v2 + l3neg, v3 + l3neg, l3Total)] * g1 * g2 * g_h1[index1(tau5 - v3 + l1neg, l1Total)];
			}
		}
	}

	lowv1= max(-l4neg, tau1-l1+1);
	hghv1= min(l4, l1neg+tau1+1);
	lowv2= max(-l4neg, tau2-l1+1);
	hghv2= min(l4, l1neg+tau2+1);
	lowv3= max(-l4neg, tau3-l1+1);
	hghv3= min(l4, l1neg+tau3+1);
	int lowv4= max3(-l4neg, tau4-l2+1, tau5-l2+1);
	int hghv4= min3(l4, l2neg+tau4+1, l2neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)] ;
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				g3 = g_h1[index1(tau3 - v3 + l1neg, l1Total)];
				for (int v4 = lowv4; v4 < hghv4; ++v4) {
					temp += k_h4[index4(v1 + l4neg, v2 + l4neg, v3 + l4neg, v4 + l4neg, l4Total)] * g1 * g2 * g3 * g_h2[index2(tau4 - v4 + l2neg, tau5 - v4 + l2neg, l2Total)];
				}
			}
		}
	}
	lowv1= max(-l4neg, tau1-l1+1);
	hghv1= min(l4, l1neg+tau1+1);
	lowv2= max(-l4neg, tau2-l1+1);
	hghv2= min(l4, l1neg+tau2+1);
	lowv3= max3(-l4neg, tau3-l2+1, tau4-l2+1);
	hghv3= min3(l4, l2neg+tau3+1, l2neg+tau4+1);
	lowv4= max(-l4neg, tau5-l1+1);
	hghv4= min(l4, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				g3 = g_h2[index2(tau3 - v3 + l2neg, tau4 - v3 + l2neg, l2Total)];
				for (int v4 = lowv4; v4 < hghv4; ++v4) {
					temp += k_h4[index4(v1 + l4neg, v2 + l4neg, v3 + l4neg, v4 + l4neg, l4Total)] * g1 * g2 * g3 * g_h1[index1(tau5 - v4 + l1neg, l1Total)];
				}
			}
		}
	}
	lowv1= max(-l4neg, tau1-l1+1);
	hghv1= min(l4, l1neg+tau1+1);
	lowv2= max3(-l4neg, tau2-l2+1, tau3-l2+1);
	hghv2= min3(l4, l2neg+tau2+1, l2neg+tau3+1);
	lowv3= max(-l4neg, tau4-l1+1);
	hghv3= min(l4, l1neg+tau4+1);
	lowv4= max(-l4neg, tau5-l1+1);
	hghv4= min(l4, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h2[index2(tau2 - v2 + l2neg, tau3 - v2 + l2neg, l2Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				g3 = g_h1[index1(tau4 - v3 + l1neg, l1Total)];
				for (int v4 = lowv4; v4 < hghv4; ++v4) {
					temp += k_h4[index4(v1 + l4neg, v2 + l4neg, v3 + l4neg, v4 + l4neg, l4Total)] * g1 * g2 * g3 * g_h1[index1(tau5 - v4 + l1neg, l1Total)];
				}
			}
		}
	}
	lowv1= max3(-l4neg, tau1-l2+1, tau2-l2+1);
	hghv1= min3(l4, l2neg+tau1+1, l2neg+tau2+1);
	lowv2= max(-l4neg, tau3-l1+1);
	hghv2= min(l4, l1neg+tau3+1);
	lowv3= max(-l4neg, tau4-l1+1);
	hghv3= min(l4, l1neg+tau4+1);
	lowv4= max(-l4neg, tau5-l1+1);
	hghv4= min(l4, l1neg+tau5+1);
	for (int v1 = lowv1; v1 < hghv1; ++v1) {
		g1 = g_h2[index2(tau1 - v1 + l2neg, tau2 - v1 + l2neg, l2Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau3 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				g3 = g_h1[index1(tau4 - v3 + l1neg, l1Total)];
				for (int v4 = lowv4; v4 < hghv4; ++v4) {
					temp += k_h4[index4(v1 + l4neg, v2 + l4neg, v3 + l4neg, v4 + l4neg, l4Total)] * g1 * g2 * g3 * g_h1[index1(tau5 - v4 + l1neg, l1Total)];
				}
			}
		}
	}

	lowv1= max(-l5neg, tau1-l1+1);
	hghv1= min(l5, l1neg+tau1+1);
	lowv2= max(-l5neg, tau2-l1+1);
	hghv2= min(l5, l1neg+tau2+1);
	lowv3= max(-l5neg, tau3-l1+1);
	hghv3= min(l5, l1neg+tau3+1);
	lowv4= max(-l5neg, tau4-l1+1);
	hghv4= min(l5, l1neg+tau4+1);
	int lowv5= max(-l5neg, tau5-l1+1);
	int hghv5= min(l5, l1neg+tau5+1);
	for (int v1 =lowv1; v1 < hghv1; ++v1) {
		g1 = g_h1[index1(tau1 - v1 + l1neg, l1Total)];
		for (int v2 = lowv2; v2 < hghv2; ++v2) {
			g2 = g_h1[index1(tau2 - v2 + l1neg, l1Total)];
			for (int v3 = lowv3; v3 < hghv3; ++v3) {
				g3 = g_h1[index1(tau3 - v3 + l1neg, l1Total)];
				for (int v4 = lowv4; v4 < hghv4; ++v4) {
					g4 = g_h1[index1(tau4 - v4 + l1neg, l1Total)];
					for (int v5 = lowv5; v5 < hghv5; ++v5) {
						temp += k_h5[index5(v1 + l5neg, v2 + l5neg, v3 + l5neg, v4 + l5neg, v5 + l5neg, l5Total)] * g1 * g2 * g3 * g4 * g_h1[index1(tau5 - v5 + l1neg, l1Total)];
					}
				}
			}
		}
	}

	res_h5[index5(tau1 + l5neg, tau2 + l5neg, tau3 + l5neg, tau4 + l5neg, tau5 + l5neg, l5Total)] = temp;
}

"""
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


def chainVolterraSystems_opencl(g, k, ctx=None, queue=None):
	res_sys = VolterraResponse(use_opencl=True, kernelName="resultSystem")

	# Kontext nur erstellen wenn nicht übergeben
	own_ctx = ctx is None
	if own_ctx:
		ctx, queue = create_context_and_queue(device_type='GPU', profiling=True)
	
	mf = cl.mem_flags

	l1 = g.len1
	l2 = g.len2
	l3 = g.len3
	l4 = g.len4
	l5 = g.len5
	
	l1neg = g.nLen1
	l2neg = g.nLen2
	l3neg = g.nLen3
	l4neg = g.nLen4
	l5neg = g.nLen5
	
	res_sys.len1=l1
	res_sys.len2=l2
	res_sys.len3=l3
	res_sys.len4=l4
	res_sys.len5=l5
	
	res_sys.nLen1=l1neg
	res_sys.nLen2=l2neg
	res_sys.nLen3=l3neg
	res_sys.nLen4=l4neg
	res_sys.nLen5=l5neg
	
	lTotal1=g.len1+g.nLen1
	lTotal2=g.len2+g.nLen2
	lTotal3=g.len3+g.nLen3
	lTotal4=g.len4+g.nLen4
	lTotal5=g.len5+g.nLen5

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
	res_h1 = np.zeros(lTotal1, dtype=np.float32)
	res_h2 = np.zeros(lTotal2 * lTotal2, dtype=np.float32)
	res_h3 = np.zeros(lTotal3 * lTotal3 * lTotal3, dtype=np.float32)
	res_h4 = np.zeros(lTotal4 * lTotal4 * lTotal4 * lTotal4, dtype=np.float32)
	res_h5 = np.zeros(lTotal5 * lTotal5 * lTotal5 * lTotal5 * lTotal5, dtype=np.float32)

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
	
	print('Composing.')
	print('H1:')	
	prg.volterraH1(queue, (lTotal1,), None, g_h1_buf, k_h1_buf, res_h1_buf, np.int32(l1), np.int32(l1neg)).wait()
	print('H2:')			   
	prg.volterraH2(queue, (lTotal2 * lTotal2,), None, g_h1_buf, g_h2_buf,k_h1_buf, k_h2_buf, res_h2_buf, np.int32(l1), np.int32(l1neg), np.int32(l2), np.int32(l2neg)).wait()
	print('H3:')			   
	prg.volterraH3(queue, (lTotal3 * lTotal3 * lTotal3,), None, g_h1_buf, g_h2_buf,g_h3_buf, k_h1_buf, k_h2_buf, k_h3_buf, res_h3_buf, np.int32(l1), np.int32(l1neg), np.int32(l2), np.int32(l2neg), np.int32(l3), np.int32(l3neg)).wait()
	print('H4:')			   
	prg.volterraH4(queue, (lTotal4 * lTotal4 * lTotal4 * lTotal4,), None, g_h1_buf, g_h2_buf, g_h3_buf,g_h4_buf, k_h1_buf, k_h2_buf, k_h3_buf, k_h4_buf, res_h4_buf, np.int32(l1), np.int32(l1neg), np.int32(l2), np.int32(l2neg), np.int32(l3), np.int32(l3neg), np.int32(l4), np.int32(l4neg)).wait()
	print('H5:')
	prg.volterraH5(queue, (lTotal5 * lTotal5 * lTotal5 * lTotal5 * lTotal5,), None, g_h1_buf, g_h2_buf, g_h3_buf,g_h4_buf, g_h5_buf, k_h1_buf, k_h2_buf, k_h3_buf, k_h4_buf, k_h5_buf, res_h5_buf, np.int32(l1), np.int32(l1neg), np.int32(l2), np.int32(l2neg), np.int32(l3), np.int32(l3neg),np.int32(l4), np.int32(l4neg), np.int32(l5), np.int32(l5neg)).wait()

	# Copy results back to host
	cl.enqueue_copy(queue, res_h1, res_h1_buf).wait()
	cl.enqueue_copy(queue, res_h2, res_h2_buf).wait()
	cl.enqueue_copy(queue, res_h3, res_h3_buf).wait()
	cl.enqueue_copy(queue, res_h4, res_h4_buf).wait()
	cl.enqueue_copy(queue, res_h5, res_h5_buf).wait()

	# Buffer explizit freigeben
	g_h1_buf.release()
	g_h2_buf.release()
	g_h3_buf.release()
	g_h4_buf.release()
	g_h5_buf.release()
	k_h1_buf.release()
	k_h2_buf.release()
	k_h3_buf.release()
	k_h4_buf.release()
	k_h5_buf.release()
	res_h1_buf.release()
	res_h2_buf.release()
	res_h3_buf.release()
	res_h4_buf.release()
	res_h5_buf.release()

	# Kontext nur freigeben wenn wir ihn selbst erstellt haben
	if own_ctx:
		queue.finish()
		del queue
		del ctx

	# Assign results
	res_sys.set_kernels(
		k.h0 + g.h0*np.sum(k.h1) + g.h0**2*np.sum(k.h2) + g.h0**3 * np.sum(k.h3) + g.h0**4*np.sum(k.h4) + g.h0**5*np.sum(k.h5),
		res_h1,
		res_h2.reshape((lTotal2, lTotal2)),
		res_h3.reshape((lTotal3, lTotal3, lTotal3)),
		res_h4.reshape((lTotal4, lTotal4, lTotal4, lTotal4)),
		res_h5.reshape((lTotal5, lTotal5, lTotal5, lTotal5, lTotal5)),
		np.array([l1neg,l2neg,l3neg,l4neg,l5neg])
	)

	return (make_symmetrical(res_sys))


# Similar to Zhu/Brazil, Behavioral Modeling of RF Power Amplifiers Based on Pruned Volterra Series (2004)
# Truncates the volterra series at strongly off-diagonal kernel entries.

# def offDiagonalTruncation(input_system, distance):
	# system = VolterraResponse(use_opencl=False, kernelName="truncatedSystem")
	# system.h0 = input_system.h0
	# system.h1 = input_system.h1
	# system.h2 = input_system.h2
	# system.h3 = input_system.h3
	# system.h4 = input_system.h4
	# system.h5 = input_system.h5


	# length1 = system.len1
	# length2 = system.len2
	# length3 = system.len3
	# length4 = system.len4
	# length5 = system.len5

	# for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):
		# if max(tau1, tau2)-min(tau1, tau2) > distance:
			# system.h2[tau1, tau2] = 0

		# perm = list(set(permutations([tau1, tau2])))
		# for [t1, t2] in perm:
			# system.h2[t1, t2] = system.h2[tau1, tau2]

	# for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
		# if max(tau1, tau2, tau3)-min(tau1, tau2, tau3) > distance:
			# system.h3[tau1, tau2, tau3] = 0

		# perm = list(set(permutations([tau1, tau2, tau3])))
		# for [t1, t2, t3] in perm:
			# system.h3[t1, t2, t3] = system.h3[tau1, tau2, tau3]

	# for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(np.arange(length4), 4)):
		# if max(tau1, tau2, tau3, tau4)-min(tau1, tau2, tau3, tau4) > distance:
			# system.h4[tau1, tau2, tau3, tau4] = 0

		# perm = list(set(permutations([tau1, tau2, tau3, tau4])))
		# for [t1, t2, t3, t4] in perm:
			# system.h4[t1, t2, t3, t4] = system.h4[tau1, tau2, tau3, tau4]

	# for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
		# if max(tau1, tau2, tau3, tau4, tau5)-min(tau1, tau2, tau3, tau4, tau5) > distance:
			# system.h5[tau1, tau2, tau3, tau4, tau5] = 0

		# perm = list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		# for [t1, t2, t3, t4, t5] in perm:
			# system.h5[t1, t2, t3, t4, t5] = system.h5[tau1,
													  # tau2, tau3, tau4, tau5]

		# return system


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

def invertKernels(inp_sys, use_opencl=True):
	inv_sys = VolterraResponse(
		use_opencl=use_opencl, kernelName="inverse_system")
	# Am Anfang von invertKernels, nach den print-Statements:
	ctx, queue = create_context_and_queue(device_type='GPU', profiling=True)

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

	l1 = inp_sys.len1
	l2 = inp_sys.len2
	l3 = inp_sys.len3
	l4 = inp_sys.len4
	l5 = inp_sys.len5
	
	l1neg = inp_sys.nLen1
	l2neg = inp_sys.nLen2
	l3neg = inp_sys.nLen3
	l4neg = inp_sys.nLen4
	l5neg = inp_sys.nLen5
	
	l1Total=l1+l1neg
	l2Total=l2+l2neg
	l3Total=l3+l3neg
	l4Total=l4+l4neg
	l5Total=l5+l5neg
	
	print(l1Total)
	print(l2Total)
	print(l3Total)
	print(l4Total)
	print(l5Total)
	

	inv_sys.set_kernels(
		0, 
		np.zeros(l1Total), 
		np.zeros(shape=(l2Total, l2Total)), 
		np.zeros(shape=(l3Total, l3Total, l3Total)), 
		np.zeros(shape=(l4Total, l4Total, l4Total, l4Total)), 
		np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)),
		nlens=([l1neg,l2neg,l3neg,l4neg,l5neg])
	)

		
	I0.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	I1.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	I2.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	I3.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	I4.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	I5.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))	

	H0.set_kernels(inp_sys.h0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	H1.set_kernels(0, inp_sys.h1, np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	H2.set_kernels(0, np.zeros(l1Total), inp_sys.h2, np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	H3.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), inp_sys.h3, np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	H4.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(
		l3Total, l3Total, l3Total)), inp_sys.h4, np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total)), nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))
	H5.set_kernels(0, np.zeros(l1Total), np.zeros(shape=(l2Total, l2Total)), np.zeros(shape=(l3Total, l3Total, l3Total)), np.zeros(
		shape=(l4Total, l4Total, l4Total, l4Total)), inp_sys.h5, nlens=([l1neg,l2neg,l3neg,l4neg,l5neg]))


	# I0:------------------------------

	I0.h0 = -inp_sys.h0

	# I1:-----------------------------------------
	#FFT-based 1st order inversion, with 1000 padding length
	padding = 1000
	impulse = np.concatenate((np.zeros(padding), np.zeros(
		l1+l1neg), np.zeros(padding)))
	impulse[2*padding+2*l1neg] = 1
	
	h1con = np.concatenate((np.zeros(padding), inp_sys.h1, np.zeros(padding)))
	
	invResponse = np.real(np.fft.ifft(np.fft.fft(impulse)/np.fft.fft(h1con)))
	
	# import sys

	# np.set_printoptions(threshold=sys.maxsize)
	# print(invResponse)

	#np.set_printoptions(threshold=100000, linewidth=200)
	#print(invResponse[padding:padding+l1+l1neg])

	print('Inverting.')
	print('	I1:')
	I1.h1 = invResponse[padding:padding+l1+l1neg]
	
	# I2:----------------------------------------
	#Calculate H'2
	Hp2=H2
	Hp3=addVolterraSystems(H2, H3)
	Hp4=addVolterraSystems(H2, H3, H4)
	Hp5=addVolterraSystems(H2, H3, H4, H5)
	
	
	# Dann alle Chain-Aufrufe mit ctx und queue:
	print('	I2:')
	I2 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(
		chainVolterraSystems_opencl(I1, Hp2, ctx, queue), I1, ctx, queue))
	print('	I3:')
	I3 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(
		chainVolterraSystems_opencl(I2, Hp3, ctx, queue), I1, ctx, queue))
	print('	I4:')
	I4 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(
		chainVolterraSystems_opencl(I3, Hp4, ctx, queue), I1, ctx, queue))
	print('	I5:')
	I5 = subtractVolterraSystems(I1, chainVolterraSystems_opencl(
		chainVolterraSystems_opencl(I4, Hp5, ctx, queue), I1, ctx, queue))

	# Am Ende aufräumen:
	queue.finish()
	del queue
	del ctx

	inv_sys	= I5
	return inv_sys


def wraparoundToOrdered(system):#Converts a wraparound system (negative/noncausal indices at the end of the array) to an ordered (causal) system; 'forgets' that the system is noncausal -> right-shift
	#Useful for plotting the system
	systemOut=copy.deepcopy(system)
	
	l1 = system.len1
	l2 = system.len2
	l3 = system.len3
	l4 = system.len4
	l5 = system.len5
	
	l1neg = system.nLen1
	l2neg = system.nLen2
	l3neg = system.nLen3
	l4neg = system.nLen4
	l5neg = system.nLen5
	
	l1Total=l1+l1neg
	l2Total=l2+l2neg
	l3Total=l3+l3neg
	l4Total=l4+l4neg
	l5Total=l5+l5neg
	
	#print(l1neg)
	
	#Convert wraparound system to ordered system	
	for tau1 in np.arange(l1Total):
		systemOut.h1[tau1]=system.h1[tau1-l1neg]
		
	for tau1 in np.arange(l2Total):
		for tau2 in np.arange(l2Total):
			systemOut.h2[tau1,tau2]=system.h2[tau1-l2neg, tau2-l2neg]
			
	for tau1 in np.arange(l3Total):
		for tau2 in np.arange(l3Total):
			for tau3 in np.arange(l3Total):
				systemOut.h3[tau1,tau2,tau3]=system.h3[tau1-l3neg, tau2-l3neg, tau3-l3neg]
				
	for tau1 in np.arange(l4Total):
		for tau2 in np.arange(l4Total):
			for tau3 in np.arange(l4Total):
				for tau4 in np.arange(l4Total):
					systemOut.h4[tau1,tau2,tau3,tau4]=system.h4[tau1-l4neg, tau2-l4neg, tau3-l4neg, tau4-l4neg]			
				
	for tau1 in np.arange(l5Total):
		for tau2 in np.arange(l5Total):
			for tau3 in np.arange(l5Total):
				for tau4 in np.arange(l5Total):
					for tau5 in np.arange(l5Total):
						systemOut.h5[tau1,tau2,tau3,tau4,tau5]=system.h5[tau1-l5neg, tau2-l5neg, tau3-l5neg, tau4-l5neg, tau5-l5neg]			
		
	# systemOut.nLen1=0
	# systemOut.nLen2=0
	# systemOut.nLen3=0
	# systemOut.nLen4=0
	# systemOut.nLen5=0
	
	# systemOut.len1=l1Total
	# systemOut.len2=l2Total
	# systemOut.len3=l3Total
	# systemOut.len4=l4Total
	# systemOut.len5=l5Total
	
	return(systemOut)	
	

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

import pyopencl as cl
import numpy as np
import numexpr as ne
import time
import matplotlib.pyplot as plt
import scipy.signal as sg
import pandas as pd


from itertools import combinations_with_replacement
from itertools import permutations
import Volterra_Kernels as kernels
from opencl_util import create_context_and_queue

import os
import csv

ctx, queue = create_context_and_queue(device_type='GPU')
mf = cl.mem_flags

# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# os.environ['PYOPENCL_CTX'] = '1'


#GPU
#The setup of the context, queue, and buffers happens by the code below:
platform = cl.get_platforms()
my_gpu_devices = [platform[0].get_devices(device_type=cl.device_type.GPU)[0]]#Device Number can be selected in the latter parentheses
ctx = cl.Context(devices=my_gpu_devices)

# #CPU
# #The setup of the context, queue, and buffers happens by the code below:
# platform = cl.get_platforms()
# my_cpu_devices = [platform[0].get_devices(device_type=cl.device_type.CPU)[0]]#Device Number can be selected in the latter parentheses
# ctx = cl.Context(devices=my_cpu_devices)



queue = cl.CommandQueue(ctx)
mf = cl.mem_flags




#The kernel is defined next.
prg2 = cl.Program(ctx, """
	// elementwise multiplication: C = A .* B.
	__kernel void matrixMul(
        __global float *A,
        __global float *B,
        __global float *C,
        ushort width, ushort height)
	{
    // ID
		int x = get_global_id(0);
		int y = get_global_id(1);

		// Multiplying
		C[y * height + x ] = A[y * height + x] * B[y * height + x];
	}
    """).build(cache_dir=None)




defineExtChkStr=''
#defineExtChkStr='#define EXTCHK '+str(length3+chunklength2)#Defines the extended chunk length	
prgC2 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC2(
        __global float *X,
        __global float *RES,
        ushort length1, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
		
        int t1= T%(length1)%length1;
		
    	int chk = T/(length1);
		
    	RES[t1 + chk*length1]=0;
        for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
            RES[t1 + chk*length1] += X[ind] * X[ind-t1];
			
		}	
	}

	__kernel void removeC2(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length1, uint n, float alpha
		)
	{
    // ID		
		
        int index = get_global_id(0);
		float Ytemp = 0;

        Y[index]=0;
		for(int t1=0; t1<length1; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
			
            float factorSymmetry=FACTOR[t1];
			float err=ERR[t1];
            
            Ytemp += - err*factorSymmetry*Xtemp1;
		}}
		Y[index] = Ytemp*alpha + X[index];
	}
    """).build(cache_dir=None)


defineExtChkStr=''
#defineExtChkStr='#define EXTCHK '+str(length3+chunklength2)#Defines the extended chunk length	
prgC3 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC3(
        __global float *X,
        __global float *RES,
        ushort length2, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
		
        int t1= T%(length2*length2)%length2;
        int t2= T%(length2*length2)/length2;
		
    	int chk = T/(length2*length2);
		
		
    	RES[t1 + t2 * length2 + chk*length2*length2]=0;
		if(t1>=t2){
			for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
				RES[t1 + t2 * length2 + chk*length2*length2] += X[ind] * X[ind-t1] * X[ind-t2];
			}
		}	
	}

	__kernel void removeC3(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length2, uint n, float alpha
		)
	{
    // ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		
		Y[index]=0;
		for(int t1=0; t1<length2; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
		for(int t2=0; t2<=t1; t2++){
			
            float factorSymmetry=FACTOR[t1 + t2 * length2];
			float err=ERR[t1 + t2 * length2];

            Ytemp += - err*factorSymmetry*Xtemp1*X[index-t2];
			
		}}}
		Y[index] = Ytemp*alpha + X[index];
	}
    """).build(cache_dir=None)	
	

defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC4 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC4(
        __global float *X,
        __global float *RES,
        ushort length3, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
        
        
        int t1= T%(length3*length3*length3)%length3%length3;
        int t2= T%(length3*length3*length3)/length3%length3;
        int t3= T%(length3*length3*length3)/length3/length3;
		
    	int chk = T/(length3*length3*length3);
		
    	RES[t1 + t2 * length3 + t3 * length3 * length3 + chk*length3*length3*length3]=0;
		if((t1>=t2) && (t2>=t3)){
			for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
				RES[t1 + t2 * length3 + t3 * length3 * length3 + chk*length3*length3*length3] += X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3];
			}
		}	
	}

	__kernel void removeC4(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length3, uint n, float alpha
		)
	{
    // ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		
		Y[index]=0;
		for(int t1=0; t1<length3; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
		for(int t2=0; t2<=t1; t2++){
		float Xtemp2=X[index-t2];
		for(int t3=0; t3<=t2; t3++){	
        
            float factorSymmetry=FACTOR[t1 + t2 * length3 + t3 * length3 * length3];
			float err=ERR[t1 + t2 * length3 + t3 * length3 * length3];
            
            Ytemp += - err*factorSymmetry*Xtemp1*Xtemp2*X[index-t3];//Reuse X[i-t1] and X[i-t2]
			
		}}}}
		Y[index] = Ytemp*alpha + X[index];
	}
    """).build(cache_dir=None)	





defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC5 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC5(
        __global float *X,
        __global float *RES,
        ushort length4, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
        
        
        int t1= T%(length4*length4*length4*length4)%length4%length4%length4;
        int t2= T%(length4*length4*length4*length4)/length4%length4%length4;
        int t3= T%(length4*length4*length4*length4)/length4/length4%length4;
		int t4= T%(length4*length4*length4*length4)/length4/length4/length4;
		
    	int chk = T/(length4*length4*length4*length4);
		
    	RES[t1 + t2*length4 + t3*length4*length4 + t4*length4*length4*length4 + chk*length4*length4*length4*length4]=0;
		if((t1>=t2) && (t2>=t3) && (t3>=t4)){
			for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
				RES[t1 + t2*length4 + t3*length4*length4 + t4*length4*length4*length4 + chk*length4*length4*length4*length4] += X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3]* X[ind-t4];
			}
		}	
	}

	__kernel void removeC5(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length4, uint n, float alpha
		)
	{
    // ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		
		Y[index]=0;
		for(int t1=0; t1<length4; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
		for(int t2=0; t2<=t1; t2++){
		float Xtemp2=X[index-t2];
		for(int t3=0; t3<=t2; t3++){	
        float Xtemp3=X[index-t3];
		for(int t4=0; t4<=t3; t4++){
		
            float factorSymmetry=FACTOR[t1 + t2*length4 + t3*length4*length4 + t4*length4*length4*length4];
			float err=ERR[t1 + t2*length4 + t3*length4*length4 + t4*length4*length4*length4];
            
            Ytemp += - err*factorSymmetry*Xtemp1*Xtemp2*Xtemp3*X[index-t4];//Reuse X[i-t1] and X[i-t2]
			
		}}}}}
		Y[index] = Ytemp*alpha + X[index];
	}
    """).build(cache_dir=None)	


defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC6 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC6(
        __global float *X,
        __global float *RES,
        ushort length5, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
        
        
        int t1= T%(length5*length5*length5*length5*length5)%length5%length5%length5%length5;
        int t2= T%(length5*length5*length5*length5*length5)/length5%length5%length5%length5;
        int t3= T%(length5*length5*length5*length5*length5)/length5/length5%length5%length5;
		int t4= T%(length5*length5*length5*length5*length5)/length5/length5/length5%length5;
		int t5= T%(length5*length5*length5*length5*length5)/length5/length5/length5/length5;
		
    	int chk = T/(length5*length5*length5*length5*length5);
		
    	RES[t1 + t2*length5 + t3*length5*length5 + t4*length5*length5*length5 + t5*length5*length5*length5*length5 +chk*length5*length5*length5*length5*length5]=0;
		if((t1>=t2) && (t2>=t3) && (t3>=t4) && (t4>=t5)){
			for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
				RES[t1 + t2*length5 + t3*length5*length5 + t4*length5*length5*length5 + t5*length5*length5*length5*length5 + chk*length5*length5*length5*length5*length5] += X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3]* X[ind-t4]* X[ind-t5];
			}
		}	
	}

	__kernel void removeC6(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length5, uint n, float alpha
		)
	{
    // ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		
		Y[index]=0;
		for(int t1=0; t1<length5; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
		for(int t2=0; t2<=t1; t2++){
		float Xtemp2=X[index-t2];
		for(int t3=0; t3<=t2; t3++){	
        float Xtemp3=X[index-t3];
		for(int t4=0; t4<=t3; t4++){
		float Xtemp4=X[index-t4];
		for(int t5=0; t5<=t4; t5++){
		
            float factorSymmetry=FACTOR[t1 + t2*length5 + t3*length5*length5 + t4*length5*length5*length5 + t5*length5*length5*length5*length5];
			float err=ERR[t1 + t2*length5 + t3*length5*length5 + t4*length5*length5*length5 + t5*length5*length5*length5*length5];
            
            Ytemp += - err*factorSymmetry*Xtemp1*Xtemp2*Xtemp3*Xtemp4*X[index-t5];//Reuse X[i-t1] and X[i-t2]
			
		}}}}}}
		Y[index] = Ytemp*alpha + X[index];
	}
    """).build(cache_dir=None)


defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC7 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC7(
        __global float *X,
        __global float *RES,
        ushort length6, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
        
        
        int t1= T%(length6*length6*length6*length6*length6*length6)%length6%length6%length6%length6%length6;
        int t2= T%(length6*length6*length6*length6*length6*length6)/length6%length6%length6%length6%length6;
        int t3= T%(length6*length6*length6*length6*length6*length6)/length6/length6%length6%length6%length6;
		int t4= T%(length6*length6*length6*length6*length6*length6)/length6/length6/length6%length6%length6;
		int t5= T%(length6*length6*length6*length6*length6*length6)/length6/length6/length6/length6%length6;
		int t6= T%(length6*length6*length6*length6*length6*length6)/length6/length6/length6/length6/length6;
		
    	int chk = T/(length6*length6*length6*length6*length6*length6);
		
    	RES[t1 + t2*length6 + t3*length6*length6 + t4*length6*length6*length6 + t5*length6*length6*length6*length6 + t6*length6*length6*length6*length6*length6 + chk*length6*length6*length6*length6*length6*length6]=0;
		if((t1>=t2) && (t2>=t3) && (t3>=t4) && (t4>=t5) && (t5>=t6)){
			for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
				RES[t1 + t2*length6 + t3*length6*length6 + t4*length6*length6*length6 + t5*length6*length6*length6*length6 + t6*length6*length6*length6*length6*length6 + chk*length6*length6*length6*length6*length6*length6] += X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3]* X[ind-t4]* X[ind-t5]* X[ind-t6];
			}
		}	
	}

	__kernel void removeC7(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length6, uint n, float alpha
		)
	{
    // ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		
		Y[index]=0;
		for(int t1=0; t1<length6; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
		for(int t2=0; t2<=t1; t2++){
		float Xtemp2=X[index-t2];
		for(int t3=0; t3<=t2; t3++){	
        float Xtemp3=X[index-t3];
		for(int t4=0; t4<=t3; t4++){
		float Xtemp4=X[index-t4];
		for(int t5=0; t5<=t4; t5++){
		float Xtemp5=X[index-t5];
		for(int t6=0; t6<=t5; t6++){
		
            float factorSymmetry=FACTOR[t1 + t2*length6 + t3*length6*length6 + t4*length6*length6*length6 + t5*length6*length6*length6*length6 + t6*length6*length6*length6*length6*length6];
			float err=ERR[t1 + t2*length6 + t3*length6*length6 + t4*length6*length6*length6 + t5*length6*length6*length6*length6 + t6*length6*length6*length6*length6*length6];
            
            Ytemp += - err*factorSymmetry*Xtemp1*Xtemp2*Xtemp3*Xtemp4*Xtemp5*X[index-t6];//Reuse X[i-t1] and X[i-t2]
			
		}}}}}}}
		Y[index] = Ytemp*alpha + X[index];

	}
    """).build(cache_dir=None)


defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC8 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC8(
        __global float *X,
        __global float *RES,
        ushort length7, uint chunklength, ushort chunks, uint begin
		)
	{
    // ID
		int T = get_global_id(0);
        
        
        int t1= T%(length7*length7*length7*length7*length7*length7*length7)%length7%length7%length7%length7%length7%length7;
        int t2= T%(length7*length7*length7*length7*length7*length7*length7)/length7%length7%length7%length7%length7%length7;
        int t3= T%(length7*length7*length7*length7*length7*length7*length7)/length7/length7%length7%length7%length7%length7;
		int t4= T%(length7*length7*length7*length7*length7*length7*length7)/length7/length7/length7%length7%length7%length7;
		int t5= T%(length7*length7*length7*length7*length7*length7*length7)/length7/length7/length7/length7%length7%length7;
		int t6= T%(length7*length7*length7*length7*length7*length7*length7)/length7/length7/length7/length7/length7%length7;
		int t7= T%(length7*length7*length7*length7*length7*length7*length7)/length7/length7/length7/length7/length7/length7;
		
    	int chk = T/(length7*length7*length7*length7*length7*length7*length7);
		
    	RES[t1 + t2*length7 + t3*length7*length7 + t4*length7*length7*length7 + t5*length7*length7*length7*length7 + t6*length7*length7*length7*length7*length7 + t7*length7*length7*length7*length7*length7*length7 + chk*length7*length7*length7*length7*length7*length7*length7]=0;
		
		if((t1>=t2) && (t2>=t3) && (t3>=t4) && (t4>=t5) && (t5>=t6) && (t6>=t7)){
			for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
				RES[t1 + t2*length7 + t3*length7*length7 + t4*length7*length7*length7 + t5*length7*length7*length7*length7 + t6*length7*length7*length7*length7*length7 + t7*length7*length7*length7*length7*length7*length7 + chk*length7*length7*length7*length7*length7*length7*length7] += X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3]* X[ind-t4]* X[ind-t5]* X[ind-t6] * X[ind-t7];
			}
		}	
	}

	__kernel void removeC8(
        __global float *X,
		__global float *ERR,
        __global float *FACTOR,
        __global float *Y,
        ushort length7, uint n, float alpha
		)
	{
    // ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		
		Y[index]=0;
		for(int t1=0; t1<length7; t1++){
        if(index>=t1){
		float Xtemp1=X[index-t1];
		for(int t2=0; t2<=t1; t2++){
		float Xtemp2=X[index-t2];
		for(int t3=0; t3<=t2; t3++){	
        float Xtemp3=X[index-t3];
		for(int t4=0; t4<=t3; t4++){
		float Xtemp4=X[index-t4];
		for(int t5=0; t5<=t4; t5++){
		float Xtemp5=X[index-t5];
		for(int t6=0; t6<=t5; t6++){
		float Xtemp6=X[index-t6];
		for(int t7=0; t7<=t6; t7++){
		
            float factorSymmetry=FACTOR[t1 + t2*length7 + t3*length7*length7 + t4*length7*length7*length7 + t5*length7*length7*length7*length7 + t6*length7*length7*length7*length7*length7+ t7*length7*length7*length7*length7*length7*length7];
			float err=ERR[t1 + t2*length7 + t3*length7*length7 + t4*length7*length7*length7 + t5*length7*length7*length7*length7 + t6*length7*length7*length7*length7*length7 + t7*length7*length7*length7*length7*length7*length7];
            
            Ytemp += - err*factorSymmetry*Xtemp1*Xtemp2*Xtemp3*Xtemp4*Xtemp5*Xtemp6*X[index-t7];//Reuse X[i-t1] and X[i-t2]
			
		}}}}}}}}
		Y[index] = Ytemp*alpha + X[index];

	}
    """).build(cache_dir=None)			

#--------------------------------------------------------------------------------------------

def expValXpowN(sigma, N):
	#https://math.stackexchange.com/questions/2751719/expected-value-of-xn-for-normal-distribution
	#returns the expected value of E[X^N], where X is a Normal(0,sigma^2) distributet random variable. 
	if N%2!=0: 
		return(0)
	
	if N==0:
		return(1)
	else: 
		return((N-1)*sigma**2 * expValXpowN(sigma, N-2))
		

#New, more general implementation of the expected value of a product of multiple time-shifted X.
def pyExpVal(tauList, sigma):
	dictionary = dict((i, tauList.count(i)) for i in tauList)

	#print(dictionary)
	product=1
	for key in dictionary:
		product=product * expValXpowN(sigma, dictionary[key])
	return(product)




def pyExpVal2(length1, Variance):
	expVal=np.zeros(shape=(length1))
	for tau1 in np.arange(length1):
		if tau1==0:
			expVal[tau1]=Variance
			
	return expVal




def pyCalcC2(xm, length1, chunks=1024):

	maxlength=len(xm)-length1
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength

    #create result array
	c = np.zeros((chunks*length1), dtype=np.float32)
	xm=xm.astype(np.float32)
    #create input and output buffers
	xm_buf = cl.Buffer\
	(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
    #call kernels
	prgC2.calcC2(queue, c.shape, None, xm_buf, c_buf, np.uint16(length1), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))
    
    #fill c4 with results from calculation and reshape
	C2 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C2, c_buf)
    
	C2=C2.reshape(chunks,length1)/actuallength
    
    #print(C2)
    
    #return reduced and symmetrized version
	C2reduced=np.sum(C2, axis=0)


	return(C2reduced)


def pyRemoveC2(xm, err, length1, alpha):
    
    n=len(xm)   
    xmout = np.empty_like(xm)
    
    
    #create the array containing the multiplication factors for kernel symmetry and covariance
    fac = np.zeros((length1), dtype = np.float32)
    for [tau1] in list(combinations_with_replacement(np.arange(length1), 1)):
        fac[tau1]=1
        

    #Use the right data types
    fac=fac.reshape(length1).astype(np.float32)
    err=err.reshape(length1).astype(np.float32)
    
    #Create buffers
    err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
    fac_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
    xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
    xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
    
    #alpha=0.1
    prgC2.removeC2(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length1), np.uint32(n), np.float32(alpha))
    cl.enqueue_copy(queue, xmout, xmout_buf)

    #print(np.var(xmout))
    #print(np.var(xm-xmout))
    return(xmout)


def decrease_c2x(iterations, xm, chunks, length1, alpha1, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):			
		xm=(xm-np.mean(xm)).astype(np.float32)
		C2=pyCalcC2(xm, length1, chunks)  
		expVal2x= pyExpVal2(length1, targetVar) 
		err2=(C2-expVal2x).astype(np.float32)
		xm=pyRemoveC2(xm, err2, length1, alpha1/targetVar)
		#print(np.sqrt(np.sum(err2**2))/(length1**2) / np.var(xm)**2)
	return(xm, err2)	


#------------------------------------------
def pyExpVal3(length2, Variance):
	expVal=np.zeros(shape=(length2, length2))
			
	return expVal# c_xxx should be zero for all tau1,tau2 for gaussian input.


def pyCalcC3(xm, length2, chunks=1024):

    maxlength=len(xm)-length2
    actuallength=int(maxlength/chunks)*chunks
    chunklength=int(maxlength/chunks)
    begin=len(xm)-actuallength

    #create result array
    c = np.zeros((chunks*length2**2), dtype=np.float32)
    xm=xm.astype(np.float32)
    #create input and output buffers
    xm_buf = cl.Buffer\
    (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
    #call kernels
    prgC3.calcC3(queue, c.shape, None, xm_buf, c_buf, np.uint16(length2), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))
    
    #fill c4 with results from calculation and reshape
    C3 = np.zeros(np.shape(c)).astype(np.float32)
    cl.enqueue_copy(queue, C3, c_buf)
    C3=C3.reshape(chunks,length2,length2)/actuallength
    
    #return reduced and symmetrized version
    C3reduced=np.sum(C3, axis=0)
    for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):
        perm=list(set(permutations([tau1, tau2])))
        for [t1, t2] in perm:
            C3reduced[t1,t2]=C3reduced[tau1,tau2]#Fill eqivalent kernel positions with copied entries  
    return(C3reduced)


def pyRemoveC3(xm, err, length2, alpha):
    
    n=len(xm)   
    xmout = np.empty_like(xm)
  
    #create the array containing the multiplication factors for kernel symmetry and covariance
    fac = np.zeros((length2,length2), dtype = np.float32)
    for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):
        divider=0
        divider+=kernels.calcDiagonalFactor((tau1,tau2))
        if tau1==0: divider+=kernels.calcDiagonalFactor((0,tau2))
        if tau2==0: divider+=kernels.calcDiagonalFactor((tau1,0))
        fac[tau1,tau2]=1/divider
        perm=list(set(permutations([tau1, tau2])))
        for [t1, t2] in perm:
            fac[t1,t2]=fac[tau1,tau2] 
   
    #Use the right data types
    fac=fac.reshape(length2**2).astype(np.float32)
    err=err.reshape(length2**2).astype(np.float32)
    
    #Create buffers
    err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
    fac_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
    xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
    xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
    
    #alpha=0.1
    prgC3.removeC3(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length2), np.uint32(n), np.float32(alpha))
    cl.enqueue_copy(queue, xmout, xmout_buf)

    #print(np.var(xmout))
    #print(np.max(np.abs(xm-xmout)))
    return(xmout)


def decrease_c3x(iterations, xm, chunks, length2, alpha2, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):	
		xm=(xm-np.mean(xm)).astype(np.float32)
		C3=pyCalcC3(xm, length2, chunks)  
		expVal3x= pyExpVal3(length2, targetVar) 
		err3=(C3-expVal3x).astype(np.float32)
		xm=pyRemoveC3(xm, err3, length2, alpha2/targetVar**2)	
	return(xm, err3)	


#------------------------------------
def pyExpVal4(length3, Variance):
	expVal=np.zeros(shape=(length3, length3, length3))
	
	for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
		perm=list(set(permutations([tau1, tau2, tau3])))#set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
		
		if tau1==0:
			if tau2==tau3: 
				expVal[tau1,tau2,tau3]=Variance**2
				if tau2==0:
					expVal[tau1,tau2,tau3]=3*Variance**2
			
		for [t1, t2, t3] in perm:
			expVal[t1,t2,t3]=expVal[tau1,tau2,tau3]#Fill eqivalent kernel positions with copied entries
			
	return expVal	


def pyCalcC4(xm, length3, chunks=1024):

    maxlength=len(xm)-length3
    actuallength=int(maxlength/chunks)*chunks
    chunklength=int(maxlength/chunks)
    begin=len(xm)-actuallength

    #create result array
    c = np.zeros((chunks*length3**3), dtype=np.float32)
    xm=xm.astype(np.float32)
    #create input and output buffers
    xm_buf = cl.Buffer\
    (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
    #call kernels
    prgC4.calcC4(queue, c.shape, None, xm_buf, c_buf, np.uint16(length3), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))
    
    #fill c4 with results from calculation and reshape
    C4 = np.zeros(np.shape(c)).astype(np.float32)
    cl.enqueue_copy(queue, C4, c_buf)
    C4=C4.reshape(chunks,length3,length3,length3)/actuallength
    
    #return reduced and symmetrized version
    C4reduced=np.sum(C4, axis=0)
    for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
        perm=list(set(permutations([tau1, tau2, tau3])))
        for [t1, t2, t3] in perm:
            C4reduced[t1,t2,t3]=C4reduced[tau1,tau2,tau3]#Fill eqivalent kernel positions with copied entries  
    return(C4reduced)


def pyRemoveC4(xm, err, length3, alpha):   
    n=len(xm)   
    xmout = np.empty_like(xm)
    
    #create the array containing the multiplication factors for kernel symmetry and covariance
    fac = np.zeros((length3,length3,length3), dtype = np.float32)
    for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
        divider=0
        divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3))
        if tau1==0: divider+=kernels.calcDiagonalFactor((0,tau2,tau3))
        if tau2==0: divider+=kernels.calcDiagonalFactor((tau1,0,tau3))
        if tau3==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,0))
        fac[tau1,tau2,tau3]=1/divider
        perm=list(set(permutations([tau1, tau2, tau3])))
        for [t1, t2, t3] in perm:
            fac[t1,t2,t3]=fac[tau1,tau2,tau3]
   
   
    #Use the right data types
    fac=(fac.reshape(length3**3)).astype(np.float32)
    err=(err.reshape(length3**3)).astype(np.float32)
    
    #Create buffers
    err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
    fac_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
    xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
    xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
    
    #alpha=0.1
    prgC4.removeC4(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length3), np.uint32(n), np.float32(alpha))
    cl.enqueue_copy(queue, xmout, xmout_buf)

    #print(np.var(xmout))
    #print(np.max(np.abs(xm-xmout)))
    return(xmout)
	
def decrease_c4x(iterations, xm, chunks, length3, alpha3, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):	
		xm=(xm-np.mean(xm)).astype(np.float32)
		C4=pyCalcC4(xm, length3, chunks)  
		expVal4x= pyExpVal4(length3, targetVar) 
		err4=(C4-expVal4x).astype(np.float32)
		xm=pyRemoveC4(xm, err4, length3, alpha3/targetVar**3)	
	return(xm, err4)	



#------------------------------------
def pyExpVal5(length4, Variance):
	expVal=np.zeros(shape=(length4, length4, length4, length4))
			
	return expVal	



def pyCalcC5(xm, length4, chunks=1024):

    maxlength=len(xm)-length4
    actuallength=int(maxlength/chunks)*chunks
    chunklength=int(maxlength/chunks)
    begin=len(xm)-actuallength

    #create result array
    c = np.zeros((chunks*length4**4), dtype=np.float32)
    xm=xm.astype(np.float32)
    #create input and output buffers
    xm_buf = cl.Buffer\
    (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
    #call kernels
    prgC5.calcC5(queue, c.shape, None, xm_buf, c_buf, np.uint16(length4), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))
    
    #fill c4 with results from calculation and reshape
    C5 = np.zeros(np.shape(c)).astype(np.float32)
    cl.enqueue_copy(queue, C5, c_buf)
    C5=C5.reshape(chunks,length4,length4,length4,length4)/actuallength
    
    #return reduced and symmetrized version
    C5reduced=np.sum(C5, axis=0)
    for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(np.arange(length4), 4)):
        perm=list(set(permutations([tau1, tau2, tau3, tau4])))
        for [t1, t2, t3, t4] in perm:
            C5reduced[t1,t2,t3,t4]=C5reduced[tau1,tau2,tau3,tau4]#Fill eqivalent kernel positions with copied entries  
    return(C5reduced)


def pyRemoveC5(xm, err, length4, alpha):   
	n=len(xm)   
	xmout = np.empty_like(xm)
	
	#create the array containing the multiplication factors for kernel symmetry and covariance
	fac = np.zeros((length4,length4,length4,length4), dtype = np.float32)
	for [tau1, tau2, tau3, tau4] in list(combinations_with_replacement(np.arange(length4), 4)):
		divider=0
		divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4))
		if tau1==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4))
		if tau2==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4))
		if tau3==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4))
		if tau4==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4))
		fac[tau1,tau2,tau3,tau4]=1/divider
		perm=list(set(permutations([tau1, tau2, tau3, tau4])))
		for [t1, t2, t3, t4] in perm:
			fac[t1,t2,t3,t4]=fac[tau1,tau2,tau3,tau4]
   
   
    #Use the right data types
	fac=(fac.reshape(length4**4)).astype(np.float32)
	err=(err.reshape(length4**4)).astype(np.float32)
	
	#Create buffers
	err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
	fac_buf = cl.Buffer\
		(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
	xm_buf = cl.Buffer\
		(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
    
    #alpha=0.1
	prgC5.removeC5(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length4), np.uint32(n), np.float32(alpha))
	cl.enqueue_copy(queue, xmout, xmout_buf)

    #print(np.var(xmout))
	#print(np.max(np.abs(xm-xmout)))
	return(xmout)
	
def decrease_c5x(iterations, xm, chunks, length4, alpha4, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):	
		xm=(xm-np.mean(xm)).astype(np.float32)
		C5=pyCalcC5(xm, length4, chunks)  
		expVal5x= pyExpVal5(length4, targetVar) 
		err5=(C5-expVal5x).astype(np.float32)
		xm=pyRemoveC5(xm, err5, length4, alpha4/targetVar**4)	
	return(xm, err5)	



#
def pyExpVal6(length5, Variance):
	expVal=np.zeros(shape=(length5, length5, length5, length5, length5))
	
	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5])))#set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
		
		if tau1!=0: expVal[tau1,tau2,tau3,tau4,tau5] = 0
		else:
			if tau2==0:
				if tau3!=0: expVal[tau1,tau2,tau3,tau4,tau5] = 0
				else: 
					if tau4==0:
						if tau5!=0: expVal[tau1,tau2,tau3,tau4,tau5] = 0 
						else: expVal[tau1,tau2,tau3,tau4,tau5] =(expValXpowN(np.sqrt(Variance), 6))
					else:#"case 2"
						if tau4!=tau5: expVal[tau1,tau2,tau3,tau4,tau5] = 0
						else:
							expVal[tau1,tau2,tau3,tau4,tau5] =(expValXpowN(np.sqrt(Variance), 4) * expValXpowN(np.sqrt(Variance), 2))
			else:#"case 1"
				if tau3!=tau2: expVal[tau1,tau2,tau3,tau4,tau5] =0#This one was maybe missing in my calculations. It was found by computing all errors, finding outliers and making sense of their t1...t5 indices. After a calculation, this additional condition was identified. 
				else:
					if tau4==tau3:
						if tau5!=tau4: expVal[tau1,tau2,tau3,tau4,tau5] = 0
						else:
							expVal[tau1,tau2,tau3,tau4,tau5] =(expValXpowN(np.sqrt(Variance), 2) * expValXpowN(np.sqrt(Variance), 4))
					else:#"case 1.1"
						if tau4!=tau5: expVal[tau1,tau2,tau3,tau4,tau5] = 0
						else:
							expVal[tau1,tau2,tau3,tau4,tau5] =(expValXpowN(np.sqrt(Variance), 2) * expValXpowN(np.sqrt(Variance), 2) * expValXpowN(np.sqrt(Variance), 2))
					
		for [t1, t2, t3, t4, t5] in perm:
			expVal[t1,t2,t3,t4,t5]=expVal[tau1,tau2,tau3,tau4,tau5]#Fill eqivalent kernel positions with copied entries
			
	return expVal


def pyCalcC6(xm, length5, chunks=1024):

	maxlength=len(xm)-length5
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength

	#create result array
	c = np.zeros((chunks*length5**5), dtype=np.float32)
	xm=xm.astype(np.float32)
	#create input and output buffers
	xm_buf = cl.Buffer\
	(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
	#call kernels
	prgC6.calcC6(queue, c.shape, None, xm_buf, c_buf, np.uint16(length5), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))

	#fill c4 with results from calculation and reshape
	C6 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C6, c_buf)
	C6=C6.reshape(chunks,length5,length5,length5,length5,length5)/actuallength

	
	#return reduced and symmetrized version
	C6reduced=np.sum(C6, axis=0)
	
	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
		#print(str(C6reduced[tau1,tau2,tau3,tau4,tau5]) + '//' + str(tau1)+ ' ' + str(tau2)+ ' ' + str(tau3)+ ' ' + str(tau4)+ ' ' + str(tau5))
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for [t1, t2, t3, t4, t5] in perm:
			C6reduced[t1,t2,t3,t4,t5]=C6reduced[tau1,tau2,tau3,tau4,tau5]#Fill eqivalent kernel positions with copied entries 		
	return(C6reduced)


def pyRemoveC6(xm, err, length5, alpha):   
	n=len(xm)   
	xmout = np.empty_like(xm)

	#create the array containing the multiplication factors for kernel symmetry and covariance
	fac = np.zeros((length5,length5,length5,length5,length5), dtype = np.float32)
	for [tau1, tau2, tau3, tau4, tau5] in list(combinations_with_replacement(np.arange(length5), 5)):
		divider=0
		divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5))
		if tau1==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5))
		if tau2==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5))
		if tau3==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5))
		if tau4==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5))
		if tau5==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5))
		
		fac[tau1,tau2,tau3,tau4,tau5]=1/divider
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5])))
		for [t1, t2, t3, t4, t5] in perm:
			fac[t1,t2,t3,t4,t5]=fac[tau1,tau2,tau3,tau4,tau5]


	#Use the right data types
	fac=(fac.reshape(length5**5)).astype(np.float32)
	err=(err.reshape(length5**5)).astype(np.float32)

	#Create buffers
	err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
	fac_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
	xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)

	#alpha=0.1
	prgC6.removeC6(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length5), np.uint32(n), np.float32(alpha))
	cl.enqueue_copy(queue, xmout, xmout_buf)

	#print(np.var(xmout))
	#print(np.max(np.abs(xm-xmout)))
	return(xmout)
	
def decrease_c6x(iterations, xm, chunks, length5, alpha5, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):	
		xm=(xm-np.mean(xm)).astype(np.float32)
		C6=pyCalcC6(xm, length5, chunks)  
		expVal6x= pyExpVal6(length5, targetVar) 
		err6=(C6-expVal6x).astype(np.float32)
		xm=pyRemoveC6(xm, err6, length5, alpha5/targetVar**5)	
	return(xm, err6)	


#New: Higher Orders
def pyExpVal7(length6, Variance):
	expVal=np.zeros(shape=(length6, length6, length6, length6, length6, length6))
	
	for [tau1, tau2, tau3, tau4, tau5, tau6] in list(combinations_with_replacement(np.arange(length6), 6)):
		expVal[tau1,tau2,tau3,tau4,tau5,tau6] = pyExpVal(list([0,tau1,tau2,tau3,tau4,tau5,tau6]), np.sqrt(Variance))
	
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6])))
		for [t1, t2, t3, t4, t5, t6] in perm:
			expVal[t1,t2,t3,t4,t5,t6]=expVal[tau1,tau2,tau3,tau4,tau5,tau6]#Fill eqivalent kernel positions with copied entries
		
	return expVal	
	

def pyCalcC7(xm, length6, chunks=1024):

	maxlength=len(xm)-length6
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength

	#create result array
	c = np.zeros((chunks*length6**6), dtype=np.float32)
	xm=xm.astype(np.float32)
	#create input and output buffers
	xm_buf = cl.Buffer\
	(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
	#call kernels
	prgC7.calcC7(queue, c.shape, None, xm_buf, c_buf, np.uint16(length6), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))

	#fill c4 with results from calculation and reshape
	C7 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C7, c_buf)
	C7=C7.reshape(chunks,length6,length6,length6,length6,length6,length6)/actuallength

	
	#return reduced and symmetrized version
	C7reduced=np.sum(C7, axis=0)
	#print(C7reduced)
	
	for [tau1, tau2, tau3, tau4, tau5, tau6] in list(combinations_with_replacement(np.arange(length6), 6)):
		#print(str(C7reduced[tau1,tau2,tau3,tau4,tau5, tau6]) + '//' + str(tau1)+ ' ' + str(tau2)+ ' ' + str(tau3)+ ' ' + str(tau4)+ ' ' + str(tau5, tau6))
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6])))
		for [t1, t2, t3, t4, t5, t6] in perm:
			C7reduced[t1,t2,t3,t4,t5,t6]=C7reduced[tau1,tau2,tau3,tau4,tau5,tau6]#Fill eqivalent kernel positions with copied entries 		
	return(C7reduced)


def pyRemoveC7(xm, err, length6, alpha):   
	n=len(xm)   
	xmout = np.empty_like(xm)

	#create the array containing the multiplication factors for kernel symmetry and covariance
	fac = np.zeros((length6,length6,length6,length6,length6,length6), dtype = np.float32)
	for [tau1, tau2, tau3, tau4, tau5, tau6] in list(combinations_with_replacement(np.arange(length6), 6)):
		divider=0
		divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		if tau1==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		if tau2==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		if tau3==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		if tau4==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		if tau5==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		if tau6==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6))
		
		fac[tau1,tau2,tau3,tau4,tau5,tau6]=1/divider
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6])))
		for [t1, t2, t3, t4, t5, t6] in perm:
			fac[t1,t2,t3,t4,t5,t6]=fac[tau1,tau2,tau3,tau4,tau5,tau6]


	#Use the right data types
	fac=(fac.reshape(length6**6)).astype(np.float32)
	err=(err.reshape(length6**6)).astype(np.float32)

	#Create buffers
	err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
	fac_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
	xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)

	#alpha=0.1
	prgC7.removeC7(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length6), np.uint32(n), np.float32(alpha))
	cl.enqueue_copy(queue, xmout, xmout_buf)

	#print(np.var(xmout))
	#print(np.max(np.abs(xm-xmout)))
	return(xmout)
	
def decrease_c7x(iterations, xm, chunks, length6, alpha6, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):	
		xm=(xm-np.mean(xm)).astype(np.float32)
		C7=pyCalcC7(xm, length6, chunks)  
		expVal7x= pyExpVal7(length6, targetVar)
		err7=(C7-expVal7x).astype(np.float32)
		xm=pyRemoveC7(xm, err7, length6, alpha6/targetVar**6)	
	return(xm, err7)	



	

def pyExpVal8(length7, Variance):
	expVal=np.zeros(shape=(length7, length7, length7, length7, length7, length7, length7))
	
	for [tau1, tau2, tau3, tau4, tau5, tau6, tau7] in list(combinations_with_replacement(np.arange(length7), 7)):
		expVal[tau1,tau2,tau3,tau4,tau5,tau6,tau7] = pyExpVal(list([0,tau1,tau2,tau3,tau4,tau5,tau6,tau7]), np.sqrt(Variance))
	
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6, tau7])))
		for [t1, t2, t3, t4, t5, t6, t7] in perm:
			expVal[t1,t2,t3,t4,t5,t6,t7]=expVal[tau1,tau2,tau3,tau4,tau5,tau6,tau7]#Fill eqivalent kernel positions with copied entries
		
	return expVal	
	

def pyCalcC8(xm, length7, chunks=1024):

	maxlength=len(xm)-length7
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength

	#create result array
	c = np.zeros((chunks*length7**7), dtype=np.float32)
	xm=xm.astype(np.float32)
	#create input and output buffers
	xm_buf = cl.Buffer\
	(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)    
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

		
	#call kernels
	prgC8.calcC8(queue, c.shape, None, xm_buf, c_buf, np.uint16(length7), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))

	#fill c4 with results from calculation and reshape
	C8 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C8, c_buf)
	C8=C8.reshape(chunks,length7,length7,length7,length7,length7,length7,length7)/actuallength
	
	#return reduced and symmetrized version
	C8reduced=np.sum(C8, axis=0)
	
	for [tau1, tau2, tau3, tau4, tau5, tau6, tau7] in list(combinations_with_replacement(np.arange(length7), 7)):
		#print(str(C8reduced[tau1,tau2,tau3,tau4,tau5, tau6, tau7]) + '//' + str(tau1)+ ' ' + str(tau2)+ ' ' + str(tau3)+ ' ' + str(tau4)+ ' ' + str(tau5, tau6, tau7))
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6, tau7])))
		for [t1, t2, t3, t4, t5, t6, t7] in perm:
			C8reduced[t1,t2,t3,t4,t5,t6,t7]=C8reduced[tau1,tau2,tau3,tau4,tau5,tau6,tau7]#Fill eqivalent kernel positions with copied entries 	
	#print(C8reduced[0,0,0,0,0,0,0])		
	return(C8reduced)


def pyRemoveC8(xm, err, length7, alpha):   
	n=len(xm)   
	xmout = np.empty_like(xm)

	#create the array containing the multiplication factors for kernel symmetry and covariance
	fac = np.zeros((length7,length7,length7,length7,length7,length7,length7), dtype = np.float32)
	for [tau1, tau2, tau3, tau4, tau5, tau6, tau7] in list(combinations_with_replacement(np.arange(length7), 7)):
		divider=0
		divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau1==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau2==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau3==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau4==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau5==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau6==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		if tau7==0: divider+=kernels.calcDiagonalFactor((tau1,tau2,tau3,tau4,tau5,tau6,tau7))
		
		fac[tau1,tau2,tau3,tau4,tau5,tau6,tau7]=1/divider
		perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6, tau7])))
		for [t1, t2, t3, t4, t5, t6, t7] in perm:
			fac[t1,t2,t3,t4,t5,t6,t7]=fac[tau1,tau2,tau3,tau4,tau5,tau6,tau7]


	#Use the right data types
	fac=(fac.reshape(length7**7)).astype(np.float32)
	err=(err.reshape(length7**7)).astype(np.float32)

	#Create buffers
	err_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
	fac_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
	xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)

	#alpha=0.1
	prgC8.removeC8(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf, xmout_buf, np.uint16(length7), np.uint32(n), np.float32(alpha))
	cl.enqueue_copy(queue, xmout, xmout_buf)

	#print(np.var(xmout))
	#print(np.max(np.abs(xm-xmout)))
	return(xmout)
	
def decrease_c8x(iterations, xm, chunks, length7, alpha7, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in range(iterations):	
		xm=(xm-np.mean(xm)).astype(np.float32)
		C8=pyCalcC8(xm, length7, chunks)  
		expVal8x= pyExpVal8(length7, targetVar)
		err8=(C8-expVal8x).astype(np.float32)
		xm=pyRemoveC8(xm, err8, length7, alpha7/targetVar**7)	
	return(xm, err8)	



#---------------------------------------------------------------------------------------------------------------
def deCorrelateFast(times, x, y, alpha=np.array([0, 0.6, 0.35, 0.25, 0.2, 0.2, 0.1, 0.1]), length=np.array([50,20,15,10,10,9,9]), iterations=15, iterations_c2=10, iterations_initial=35, write=True, name='', chunks=1024, debug=False):
	#Higher values for alpha increase the speed of convergence, but too high values lead to instability
	
#test---------
	length1=length[0]
	length2=length[1]
	length3=length[2]
	length4=length[3]
	length5=length[4]
	length6=length[5]
	length7=length[6]
	
	sigmaClip=5.5
	
	err2=0
	err3=0
	err4=0
	err5=0
	err6=0
	err7=0
	err8=0
#--------------
	
	xmodified=np.copy(x)#*100# Make better use of 32bit float range
	tvar=np.var(xmodified)#target variance of the output signal

	pointsMax=9

	
	re=np.zeros(shape=(pointsMax, iterations))#relative error
	mu=np.zeros(shape=(pointsMax, iterations))#rate of convergence
	incAlpha=np.zeros(shape=(pointsMax, iterations))#Tells if alpha has been increased in this iteration
	al=np.zeros(shape=(pointsMax, iterations))#Tells if alpha has been increased in this iteration
	al[:-1,0]=alpha
	
	if(not debug):
		xmodified, _=decrease_c2x(iterations_initial, xmodified, chunks, length1, 0.1, targetVar=tvar)#0.3

	
	for i in np.arange(iterations):
		# if i==50:
			# alpha[1]=alpha[1]*1.1
			# alpha[2]=alpha[2]*1.1
			# alpha[3]=alpha[3]*1.1
			# alpha[4]=alpha[4]*1.1
			# alpha[5]=alpha[5]*1.1
			# alpha[6]=alpha[6]*1.1
			# alpha[7]=alpha[7]*1.1
		# if i == 35:
			# alpha[1]=0.2#0.2
			# alpha[2]=0.2
			# alpha[3]=0.2
		

		#Strategy 2
		xcopy=np.copy(xmodified)
		xcorr2=np.zeros(len(xmodified))
		xcorr3=np.zeros(len(xmodified))
		xcorr4=np.zeros(len(xmodified))
		xcorr5=np.zeros(len(xmodified))
		xcorr6=np.zeros(len(xmodified))
		xcorr7=np.zeros(len(xmodified))
		xcorr8=np.zeros(len(xmodified))
		
		
		xmod2, err2 =decrease_c2x(iterations_c2, xcopy, chunks, length1, al[1,i-1], targetVar=tvar)
		xcorr2=xmod2-xcopy
			
	#if i%2==0:
		xmod3, err3=decrease_c3x(1, xcopy, chunks, length2, al[2,i-1], targetVar=tvar)
		xcorr3=xmod3-xcopy
	
	#if i%int(1.5**2)==0:
		xmod4, err4=decrease_c4x(1, xcopy, chunks, length3, al[3,i-1], targetVar=tvar)
		xcorr4=xmod4-xcopy
	
	#if i%int(1.5**3)==0:
		xmod5, err5=decrease_c5x(1, xcopy, chunks, length4, al[4,i-1], targetVar=tvar)
		xcorr5=xmod5-xcopy
	
	#if i%int(1.5**4)==0:
		xmod6, err6=decrease_c6x(1, xcopy, chunks, length5, al[5,i-1], targetVar=tvar)
		xcorr6=xmod6-xcopy
	
		# if i==int(iterations/3): 
			# al[2,i-1]=0.2
			# al[3,i-1]=0.2
			# al[4,i-1]=0.2
			# al[5,i-1]=0.2
			# al[6,i-1]=0.1
			# al[7,i-1]=0.1
			
		if (i>iterations/4):
			

			xmod7, err7=decrease_c7x(1, xcopy, chunks, length6, al[6,i-1], targetVar=tvar)
			xcorr7=xmod7-xcopy
				
		
			xmod8, err8=decrease_c8x(1, xcopy, chunks, length7, al[7,i-1], targetVar=tvar)
			xcorr8=xmod8-xcopy
			

		xmodified=xcopy +(xcorr2 +xcorr3 +xcorr4 +xcorr5 +xcorr6 +xcorr7 +xcorr8)*0.35
		#print(np.sqrt(np.var(([xcorr2,xcorr3,xcorr4,xcorr5,xcorr6,xcorr7,xcorr8]), axis=1)))
		#print(np.argmax(abs(xmodified)))
		#print(np.max(abs(xmodified)))
		xmodified,_ = decrease_c2x(iterations_c2, xmodified, chunks, length1, al[1,i-1], targetVar=tvar)

#-------------------------------------------------------------------------


		
		#Print relative errors
		re[2,i]=np.sqrt(np.sum(err2**2))/(length1**1) / np.var(xmodified)
		re[3,i]=np.sqrt(np.sum(err3**2))/(length2**2) / np.var(xmodified)**1.5
		re[4,i]=np.sqrt(np.sum(err4**2))/(length3**3) / np.var(xmodified)**2
		re[5,i]=np.sqrt(np.sum(err5**2))/(length4**4) / np.var(xmodified)**2.5
		re[6,i]=np.sqrt(np.sum(err6**2))/(length5**5) / np.var(xmodified)**3
		re[7,i]=np.sqrt(np.sum(err7**2))/(length6**6) / np.var(xmodified)**3.5
		re[8,i]=np.sqrt(np.sum(err8**2))/(length7**7) / np.var(xmodified)**4
		#print('Iteration {it:>3}: 2pt: {a:5.2e} ; 3pt: {b:5.2e} ; 4pt: {c:5.2e} ; 5pt: {d:5.2e} ; 6pt: {e:5.2e}   '.format(it = i, a=re[2,i], b=re[3,i], c=re[4,i], d=re[5,i], e=re[6,i]))
		#print('Iteration {it:>3}: 2pt: {a:5.2e} ; 3pt: {b:5.2e} ; 4pt: {c:5.2e} ; 5pt: {d:5.2e} ; 6pt: {e:5.2e} ; 7pt: {f:5.2e}   '.format(it = i, a=re[2,i], b=re[3,i], c=re[4,i], d=re[5,i], e=re[6,i], f=re[7,i]))
		print('Iteration {it:>3}: 2pt: {a:5.2e} ; 3pt: {b:5.2e} ; 4pt: {c:5.2e} ; 5pt: {d:5.2e} ; 6pt: {e:5.2e} ; 7pt: {f:5.2e} ; 8pt: {g:5.2e}  '.format(it = i, a=re[2,i], b=re[3,i], c=re[4,i], d=re[5,i], e=re[6,i], f=re[7,i], g=re[8,i]))


#test: adaptive convergence speed------------------
		adaptive=0
		if(adaptive):
			#Calculate Convergence Rate
			for point in np.arange(2,pointsMax):
				if i>0:
					mu[point,i]=re[point,i]/re[point,i-1]#Current error divided by last error
				
			#Decision Algorithm	
			for point in np.arange(2,pointsMax):
				interval=1
				if i>0 and i%interval==0:
					if mu[point,i] - 0.7*mu[point,i-interval] - 0.3*mu[point,i-2*interval] + np.random.normal(0,0.005) + 0.001*(np.sum(mu[3:,i]**20)**0.05 - np.sum(mu[3:,i-interval]**20)**0.05) <0:#Improvement of the current µ, and of the 10-norm of other µs
						
						if al[point-1,i-2]>al[point-1, i-interval-2]:
							incAlpha[point-1,i]=1
						else:
							incAlpha[point-1,i]=0
					else:
						if al[point-1,i-2]>al[point-1, i-interval-2]:
							incAlpha[point-1,i]=0
						else:
							incAlpha[point-1,i]=1
				
					#if mu[point,i]>1: incAlpha[point-1,i]=0#Don't increase alpha if re[i]>re[i-1], since this means instability
				
				
					if incAlpha[point-1,i]==1:
						al[point-1,i]=al[point-1,i-1]*1.03
						#alpha=alpha*1.005
						
					else: 
						al[point-1,i]=al[point-1,i-1]/1.03
						#alpha=alpha/1.005
			
					#al[point-1,i]+=np.random.normal(0,0.01)
			# #Limit the fastest convergence
			# j=np.argmin(mu[2:,i])+2
			# al[j-1,i]=al[j-1,i]/1.03	
			
			#Fixed Convergence Rate
			for point in np.arange(2,pointsMax):
				if mu[point,i]<0.955:
					al[point-1,i]=al[point-1,i]/1.2

			
			for point in np.arange(2,pointsMax):
				if(al[point-1,i] > 0.8): al[point-1,i]=0.8	
				if point > 3:
					if(al[point-1,i] > 0.4): al[point-1,i]=0.4
			
			#if(alpha[1] < 0.3): alpha[1]=0.3
			al[1,i]=0.35
			
			#print(mu[:,i])
			#print(incAlpha[:,i])
#---------------------------------------------		
		
		#Copy old alpha to the next iteration
		if i<iterations-1:
			for point in np.arange(1,pointsMax):
				al[point-1,i+1]=al[point-1,i]
				
		print(al[:,i])	
#-----------------------------------------

		
	
	#Spectrum has priority
	xmodified, _=decrease_c2x(3*iterations_c2, xmodified, chunks, length1, alpha[1], targetVar=tvar)
	
	xmodified=xmodified#/100#Make better use of 32bit float range
	
	
	if(write):
		with open(name+'_dec.csv', mode='w', encoding="utf-8", newline='') as csvfile:
			csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for n in np.arange(len(x)):
				#csv_writer.writerow([times[n], xmodified[n], y[n], x[n]])
				csv_writer.writerow([times[n], xmodified[n], x[n], y[n]])
	
				
				
	if(debug): return(xmodified, (re[2],re[3],re[4],re[5],re[6],re[7],re[8]))
	else: return(xmodified)
#-----------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

	length1=50
	length2=10
	length3=10
	length4=10
	length5=5
	(T1, T2, T3)=(length3,length3,length3)

	chunks=512#use so many chunks that chunklength<1e6 (or better, <1e5),  to reduce float rounding errors and make better use of float16 dynamic range
	n=2**22+chunks 
	#n=4194304



	#chunks2=10000
	#chunklength2=100




		
	

	if(1):#Random input data
		x = np.random.normal(loc=0.0, scale=1.0, size=n)
		x = x.astype(np.float32)
		
		timestep=100e-12
		times=np.arange(n)


		#x=np.convolve(x, np.array([0,0,0.9,0.07,0.03]), mode='same')
		#x[1:]=5*np.tanh(0.2* (x[1:] +0.01*x[1:]**2 - 0.005*(x[1:]-x[:-1])**3 ) )
		#x[4:]= 0.1*x[4:] + 0.3*x[3:-1] + x[0:-4] - 0.01*x[4:]*x[4:]*x[4:] + 0.01*x[3:-1]*x[3:-1]*x[3:-1] - 0.01*x[2:-2]*x[2:-2]*x[2:-2]+ 0.01*x[1:-3]*x[1:-3]*x[1:-3]- 0.01*x[0:-4]*x[0:-4]*x[0:-4] - 0.15*x[4:]*x[1:-3]
		#x=np.convolve(x, np.array([0,0,0.9,0.07,0.03]), mode='same')
		y=x



	if(0):#Read input data
		datastart=0
		datalength=2000000
		wdin='Volterra_Input_Data'
		filename='6_Output_mono'
		print('File: '+filename+'.csv')
		df=pd.read_csv(wdin+'/'+filename+'.csv', sep=',', header=None, dtype=str)#Oscilloscope Data

		#normal file
		times		= np.transpose(df.values[0:])[0].astype(float)
		x  			= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
		y			= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
		#decorrelated file
		#times		= np.transpose(df.values[0:])[0].astype(float)
		#xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
		#x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
		#y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row
		
		#x=np.tanh(x/1200)
		#y=np.tanh(y/1200)
		
		x=x[::3]
		y=y[::3]
		times=times[::3]
		#Check if data is too long or to short
		if (len(times)>=datastart+datalength):
			times=times[datastart:datastart+datalength]
			x=x[datastart:datastart+datalength]
			y=y[datastart:datastart+datalength]
		datalength=len(times)
		n=datalength
		x-=np.mean(x)



    
	maxlength=n-length3
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=n-actuallength


	print('Correlation determination: ')
	print('input: 		'+str(n))
	print('used:  		'+str(actuallength))
	print('chunklength:	'+str(chunklength))
	print('chunks:		'+str(chunks))
	print('begin index:	'+str(begin))
	print('')
	print('')
	print('')
    
	
	seconds1=time.time()
	
	alpha=np.array([0, 0.6, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1])
	length=np.array([50,20,15,10,10,9,9])
	
	#xm = deCorrelateFast(times, x, y, 0.5, 1.0, 1.0, length1, length2, length3, write=True, name='test', iterations=45, debug=False)
	#xm, re = deCorrelateFast(times, x, y, 0.5, 1.0, 1.0, length1, length2, length3, write=True, name='test', iterations=30, chunks=chunks, debug=True)
	xm, re = deCorrelateFast(times, x, y, alpha, length, write=True, name='test', iterations=100, chunks=chunks, debug=True)
	#xm, re = deCorrelateFast(times, x, y, 0.0, 0.0, 0.0, 0.0, 0.0, length1, length2, length3, length4, length5, write=True, name='test', iterations=100, chunks=chunks, debug=True)
	
	
	seconds2=time.time()
	print('Execution Time:')
	print(seconds2-seconds1)
	
	
	plt.plot(np.log10(re[0]), label='2-point', color='royalblue')
	plt.plot(np.log10(re[1]), label='3-point', color='springgreen')
	plt.plot(np.log10(re[2]), label='4-point', color='chartreuse')
	plt.plot(np.log10(re[3]), label='5-point', color='yellow')
	plt.plot(np.log10(re[4]), label='6-point', color='orange')
	plt.plot(np.log10(re[5]), label='7-point', color='red')
	plt.plot(np.log10(re[6]), label='8-point', color='violet')
	
	plt.grid()
	plt.legend()
	plt.title(r'Correlation MSE: $\frac{\sqrt{\sum_{\tau_1,...,\tau_p} \left(\epsilon^{(p)}-c_{yy}^{(p)}\right)^2}}{T_p^p \cdot \sigma_x^p}$')#np.sqrt(np.sum(err3**2))/(length2**2) / np.var(xmodified)**2
	plt.ylabel(r'$\mathrm{log_{10}(MSE)}$')
	plt.xlabel('Iteration')
	plt.show()
	
	



	#x=np.convolve(x, np.array([1,-1]), mode='same')
	#xm=np.convolve(xm, np.array([1,-1]), mode='same')







	if(1):
		ax1=plt.subplot(212)
		f,p = sg.welch(x, fs=1.0, nperseg=1000)
		f,pm = sg.welch(xm, fs=1.0, nperseg=1000)
		ax1.plot (p, color='orange', label='input')
		ax1.plot (pm, color='lightblue', label='output')
		ax1.legend()

		range=4
		vals=np.linspace(-range,range,1000)
		vals2=vals
		from scipy.stats import norm
		from scipy.stats import gaussian_kde
		kernel = gaussian_kde(x,bw_method=0.01)
		PDFx=kernel(vals)

		kernel2 = gaussian_kde(xm,bw_method=0.01)
		PDFx2=kernel2(vals2)

		PDFExpectedX=norm.pdf(vals, loc=0, scale=np.sqrt(np.var(x)))#ideal
		PDFExpectedX2=norm.pdf(vals2, loc=0, scale=np.sqrt(np.var(xm)))

		ax2=plt.subplot(221)
		ax2.plot(vals, PDFx, color='orange', label='input')
		ax2.plot(vals, PDFExpectedX, color='red', linestyle='dotted')
		ax2.legend()

		ax3=plt.subplot(222)
		ax3.plot(vals2, PDFx2, color='lightblue', label='output')
		ax3.plot(vals2, PDFExpectedX2, color='blue', linestyle='dotted')
		ax3.legend()
		plt.show()
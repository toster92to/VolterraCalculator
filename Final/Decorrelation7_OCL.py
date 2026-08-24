import pyopencl as cl
import numpy as np
import numexpr as ne
import time
import matplotlib.pyplot as plt
import scipy.signal as sg
import pandas as pd
import math
import orthoMatrices

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
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC2 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC2(
		__global float *X,
		__global uint *tau1,
		__global float *RES,
		uint cvectorlength, uint chunklength, uint chunks, uint begin
		)
	{
	// ID
		int T = get_global_id(0);
		
		int chk = T/cvectorlength;
		int t1= tau1[T%cvectorlength];
		int resIndex=T;
		
		RES[resIndex]=0;
		float temp=0;
		float tempOld=0;
		float err=0;
		float c;

		for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
			c= X[ind] * X[ind-t1];//slower, more accurate
			tempOld=temp;
			temp += c + err;
			err = (tempOld-temp)+c+err;
			//temp += X[ind] * X[ind-t1];//faster, less accurate
		}
		RES[resIndex]=temp;
			
	}

	__kernel void corr2(
		__global float *X,
		__global float *HCORR,
		__global float *Y,
		ushort length1
		)
	{
	// ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		Y[index]=0;
		
		int i=0;
		for(int t1=0; t1<length1; t1++){
		if(index>=t1){
			float hcorr=HCORR[i];
			Ytemp += hcorr*X[index-t1];//Reuse X[i-t1] and X[i-t2]
			i+=1;
		}}
		Y[index] = Ytemp;
	}
	""").build(cache_dir=None)	
	


defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC3 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC3(
		__global float *X,
		__global uint *tau1,
		__global uint *tau2,
		__global float *RES,
		uint cvectorlength, uint chunklength, uint chunks, uint begin
		)
	{
	// ID
		int T = get_global_id(0);
		
		int chk = T/cvectorlength;
		int t1= tau1[T%cvectorlength];
		int t2= tau2[T%cvectorlength];
		int resIndex=T;
		
		RES[resIndex]=0;
		float temp=0;
		float tempOld=0;
		float err=0;
		float c;

		for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
			c= X[ind] * X[ind-t1] * X[ind-t2];//slower, more accurate
			tempOld=temp;
			temp += c + err;
			err = (tempOld-temp)+c+err;
			//temp += X[ind] * X[ind-t1] * X[ind-t2];//faster, less accurate
		}
		RES[resIndex]=temp;
			
	}

	__kernel void corr3(
		__global float *X,
		__global float *HCORR,
		__global float *Y,
		ushort length2
		)
	{
	// ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		Y[index]=0;
		
		int i=0;
		for(int t1=0; t1<length2; t1++){
		float Xtemp1=X[index-t1];
		for(int t2=t1; t2<length2; t2++){
		if(index>=t2){
			float hcorr=HCORR[i];
			Ytemp += hcorr*Xtemp1*X[index-t2];//Reuse X[i-t1] and X[i-t2]
			i+=1;
		}}}
		Y[index] = Ytemp;
	}
	""").build(cache_dir=None)	
	
	
	

defineExtChkStr=''
#defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length	
prgC4 = cl.Program(ctx, defineExtChkStr + """	
	//extern queue_t get_default_queue();
	//extern queue_t enqueue_kernel();


	__kernel void calcC4(
		__global float *X,
		__global uint *tau1,
		__global uint *tau2,
		__global uint *tau3,
		__global float *RES,
		uint cvectorlength, uint chunklength, uint chunks, uint begin
		)
	{
	// ID
		int T = get_global_id(0);
		
		int chk = T/cvectorlength;
		int t1= tau1[T%cvectorlength];
		int t2= tau2[T%cvectorlength];
		int t3= tau3[T%cvectorlength];
		int resIndex=T;
		
		RES[resIndex]=0;
		float temp=0;
		float tempOld=0;
		float err=0;
		float c;

		for(int ind = begin + chk*chunklength; ind < begin + chk*chunklength+chunklength; ind++){
			c= X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3];//slower, more accurate
			tempOld=temp;
			temp += c + err;
			err = (tempOld-temp)+c+err;
			//temp += X[ind] * X[ind-t1] * X[ind-t2] * X[ind-t3];//faster, less accurate
		}
		RES[resIndex]=temp;
			
	}

	__kernel void corr4(
		__global float *X,
		__global float *HCORR,
		__global float *Y,
		ushort length3
		)
	{
	// ID		
		int index = get_global_id(0);
		float Ytemp = 0;

		Y[index]=0;
		
		int i=0;
		for(int t1=0; t1<length3; t1++){
		float Xtemp1=X[index-t1];
		for(int t2=t1; t2<length3; t2++){
		float Xtemp2=X[index-t2];
		for(int t3=t2; t3<length3; t3++){
		if(index>=t3){
			float hcorr=HCORR[i];
			Ytemp += hcorr*Xtemp1*Xtemp2*X[index-t3];//Reuse X[i-t1] and X[i-t2]
			i+=1;
		}}}}
		Y[index] = Ytemp;
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


#------------------------------------
def pyExpValList(order, length, Variance):#Creates a "reduced" type list of all the expected values of x(n)x(n-tau1)x(n-tau2)... for all non-redundant combinations of tau
	
	index, taus = orthoMatrices.index2taus(order=order, length=length)
	expVal=np.zeros(len(index))
	
	zero=list(np.atleast_1d(np.array([0])))
	for i in index:	
		expVal[i]= pyExpVal([*zero,*taus[:,i]], np.sqrt(Variance))	
	return expVal	

# print(pyExpValList(order=3,length=3,Variance=1))
# while(1):
# pass

	
length=(15,15,15,3,3,3,3)
A1inv,A2inv,A3inv,A4inv,A5inv,A6inv,A7inv = orthoMatrices.getInvOrthoMatrices(length, write=True)
# A3inv=A3inv.astype(np.float32)

# print(np.linalg.cond(A3inv, p=np.inf))

def estimateTargetVar(length1,length2,length3, C2redRaw,C3redRaw,C4redRaw, xm):#This function calculates the variance of a signal xm after its non-ideality is removed: var(x) with xm=x+ h1*x + ... + h3(...)*x^3.
	#C4red is the "reduced" variant of C4(tau1,tau2,tau3)
	
	newtonI=12#Maximum Newton Iterations
	residual=np.zeros(newtonI)
	targetVar=np.zeros(newtonI)
	
	C2red=np.copy(C2redRaw)
	C3red=np.copy(C3redRaw)
	C4red=np.copy(C4redRaw)
	#C4 is calculated later
	

	
	targetVar[0]=np.var(xm)#initial choice for the variance
	
	#Load the indices and taus of each order into memory
	index1,taus1_1 = orthoMatrices.index2taus(order=1,length=length1)
	taus1_1=taus1_1[0]
	index2,(taus1_2,taus2_2) = orthoMatrices.index2taus(order=2,length=length2)
	index3,(taus1_3,taus2_3,taus3_3) = orthoMatrices.index2taus(order=3,length=length3)
	# index4,(taus1,taus2,taus3,taus4) = orthoMatrices.index2taus(order=4,length=length4)
	# index5,(taus1,taus2,taus3,taus4,taus5) = orthoMatrices.index2taus(order=5,length=length5)
	# index6,(taus1,taus2,taus3,taus4,taus5,taus6)= orthoMatrices.index2taus(order=6,length=length6)
	# index7,(taus1,taus2,taus3,taus4,taus5,taus6,taus7) = orthoMatrices.index2taus(order=7,length=length7)	
	
	A13=orthoMatrices.calcHtoCMatrix(length=length3, orderH=1, orderC=3)
	
	for k in np.arange(newtonI-1):
		
		expVal2x= pyExpValList(order=1, length=length1, Variance=targetVar[k]) 
		err2=(C2red-expVal2x).astype(np.float32)
		k1est=np.matmul(A1inv, err2)/np.var(xm)**1#reduced first estimate for h3
		k1est=orthoMatrices.redToRedPerm(k1est, order=1, length=length1)#convert to reduced*nPerm version	
	
		
		expVal3x= pyExpValList(order=2, length=length2, Variance=targetVar[k]) 
		err3=(C3red-expVal3x).astype(np.float32)	
		k2est=np.matmul(A2inv, err3)/np.var(xm)**2#reduced first estimate for h3
		k2est=orthoMatrices.redToRedPerm(k2est, order=2, length=length2)#convert to reduced*nPerm version	
		
		
		C4red = C4redRaw - np.matmul(A13, k1est) * targetVar[k]**2#Orthogonalisation with respect to h1
		expVal4x= pyExpValList(order=3, length=length3, Variance=targetVar[k]) 
		err4=(C4red-expVal4x).astype(np.float32)
		k3est=np.matmul(A3inv, err4)/np.var(xm)**3#reduced first estimate for h3
		k3est=orthoMatrices.redToRedPerm(k3est, order=3, length=length3)#convert to reduced*nPerm version
		#C2red=C2redRaw - np.matmul(A31, h3est) * targetVar[k]**2#Orthogonalisation with respect to h1
		
		# expVal5x= pyExpValList(order=3, length=length4, Variance=targetVar[k]) 
		# err5=(C5red-expVal5x).astype(np.float32)
		
		# expVal6x= pyExpValList(order=3, length=length5, Variance=targetVar[k]) 
		# err6=(C6red-expVal6x).astype(np.float32)
		
		# expVal7x= pyExpValList(order=3, length=length6, Variance=targetVar[k]) 
		# err7=(C7red-expVal7x).astype(np.float32)
		
		# expVal8x= pyExpValList(order=3, length=length7, Variance=targetVar[k]) 
		# err8=(C8red-expVal8x).astype(np.float32)

		#Wiener to Volterra Conversion
		h1est=k1est
		for i in np.arange(len(k1est)):
			for j in np.arange(len(k3est)):
				if taus1_3[j] == taus2_3[j]:
					if taus1_1[i]==taus3_3[j]:
						h1est[i] += - 3*targetVar[k]*k3est[j]
						print(taus1_1[i], taus1_3[j],taus2_3[j],taus3_3[j])
				elif taus2_3[j]==taus3_3[j]:
					if taus1_1[i]==taus1_3[j]:
						h1est[i] += - 3*targetVar[k]*k3est[j]
						print(taus1_1[i], taus1_3[j],taus2_3[j],taus3_3[j])
				
				
		h2est=k2est
		h3est=k3est
	
		covarError1=0
		varError1=0
		covarError2=0
		varError2=0
		covarError3=0
		varError3=0

		#1st order
		for j in index1: #Go through all indices of the covariance function, and add up their contributions
			if j>=1:#Skip the 1st index, since this is just the target variance, which is already part of the equation.
				covarError1+=2*h1est[j]*orthoMatrices.pyExpVal((0, taus1_1[j]), targetVar[k])#The variance introduced by the covariance between x and x^3
				varError1+=h1est[j]**2*orthoMatrices.pyExpVal((taus1_1[j],taus1_1[j]), targetVar[k])#The variance introduced by the variance of x^3. This term is usually so small that it can be neglected
				#instead of using pyExpVal, later, a global array could be used, saving a lot of calculations.
		#2nd order
		for j in index2: #Go through all indices of the covariance function, and add up their contributions
			covarError2+=2*h2est[j]*orthoMatrices.pyExpVal((0, taus1_2[j],taus2_2[j]), targetVar[k])#The variance introduced by the covariance between x and x^3
			varError2+=h2est[j]**2*orthoMatrices.pyExpVal((taus1_2[j],taus1_2[j],taus2_2[j],taus2_2[j]), targetVar[k])#The variance introduced by the variance of x^3. This term is usually so small that it can be neglected

		#3rd order
		for j in index3: #Go through all indices of the covariance function, and add up their contributions
			covarError3+=2*h3est[j]*orthoMatrices.pyExpVal((0, taus1_3[j],taus2_3[j],taus3_3[j]), targetVar[k])#The variance introduced by the covariance between x and x^3
			varError3+=h3est[j]**2*orthoMatrices.pyExpVal((taus1_3[j],taus1_3[j],taus2_3[j],taus2_3[j],taus3_3[j],taus3_3[j]), targetVar[k])#The variance introduced by the variance of x^3. This term is usually so small that it can be neglected
			
		residual[k]= np.var(xm) - targetVar[k] - covarError1-varError1-covarError2-varError2-covarError3-varError3 #let "var(xm)-var(x)-covars=0" converge towards zero
		
		# print(covarError1)
		# print(varError1)
		# print(covarError2)
		# print(varError2)
		# print(covarError3)
		# print(varError3)
		# # print(targetVar[k])
		# # print(np.var(xm))
		# print('')
		
		np.set_printoptions(threshold=500000, linewidth=3000, precision=5, suppress=True)
		print(h1est)
		print(taus1_1)
		print(h2est)
		print(taus1_2)
		print(taus2_2)
		print(h3est)
		print(taus1_3)
		print(taus2_3)
		print(taus3_3)
		print('')
		
		
		#Interrupt condition, if sufficiently good
		if abs(residual[k])<1e-7*targetVar[k]: return targetVar[k]
		
		if k==0: targetVar[k+1] = targetVar[k] + 1*residual[k]#pass#Calculate 1st point with given start value
		if k>=1: targetVar[k+1] = targetVar[k] - residual[k]*(targetVar[k]-targetVar[k-1])/(residual[k]-residual[k-1])

	return targetVar[-1]
#-----------------------------------------------------------------------------------------------------------------	
#2--------------------------------
def pyCalcC2(xm, length1, chunks=1024):
	maxlength=len(xm)-length1
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength
	
	#create result array	
	index, tau1 = orthoMatrices.index2taus(order=1, length=length1)
	c=np.zeros(chunks*len(index), dtype=np.float32)
	tau1=tau1.astype(np.uint32)
	
	xm=xm.astype(np.float32)
	#create input and output buffers
	xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)	
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
	tau1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau1)

	#call kernels
	prgC2.calcC2(queue, c.shape, None, xm_buf, tau1_buf, c_buf, np.uint32(len(index)), np.uint32(chunklength), np.uint32(chunks), np.uint32(begin))
	
	#fill C2 with results from calculation and reshape
	C2 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C2, c_buf)
	C2=C2.reshape(chunks,len(index))
	
	#return mean symmetrized version
	C2mean=np.sum(C2, axis=0)/actuallength
	
	return(C2mean)



def pyRemoveC2(xm, hcorr, length1):   

	#Use the right data types
	xm=xm.astype(np.float32)
	hcorr=hcorr.astype(np.float32)
	xmout = np.empty_like(xm)
	
	#Create buffers
	hcorr_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hcorr)
	xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
	
	prgC2.corr2(queue, np.shape(xm), None, xm_buf, hcorr_buf, xmout_buf, np.uint16(length1))
	cl.enqueue_copy(queue, xmout, xmout_buf)

	return(xmout)
	


def estimateTargetVar2(length1, C2red, xm):#This function calculates the variance of a signal xm after its 2rd order distortion is removed: var(x) with xm=x+h2(...)*x^2.
	#C2red is the "reduced" variant of C2(tau1,tau2,tau2)
	
	newtonI=12#Maximum Newton Iterations
	residual=np.zeros(newtonI)
	targetVar=np.zeros(newtonI)
	
	targetVar[0]=np.var(xm)#initial choice for the variance
	
	
	for k in np.arange(newtonI-1):
		expVal2x= pyExpValList(order=1, length=length1, Variance=targetVar[k]) 
		err2=(C2red-expVal2x).astype(np.float32)
		
		h2est=np.matmul(A2inv, err3)/np.var(xm)**2#reduced first estimate for h2
		h2est=orthoMatrices.redToRedPerm(h2est, order=1, length=length1)#convert to reduced*nPerm version	
	
		covarError=0
		varError=0
		index, tau1 = orthoMatrices.index2taus(order=1,length=length1)
		for j in index: #Go through all indices of the covariance function, and add up their contributions
			covarError+=2*h1est[j]*orthoMatrices.pyExpVal((0, tau1[j]), targetVar[k])#The variance introduced by the covariance between x and x^2
			varError+=h1est[j]**2*orthoMatrices.pyExpVal((tau1[j],tau1[j]), targetVar[k])#The variance introduced by the variance of x^2. This term is usually so small that it can be neglected
		residual[k]= np.var(xm) - targetVar[k] - covarError-varError #let "var(xm)-var(x)-covars=0" converge towards zero
		
		#Interrupt condition, if sufficiently good
		if abs(residual[k])<1e-6*targetVar[k]: return targetVar[k]
		
		if k==0: targetVar[k+1] = targetVar[k] + 1*residual[k]#pass#Calculate 1st point with given start value
		if k>=1: targetVar[k+1] = targetVar[k] - residual[k]*(targetVar[k]-targetVar[k-1])/(residual[k]-residual[k-1])

	return targetVar[-1]
	
	
	

def decrease_c2x(iterations, xm, chunks, length1, alpha1, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in np.arange(iterations):	
		xm=xm-np.mean(xm)
		C2=pyCalcC2(xm, length1, chunks)
		
		#targetVar=estimateTargetVar2(length1,C2,xm)

		expVal2x= pyExpValList(order=1, length=length1, Variance=targetVar) 
		err2=(C2-expVal2x).astype(np.float32)

		hcorr=np.matmul(A1inv, err2)/targetVar**1
		hcorr=orthoMatrices.redToRedPerm(hcorr, order=1, length=length1)
		

		xm= xm -  alpha1*pyRemoveC2(xm, hcorr, length1)	
	return(xm, err2)	
#3----------------------------------	
	
def pyCalcC3(xm, length2, chunks=1023):
	maxlength=len(xm)-length2
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength
	
	#create result array	
	index, ([tau1,tau2]) = orthoMatrices.index2taus(order=2, length=length2)
	c=np.zeros(chunks*len(index), dtype=np.float32)
	tau1=tau1.astype(np.uint32)
	tau2=tau2.astype(np.uint32)
	
	xm=xm.astype(np.float32)
	#create input and output buffers
	xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)	
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
	tau1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau1)
	tau2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau2)

	#call kernels
	prgC3.calcC3(queue, c.shape, None, xm_buf, tau1_buf, tau2_buf, c_buf, np.uint32(len(index)), np.uint32(chunklength), np.uint32(chunks), np.uint32(begin))
	
	#fill c3 with results from calculation and reshape
	C3 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C3, c_buf)
	C3=C3.reshape(chunks,len(index))
	
	#return mean symmetrized version
	C3mean=np.sum(C3, axis=0)/actuallength
	
	return(C3mean)



def pyRemoveC3(xm, hcorr, length2):   

	#Use the right data types
	xm=xm.astype(np.float32)
	hcorr=hcorr.astype(np.float32)
	xmout = np.empty_like(xm)
	
	#Create buffers
	hcorr_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hcorr)
	xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
	
	prgC3.corr3(queue, np.shape(xm), None, xm_buf, hcorr_buf, xmout_buf, np.uint16(length2))
	cl.enqueue_copy(queue, xmout, xmout_buf)

	return(xmout)
	


def estimateTargetVar3(length2, C3red, xm):#This function calculates the variance of a signal xm after its 2rd order distortion is removed: var(x) with xm=x+h2(...)*x^2.
	#C3red is the "reduced" variant of C3(tau1,tau2,tau2)
	
	newtonI=12#Maximum Newton Iterations
	residual=np.zeros(newtonI)
	targetVar=np.zeros(newtonI)
	
	targetVar[0]=np.var(xm)#initial choice for the variance
	
	
	for k in np.arange(newtonI-1):
		expVal3x= pyExpValList(order=2, length=length2, Variance=targetVar[k]) 
		err3=(C3red-expVal3x).astype(np.float32)
		
		h2est=np.matmul(A2inv, err3)/np.var(xm)**2#reduced first estimate for h2
		h2est=orthoMatrices.redToRedPerm(h2est, order=2, length=length2)#convert to reduced*nPerm version	
	
		covarError=0
		varError=0
		index,(tau1,tau2,tau2) = orthoMatrices.index2taus(order=2,length=length2)
		for j in index: #Go through all indices of the covariance function, and add up their contributions
			covarError+=2*h2est[j]*orthoMatrices.pyExpVal((0, tau1[j],tau2[j]), targetVar[k])#The variance introduced by the covariance between x and x^2
			varError+=h2est[j]**2*orthoMatrices.pyExpVal((tau1[j],tau1[j],tau2[j],tau2[j]), targetVar[k])#The variance introduced by the variance of x^2. This term is usually so small that it can be neglected
		residual[k]= np.var(xm) - targetVar[k] - covarError-varError #let "var(xm)-var(x)-covars=0" converge towards zero
		
		#Interrupt condition, if sufficiently good
		if abs(residual[k])<1e-7*targetVar[k]: return targetVar[k]
		
		if k==0: targetVar[k+1] = targetVar[k] + 1*residual[k]#pass#Calculate 1st point with given start value
		if k>=1: targetVar[k+1] = targetVar[k] - residual[k]*(targetVar[k]-targetVar[k-1])/(residual[k]-residual[k-1])

	return targetVar[-1]
	
	
	

def decrease_c3x(iterations, xm, chunks, length2, alpha2, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in np.arange(iterations):	
		xm=xm-np.mean(xm)
		C3=pyCalcC3(xm, length2, chunks)
		
		#targetVar=estimateTargetVar3(length2,C3,xm)

		expVal3x= pyExpValList(order=2, length=length2, Variance=targetVar) 
		err3=(C3-expVal3x).astype(np.float32)

		hcorr=np.matmul(A2inv, err3)/targetVar**1.5
		hcorr=orthoMatrices.redToRedPerm(hcorr, order=2, length=length2)
		

		xm= xm -  alpha2*pyRemoveC3(xm, hcorr, length2)	
	return(xm, err3)	
#4---------------------------------	
	
def pyCalcC4(xm, length3, chunks=1024):
	maxlength=len(xm)-length3
	actuallength=int(maxlength/chunks)*chunks
	chunklength=int(maxlength/chunks)
	begin=len(xm)-actuallength
	
	#create result array	
	index, ([tau1,tau2,tau3]) = orthoMatrices.index2taus(order=3, length=length3)
	c=np.zeros(chunks*len(index), dtype=np.float32)
	tau1=tau1.astype(np.uint32)
	tau2=tau2.astype(np.uint32)
	tau3=tau3.astype(np.uint32)
	
	xm=xm.astype(np.float32)
	#create input and output buffers
	xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)	
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
	tau1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau1)
	tau2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau2)
	tau3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau3)

	#call kernels
	prgC4.calcC4(queue, c.shape, None, xm_buf, tau1_buf, tau2_buf, tau3_buf, c_buf, np.uint32(len(index)), np.uint32(chunklength), np.uint32(chunks), np.uint32(begin))
	
	#fill c4 with results from calculation and reshape
	C4 = np.zeros(np.shape(c)).astype(np.float32)
	cl.enqueue_copy(queue, C4, c_buf)
	C4=C4.reshape(chunks,len(index))
	
	#return mean symmetrized version
	C4mean=np.sum(C4, axis=0)/actuallength
	
	return(C4mean)



def pyRemoveC4(xm, hcorr, length3):   

	#Use the right data types
	xm=xm.astype(np.float32)
	hcorr=hcorr.astype(np.float32)
	xmout = np.empty_like(xm)
	
	#Create buffers
	hcorr_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hcorr)
	xm_buf = cl.Buffer\
	   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm) 
	xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)
	
	prgC4.corr4(queue, np.shape(xm), None, xm_buf, hcorr_buf, xmout_buf, np.uint16(length3))
	cl.enqueue_copy(queue, xmout, xmout_buf)

	return(xmout)
	


def estimateTargetVar4(length3, C4red, xm):#This function calculates the variance of a signal xm after its 3rd order distortion is removed: var(x) with xm=x+h3(...)*x^3.
	#C4red is the "reduced" variant of C4(tau1,tau2,tau3)
	
	newtonI=12#Maximum Newton Iterations
	residual=np.zeros(newtonI)
	targetVar=np.zeros(newtonI)
	
	targetVar[0]=np.var(xm)#initial choice for the variance
	
	
	for k in np.arange(newtonI-1):
		expVal4x= pyExpValList(order=3, length=length3, Variance=targetVar[k]) 
		err4=(C4red-expVal4x).astype(np.float32)
		
		h3est=np.matmul(A3inv, err4)/np.var(xm)**3#reduced first estimate for h3
		h3est=orthoMatrices.redToRedPerm(h3est, order=3, length=length3)#convert to reduced*nPerm version	
	
		covarError=0
		varError=0
		index,(tau1,tau2,tau3) = orthoMatrices.index2taus(order=3,length=length3)
		for j in index: #Go through all indices of the covariance function, and add up their contributions
			covarError+=2*h3est[j]*orthoMatrices.pyExpVal((0, tau1[j],tau2[j],tau3[j]), targetVar[k])#The variance introduced by the covariance between x and x^3
			varError+=h3est[j]**2*orthoMatrices.pyExpVal((tau1[j],tau1[j],tau2[j],tau2[j],tau3[j],tau3[j]), targetVar[k])#The variance introduced by the variance of x^3. This term is usually so small that it can be neglected
		residual[k]= np.var(xm) - targetVar[k] - covarError-varError #let "var(xm)-var(x)-covars=0" converge towards zero
		
		#Interrupt condition, if sufficiently good
		if abs(residual[k])<1e-6*targetVar[k]: return targetVar[k]
		
		if k==0: targetVar[k+1] = targetVar[k] + 1*residual[k]#pass#Calculate 1st point with given start value
		if k>=1: targetVar[k+1] = targetVar[k] - residual[k]*(targetVar[k]-targetVar[k-1])/(residual[k]-residual[k-1])

	return targetVar[-1]
	
	
	

def decrease_c4x(iterations, xm, chunks, length3, alpha3, targetVar=0):
	if targetVar==0: targetVar=np.var(xm)#if target variance is not given, use the input variance instead. In this case, variance will not be preserved. 
	for i in np.arange(iterations):	
		xm=xm-np.mean(xm)
		C4=pyCalcC4(xm, length3, chunks)
		
		#targetVar=estimateTargetVar4(length3,C4,xm)

		expVal4x= pyExpValList(order=3, length=length3, Variance=targetVar) 
		err4=(C4-expVal4x).astype(np.float32)

		hcorr=np.matmul(A3inv, err4)/targetVar**3
		hcorr=orthoMatrices.redToRedPerm(hcorr, order=3, length=length3)
		

		xm= xm -  alpha3*pyRemoveC4(xm, hcorr, length3)	
	return(xm, err4)	



x=np.random.normal(0,1,1000000)
#x[:100]=x[:100]*0.9

x[1:]=x[1:]+0.01*x[:-1]+0.02*x[1:]**2+0.01*x[:-1]**3
#x[1:]=x[1:] + 0.1*x[:-1]
#x-=np.mean(x)
#tvar=np.var(x)

# index,(tau1,tau2,tau3) = orthoMatrices.index2taus(order=3,length=15)
# print(tau1)
# print(tau2)
# print(tau3)

# C2red=pyCalcC2(x, length[0], chunks=1024)
# C3red=pyCalcC3(x, length[1], chunks=1024)
# C4red=pyCalcC4(x, length[2], chunks=1024)


# #print(pyExpVal((0, 2), 1))

# print(estimateTargetVar(length[0],length[1],length[2], C2red,C3red,C4red, x))




for j in np.arange(45):
	#if j%5==0: x,err2=decrease_c2x(1, x, chunks=1000, length1=15, alpha1=0.5, targetVar=1)
	x,err4=decrease_c4x(1, x, chunks=1000, length3=15, alpha3=0.5, targetVar=1)
	#re2=np.sqrt(np.sum(err2**2))/(len(err2)) / np.var(x)**1	
	re4=np.sqrt(np.sum(err4**2))/(len(err4)) / np.var(x)**2		
	#print(re2, re4)
	print(re4)

while(1):
	pass

#------------------------------------
# def pyExpVal8(length7, Variance):
	# expVal=np.zeros(shape=(length7, length7, length7, length7, length7, length7, length7))
	
	# for [tau1, tau2, tau3, tau4, tau5, tau6, tau7] in list(combinations_with_replacement(np.arange(length7), 7)):
		# expVal[tau1,tau2,tau3,tau4,tau5,tau6,tau7] = pyExpVal(list([0,tau1,tau2,tau3,tau4,tau5,tau6,tau7]), np.sqrt(Variance))
	
		# perm=list(set(permutations([tau1, tau2, tau3, tau4, tau5, tau6, tau7])))
		# for [t1, t2, t3, t4, t5, t6, t7] in perm:
			# expVal[t1,t2,t3,t4,t5,t6,t7]=expVal[tau1,tau2,tau3,tau4,tau5,tau6,tau7]#Fill eqivalent kernel positions with copied entries
		
	# return expVal	





	


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
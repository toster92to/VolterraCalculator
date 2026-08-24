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

# Initialize context and queue
ctx, queue = create_context_and_queue(device_type='GPU')
mf = cl.mem_flags

# The kernel is defined next.
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


defineExtChkStr = ''
# defineExtChkStr='#define EXTCHK '+str(length3+chunklength2)#Defines the extended chunk length
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


defineExtChkStr = ''
# defineExtChkStr='#define EXTCHK '+str(length3+chunklength2)#Defines the extended chunk length
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


defineExtChkStr = ''
# defineExtChkStr='#define LENGTH3 '+str(length3)#Defines the extended chunk length
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


# --------------------------------------------------------------------------------------------


def pyExpVal2(length1, Variance):
    expVal = np.zeros(shape=(length1))
    for tau1 in np.arange(length1):
        if tau1 == 0:
            expVal[tau1] = Variance

    return expVal


def pyCalcC2(xm, length1, chunks=1024):

    maxlength = len(xm)-length1
    actuallength = int(maxlength/chunks)*chunks
    chunklength = int(maxlength/chunks)
    begin = len(xm)-actuallength

# create result array
    c = np.zeros((chunks*length1), dtype=np.float32)
    xm = xm.astype(np.float32)
# create input and output buffers
    xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)


# call kernels
    prgC2.calcC2(queue, c.shape, None, xm_buf, c_buf, np.uint16(
        length1), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))

# fill c4 with results from calculation and reshape
    C2 = np.zeros(np.shape(c)).astype(np.float32)
    cl.enqueue_copy(queue, C2, c_buf)

    C2 = C2.reshape(chunks, length1)/actuallength

# print(C2)

# return reduced and symmetrized version
    C2reduced = np.sum(C2, axis=0)

    return (C2reduced)


def pyRemoveC2(xm, err, length1, alpha):

    n = len(xm)
    xmout = np.empty_like(xm)

    # create the array containing the multiplication factors for kernel symmetry and covariance
    fac = np.zeros((length1), dtype=np.float32)
    for [tau1] in list(combinations_with_replacement(np.arange(length1), 1)):
        fac[tau1] = 1

    # Use the right data types
    fac = fac.reshape(length1).astype(np.float32)
    err = err.reshape(length1).astype(np.float32)

    # Create buffers
    err_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
    fac_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
    xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)
    xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)

    # alpha=0.1
    prgC2.removeC2(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf,
                   xmout_buf, np.uint16(length1), np.uint32(n), np.float32(alpha))
    cl.enqueue_copy(queue, xmout, xmout_buf)

    # print(np.var(xmout))
    # print(np.var(xm-xmout))
    return (xmout)


def decrease_c2x(iterations, xm, chunks, length1, alpha1, targetVar=0):
    if targetVar == 0:
        # if target variance is not given, use the input variance instead. In this case, variance will not be preserved.
        targetVar = np.var(xm)
    for i in range(iterations):
        xm = (xm-np.mean(xm)).astype(np.float32)
        C2 = pyCalcC2(xm, length1, chunks)
        expVal2x = pyExpVal2(length1, targetVar)
        err2 = (C2-expVal2x).astype(np.float32)
        xm = pyRemoveC2(xm, err2, length1, alpha1/targetVar)
        # print(np.sqrt(np.sum(err2**2))/(length1**2) / np.var(xm)**2)
    return (xm, err2)


# ------------------------------------------
def pyExpVal3(length2, Variance):
    expVal = np.zeros(shape=(length2, length2))

    return expVal  # c_xxx should be zero for all tau1,tau2 for gaussian input.


def pyCalcC3(xm, length2, chunks=1024):

    maxlength = len(xm)-length2
    actuallength = int(maxlength/chunks)*chunks
    chunklength = int(maxlength/chunks)
    begin = len(xm)-actuallength

    # create result array
    c = np.zeros((chunks*length2**2), dtype=np.float32)
    xm = xm.astype(np.float32)
    # create input and output buffers
    xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

    # call kernels
    prgC3.calcC3(queue, c.shape, None, xm_buf, c_buf, np.uint16(
        length2), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))

    # fill c4 with results from calculation and reshape
    C3 = np.zeros(np.shape(c)).astype(np.float32)
    cl.enqueue_copy(queue, C3, c_buf)
    C3 = C3.reshape(chunks, length2, length2)/actuallength

    # return reduced and symmetrized version
    C3reduced = np.sum(C3, axis=0)
    for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):
        perm = list(set(permutations([tau1, tau2])))
        for [t1, t2] in perm:
            # Fill eqivalent kernel positions with copied entries
            C3reduced[t1, t2] = C3reduced[tau1, tau2]
    return (C3reduced)


def pyRemoveC3(xm, err, length2, alpha):

    n = len(xm)
    xmout = np.empty_like(xm)

    # create the array containing the multiplication factors for kernel symmetry and covariance
    fac = np.zeros((length2, length2), dtype=np.float32)
    for [tau1, tau2] in list(combinations_with_replacement(np.arange(length2), 2)):
        divider = 0
        divider += kernels.calcDiagonalFactor((tau1, tau2))
        if tau1 == 0:
            divider += kernels.calcDiagonalFactor((0, tau2))
        if tau2 == 0:
            divider += kernels.calcDiagonalFactor((tau1, 0))
        fac[tau1, tau2] = 1/divider
        perm = list(set(permutations([tau1, tau2])))
        for [t1, t2] in perm:
            fac[t1, t2] = fac[tau1, tau2]

    # Use the right data types
    fac = fac.reshape(length2**2).astype(np.float32)
    err = err.reshape(length2**2).astype(np.float32)

    # Create buffers
    err_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
    fac_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
    xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)
    xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)

    # alpha=0.1
    prgC3.removeC3(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf,
                   xmout_buf, np.uint16(length2), np.uint32(n), np.float32(alpha))
    cl.enqueue_copy(queue, xmout, xmout_buf)

    # print(np.var(xmout))
    # print(np.var(xm-xmout))
    return (xmout)


def decrease_c3x(iterations, xm, chunks, length2, alpha2, targetVar=0):
    if targetVar == 0:
        # if target variance is not given, use the input variance instead. In this case, variance will not be preserved.
        targetVar = np.var(xm)
    for i in range(iterations):
        xm = (xm-np.mean(xm)).astype(np.float32)
        C3 = pyCalcC3(xm, length2, chunks)
        expVal3x = pyExpVal3(length2, targetVar)
        err3 = (C3-expVal3x).astype(np.float32)
        xm = pyRemoveC3(xm, err3, length2, alpha2/targetVar**2)
    return (xm, err3)


# ------------------------------------
def pyExpVal4(length3, Variance):
    expVal = np.zeros(shape=(length3, length3, length3))

    for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
        # set: get rid of duplicates(in the case of multiple taus being the same), permutations: find all possible combinations of tau1,2,3
        perm = list(set(permutations([tau1, tau2, tau3])))

        if tau1 == 0:
            if tau2 == tau3:
                expVal[tau1, tau2, tau3] = Variance**2
                if tau2 == 0:
                    expVal[tau1, tau2, tau3] = 3*Variance**2

        for [t1, t2, t3] in perm:
            # Fill eqivalent kernel positions with copied entries
            expVal[t1, t2, t3] = expVal[tau1, tau2, tau3]

    return expVal


def pyCalcC4(xm, length3, chunks=1024):

    maxlength = len(xm)-length3
    actuallength = int(maxlength/chunks)*chunks
    chunklength = int(maxlength/chunks)
    begin = len(xm)-actuallength

    # create result array
    c = np.zeros((chunks*length3**3), dtype=np.float32)
    xm = xm.astype(np.float32)
    # create input and output buffers
    xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

    # call kernels
    prgC4.calcC4(queue, c.shape, None, xm_buf, c_buf, np.uint16(
        length3), np.uint32(chunklength), np.uint16(chunks), np.uint32(begin))

    # fill c4 with results from calculation and reshape
    C4 = np.zeros(np.shape(c)).astype(np.float32)
    cl.enqueue_copy(queue, C4, c_buf)
    C4 = C4.reshape(chunks, length3, length3, length3)/actuallength

    # return reduced and symmetrized version
    C4reduced = np.sum(C4, axis=0)
    for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
        perm = list(set(permutations([tau1, tau2, tau3])))
        for [t1, t2, t3] in perm:
            # Fill eqivalent kernel positions with copied entries
            C4reduced[t1, t2, t3] = C4reduced[tau1, tau2, tau3]
    return (C4reduced)


def pyRemoveC4(xm, err, length3, alpha):
    n = len(xm)
    xmout = np.empty_like(xm)

    # create the array containing the multiplication factors for kernel symmetry and covariance
    fac = np.zeros((length3, length3, length3), dtype=np.float32)
    for [tau1, tau2, tau3] in list(combinations_with_replacement(np.arange(length3), 3)):
        divider = 0
        divider += kernels.calcDiagonalFactor((tau1, tau2, tau3))
        if tau1 == 0:
            divider += kernels.calcDiagonalFactor((0, tau2, tau3))
        if tau2 == 0:
            divider += kernels.calcDiagonalFactor((tau1, 0, tau3))
        if tau3 == 0:
            divider += kernels.calcDiagonalFactor((tau1, tau2, 0))
        fac[tau1, tau2, tau3] = 1/divider
        perm = list(set(permutations([tau1, tau2, tau3])))
        for [t1, t2, t3] in perm:
            fac[t1, t2, t3] = fac[tau1, tau2, tau3]

    # Use the right data types
    fac = (fac.reshape(length3**3)).astype(np.float32)
    err = (err.reshape(length3**3)).astype(np.float32)

    # Create buffers
    err_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=err)
    fac_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fac)
    xm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xm)
    xmout_buf = cl.Buffer(ctx, mf.READ_WRITE, xmout.nbytes)

    # alpha=0.1
    prgC4.removeC4(queue, np.shape(xm), None, xm_buf, err_buf, fac_buf,
                   xmout_buf, np.uint16(length3), np.uint32(n), np.float32(alpha))
    cl.enqueue_copy(queue, xmout, xmout_buf)

    # print(np.var(xmout))
    # print(np.var(xm-xmout))
    return (xmout)


def decrease_c4x(iterations, xm, chunks, length3, alpha3, targetVar=0):
    if targetVar == 0:
        # if target variance is not given, use the input variance instead. In this case, variance will not be preserved.
        targetVar = np.var(xm)
    for i in range(iterations):
        xm = (xm-np.mean(xm)).astype(np.float32)
        C4 = pyCalcC4(xm, length3, chunks)
        expVal4x = pyExpVal4(length3, targetVar)
        err4 = (C4-expVal4x).astype(np.float32)
        xm = pyRemoveC4(xm, err4, length3, alpha3/targetVar**3)
    return (xm, err4)

# def decrease_c4x(iterations, xm, chunks, length3, alpha3):
    # for i in range(iterations):
    # xm=(xm-np.mean(xm)).astype(np.float32)
    # C4=pyCalcC4(xm, length3, chunks)
    # expVal4x= pyExpVal4(length3, np.var(xm))
    # err4=(C4-expVal4x).astype(np.float32)
    # xm=pyRemoveC4(xm, err4, length3, alpha3/np.var(xm)**3)
    # return(xm, err4)


# ---------------------------------------------------------------------------------------------------------------
def deCorrelateFast(times, x, y, alpha1=1, alpha2=1, alpha3=1, length1=50, length2=10, length3=10, iterations=15, iterations_c2=5, iterations_initial=35, write=True, name='', chunks=1024, debug=False):
    # Higher values for alpha increase the speed of convergence, but too high values lead to instability

    xmodified = 100*np.copy(x)  # Make better use of 32bit float range
    tvar = np.var(xmodified)  # target variance of the output signal

    re2 = np.zeros(iterations)
    re3 = np.zeros(iterations)
    re4 = np.zeros(iterations)

    if (not debug):
        xmodified, _ = decrease_c2x(
            iterations_initial, xmodified, chunks, length1, 0.3, targetVar=tvar)  # 0.3

    for i in np.arange(iterations):
        # 1st order
        xmodified, _ = decrease_c2x(
            iterations_c2, xmodified, chunks, length1, alpha1, targetVar=tvar)
        # xmodified=np.clip(xmodified,-5,5)
# 2nd order
        xmodified, err3 = decrease_c3x(
            1, xmodified, chunks, length2, alpha2, targetVar=tvar)
        # xmodified=np.clip(xmodified,-5,5)
# 1st order again
        xmodified, err2 = decrease_c2x(
            iterations_c2, xmodified, chunks, length1, alpha1, targetVar=tvar)
        # xmodified=np.clip(xmodified,-5,5)
# 3rd order
        xmodified, err4 = decrease_c4x(
            1, xmodified, chunks, length3, alpha3, targetVar=tvar)
        # xmodified=np.clip(xmodified,-5,5)

        # Print relative errors
        re2[i] = np.sqrt(np.sum(err2**2))/(length1**1) / np.var(xmodified)
        re3[i] = np.sqrt(np.sum(err3**2))/(length2**2) / np.var(xmodified)**1.5
        re4[i] = np.sqrt(np.sum(err4**2))/(length3**3) / np.var(xmodified)**2
        print('Iteration {it:>3}: 2pt: {a:5.2e} ; 3pt: {b:5.2e} ; 4pt: {c:5.2e}       '.format(
            it=i, a=re2[i], b=re3[i], c=re4[i]))

    # Spectrum has priority
    xmodified, _ = decrease_c2x(
        3*iterations_c2, xmodified, chunks, length1, alpha1, targetVar=tvar)

    xmodified = xmodified/100  # Make better use of 32bit float range

    if (write):
        with open(name+'_dec.csv', mode='w', encoding="utf-8", newline='') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n in np.arange(len(x)):
                # csv_writer.writerow([times[n], xmodified[n], y[n], x[n]])
                csv_writer.writerow([times[n], xmodified[n], x[n], y[n]])

    if (debug):
        return (xmodified, (re2, re3, re4))
    else:
        return (xmodified)
# -----------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    length1 = 50
    length2 = 20
    length3 = 20
    (T1, T2, T3) = (length3, length3, length3)

    # use so many chunks that chunklength<1e6 (or better, <1e5),  to reduce float rounding errors and make better use of float16 dynamic range
    chunks = 512
    n = 2**22+chunks
    # n=4194304

    # chunks2=10000
    # chunklength2=100

    if (1):  # Random input data
        x = np.random.normal(loc=0.0, scale=1.0, size=n)
        x = x.astype(np.float32)

        timestep = 100e-12
        times = np.arange(n)

        # x=np.convolve(x, np.array([0,0,0.9,0.07,0.03]), mode='same')
        # x[1:]=5*np.tanh(0.2* (x[1:] +0.01*x[1:]**2 - 0.005*(x[1:]-x[:-1])**3 ) )
        x[4:] = 0.1*x[4:] + 0.3*x[3:-1] + x[0:-4] - 0.01*x[4:]*x[4:]*x[4:] + 0.01*x[3:-1]*x[3:-1]*x[3:-1] - 0.01 * \
            x[2:-2]*x[2:-2]*x[2:-2] + 0.01*x[1:-3]*x[1:-3]*x[1:-3] - \
            0.01*x[0:-4]*x[0:-4]*x[0:-4] - 0.15*x[4:]*x[1:-3]
        x = np.convolve(x, np.array([0, 0, 0.9, 0.07, 0.03]), mode='same')
        y = x

    if (0):  # Read input data
        datastart = 0
        datalength = 2000000
        wdin = 'Volterra_Input_Data'
        filename = 'LMH3401XY00'
        print('File: '+filename+'.csv')
        df = pd.read_csv(wdin+'/'+filename+'.csv', sep=',',
                         header=None, dtype=str)  # Oscilloscope Data

        # normal file
        times = np.transpose(df.values[0:])[0].astype(float)
        x = np.transpose(df.values[0:])[1].astype(float)  # Start with 0th row
        y = np.transpose(df.values[0:])[2].astype(float)  # Start with 0th row
        # decorrelated file
        # times		= np.transpose(df.values[0:])[0].astype(float)
        # xmodified	= np.transpose(df.values[0:])[1].astype(float)#Start with 0th row
        # x  		= np.transpose(df.values[0:])[2].astype(float)#Start with 0th row
        # y			= np.transpose(df.values[0:])[3].astype(float)#Start with 0th row

        x = x[::3]
        y = y[::3]
        times = times[::4]
        # Check if data is too long or to short
        if (len(times) >= datastart+datalength):
            times = times[datastart:datastart+datalength]
            x = x[datastart:datastart+datalength]
            y = y[datastart:datastart+datalength]
        datalength = len(times)
        n = datalength
        x -= np.mean(x)

    maxlength = n-length3
    actuallength = int(maxlength/chunks)*chunks
    chunklength = int(maxlength/chunks)
    begin = n-actuallength

    print('Correlation determination: ')
    print('input: 		'+str(n))
    print('used:  		'+str(actuallength))
    print('chunklength:	'+str(chunklength))
    print('chunks:		'+str(chunks))
    print('begin index:	'+str(begin))
    print('')
    print('')
    print('')

    seconds1 = time.time()

    # xm = deCorrelateFast(times, x, y, 0.5, 1.0, 1.0, length1, length2, length3, write=True, name='test', iterations=45, debug=False)
    xm, re = deCorrelateFast(times, x, y, 0.5, 1.0, 1.0, length1, length2, length3,
                             write=True, name='test', iterations=30, chunks=chunks, debug=True)

    seconds2 = time.time()
    print('Execution Time:')
    print(seconds2-seconds1)

    plt.plot(np.log10(re[0]), label='2-point', color='royalblue')
    plt.plot(np.log10(re[1]), label='3-point', color='springgreen')
    plt.plot(np.log10(re[2]), label='4-point', color='chartreuse')
    plt.grid()
    plt.legend()
    # np.sqrt(np.sum(err3**2))/(length2**2) / np.var(xmodified)**2
    plt.title(
        r'Correlation MSE: $\frac{\sqrt{\sum_{\tau_1,...,\tau_p} \left(\epsilon^{(p)}-c_{yy}^{(p)}\right)^2}}{T_p^p \cdot \sigma_x^p}$')
    plt.ylabel(r'$\mathrm{log_{10}(MSE)}$')
    plt.xlabel('Iteration')
    plt.show()

    # x=np.convolve(x, np.array([1,-1]), mode='same')
    # xm=np.convolve(xm, np.array([1,-1]), mode='same')

    if (1):
        ax1 = plt.subplot(212)
        f, p = sg.welch(x, fs=1.0, nperseg=1000)
        f, pm = sg.welch(xm, fs=1.0, nperseg=1000)
        ax1.plot(p, color='orange', label='input')
        ax1.plot(pm, color='lightblue', label='output')
        ax1.legend()

        range = 4
        vals = np.linspace(-range, range, 1000)
        vals2 = vals
        from scipy.stats import norm
        from scipy.stats import gaussian_kde
        kernel = gaussian_kde(x, bw_method=0.01)
        PDFx = kernel(vals)

        kernel2 = gaussian_kde(xm, bw_method=0.01)
        PDFx2 = kernel2(vals2)

        PDFExpectedX = norm.pdf(vals, loc=0, scale=np.sqrt(np.var(x)))  # ideal
        PDFExpectedX2 = norm.pdf(vals2, loc=0, scale=np.sqrt(np.var(xm)))

        ax2 = plt.subplot(221)
        ax2.plot(vals, PDFx, color='orange', label='input')
        ax2.plot(vals, PDFExpectedX, color='red', linestyle='dotted')
        ax2.legend()

        ax3 = plt.subplot(222)
        ax3.plot(vals2, PDFx2, color='lightblue', label='output')
        ax3.plot(vals2, PDFExpectedX2, color='blue', linestyle='dotted')
        ax3.legend()
        plt.show()

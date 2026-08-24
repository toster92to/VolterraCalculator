import numpy as np
import pandas as pd
import csv
import os
from Volterra_Operations_noncausal import chainVolterraSystems_opencl, addVolterraSystems, subtractVolterraSystems, multiplyVolterraSystem, invertKernels
from Volterra_In_Out_7 import VolterraResponse
import copy

# Handles saving/reading of Volterra System files, that contain the kernels.



def readKernels(name='', folderName=''):
	wdin = 'Volterra_Kernels'
	# Read Kernel files
	df = pd.read_csv(wdin+folderName+'/'+'h0'+name+'.csv',
					 sep='\t', header=None, dtype=str)
	h0raw = (df.values[0:]).astype(float)

	df = pd.read_csv(wdin+folderName+'/'+'h1'+name+'.csv',
					 sep='\t', header=None, dtype=str)
	h1raw = (df.values[0:]).astype(float)

	df = pd.read_csv(wdin+folderName+'/'+'h2'+name+'.csv',
					 sep='\t', header=None, dtype=str)
	h2raw = (df.values[0:]).astype(float)

	df = pd.read_csv(wdin+folderName+'/'+'h3'+name+'.csv',
					 sep='\t', header=None, dtype=str)
	h3raw = (df.values[0:]).astype(float)

	df = pd.read_csv(wdin+folderName+'/'+'h4'+name+'.csv',
					 sep='\t', header=None, dtype=str)
	h4raw = (df.values[0:]).astype(float)

	df = pd.read_csv(wdin+folderName+'/'+'h5'+name+'.csv',
					 sep='\t', header=None, dtype=str)
	h5raw = (df.values[0:]).astype(float)

	# Fill Kernel arrays with input data from kernel files.
	# gnoise_inverted=gn_invraw

	# 0th order kernel
	h0raw = h0raw[0:]
	h0 = h0raw

	# 1st order kernel
	l1 = int(h1raw[0])  # Get length of kernel
	l1neg = int(h1raw[1])  # Get length of kernel
	l1Total=l1+l1neg
	
	h1raw = h1raw[2:]
	h1 = h1raw

	# 2nd order kernel
	l2 = int(h2raw[0])
	l2neg = int(h2raw[1])
	l2Total=l2+l2neg
	h2raw = h2raw[2:]
	h2 = np.zeros(shape=(l2Total, l2Total))
	for i in np.arange(l2Total):
		for j in np.arange(l2Total):
			h2[i, j] = h2raw[i+l2Total*j]

	# 3rd order kernel
	l3 = int(h3raw[0])
	l3neg = int(h3raw[1])
	l3Total=l3+l3neg
	h3raw = h3raw[2:]
	h3 = np.zeros(shape=(l3Total, l3Total, l3Total))
	for i in np.arange(l3Total):
		for j in np.arange(l3Total):
			for k in np.arange(l3Total):
				h3[i, j, k] = h3raw[i+l3Total*j+l3Total*l3Total*k]

	# 4th order kernel
	l4 = int(h4raw[0])
	l4neg = int(h4raw[1])
	l4Total=l4+l4neg
	h4raw = h4raw[2:]
	h4 = np.zeros(shape=(l4Total, l4Total, l4Total, l4Total))
	for i in np.arange(l4Total):
		for j in np.arange(l4Total):
			for k in np.arange(l4Total):
				for l in np.arange(l4Total):
					h4[i, j, k, l] = h4raw[i+l4Total*j+l4Total *
										   l4Total*k+l4Total*l4Total*l4Total*l]

	# 5th order kernel
	l5 = int(h5raw[0])
	l5neg = int(h5raw[1])
	l5Total=l5+l5neg
	h5raw = h5raw[2:]
	h5 = np.zeros(shape=(l5Total, l5Total, l5Total, l5Total, l5Total))
	for i in np.arange(l5Total):
		for j in np.arange(l5Total):
			for k in np.arange(l5Total):
				for l in np.arange(l5Total):
					for m in np.arange(l5Total):
						h5[i, j, k, l, m] = h5raw[i+l5Total*j+l5Total *
												  l5Total*k+l5Total*l5Total*l5Total*l+l5Total*l5Total*l5Total*l5Total*m]

	h0 = h0[0][0]
	h1 = h1[:, 0]
	h2 = h2[:, :]
	h3 = h3[:, :, :]
	h4 = h4[:, :, :, :]
	h5 = h5[:, :, :, :, :]
	system = VolterraResponse(use_opencl=True)
	system.set_kernels(h0, h1, h2, h3, h4, h5, np.array([l1neg, l2neg,l3neg,l4neg,l5neg]))
	return system


# def writeKernels(h0, h1, h2, h3, h4, h5, name=''):
def writeKernels(system, name=''):
	h0 = system.h0
	h1 = system.h1
	h2 = system.h2
	h3 = system.h3
	h4 = system.h4
	h5 = system.h5
	# Check if the directory exists
	if not os.path.exists('Volterra_Kernels'):
		# If not, create it
		os.makedirs('Volterra_Kernels')

	# prepare raw data for write
	wdout = 'Volterra_Kernels'
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
	

	h1raw = np.zeros(l1Total+2)
	h2raw = np.zeros(l2Total*l2Total+2)
	h3raw = np.zeros(l3Total*l3Total*l3Total+2)
	h4raw = np.zeros(l4Total*l4Total*l4Total*l4Total+2)
	h5raw = np.zeros(l5Total*l5Total*l5Total*l5Total*l5Total+2)

	h1raw[0] = l1
	h1raw[1] = l1neg
	
	h2raw[0] = l2
	h2raw[1] = l2neg
	
	h3raw[0] = l3
	h3raw[1] = l3neg
	
	h4raw[0] = l4
	h4raw[1] = l4neg
	
	h5raw[0] = l5
	h5raw[1] = l5neg

	h0raw = h0

	for n in np.arange(l1Total):
		h1raw[n+2] = h1[n]

	for n in np.arange(l2Total*l2Total):
		h2raw[n+2] = h2[n % l2Total, n // l2Total]

	for n in np.arange(l3Total*l3Total*l3Total):
		h3raw[n+2] = h3[n % l3Total, (n // l3Total) %
						l3Total, n // (l3Total*l3Total)]

	for n in np.arange(l4Total*l4Total*l4Total*l4Total):
		h4raw[n+2] = h4[n % l4Total, (n // l4Total) % l4Total, (n // (
			l4Total*l4Total)) % l4Total, n // (l4Total*l4Total*l4Total)]

	for n in np.arange(l5Total*l5Total*l5Total*l5Total*l5Total):
		h5raw[n+2] = h5[n % l5Total, (n // l5Total) % l5Total, (n // (l5Total*l5Total)) % l5Total, (n // (
			l5Total*l5Total*l5Total)) % l5Total, n // (l5Total*l5Total*l5Total*l5Total)]

	with open(wdout+'/'+'h0'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow([h0raw])

	with open(wdout+'/'+'h1'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h1raw)):
			csv_writer.writerow([h1raw[n]])

	with open(wdout+'/'+'h2'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h2raw)):
			csv_writer.writerow([h2raw[n]])

	with open(wdout+'/'+'h3'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h3raw)):
			csv_writer.writerow([h3raw[n]])

	with open(wdout+'/'+'h4'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h4raw)):
			csv_writer.writerow([h4raw[n]])

	with open(wdout+'/'+'h5'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h5raw)):
			csv_writer.writerow([h5raw[n]])

# ------------------------------------------------------------------------------------------
# File-level Kernel combination operations


def invertKernelFiles(nameIn, nameOut, skew=0):
	print('Inverting h'+nameIn+'.csv')
	Hinv = invertKernels(readKernels(name=nameIn), skew)
	writeKernels(Hinv, name=nameOut)
	print('Writing h'+nameOut+'.csv')


def combineKernelFiles(nameInA, nameInB, nameOut):  # System A is followed by system B
	print('Combining h'+nameInA+'.csv and h'+nameInB+'.csv')
	HsourceInv = readKernels(name=nameInA)
	Hcombined = readKernels(name=nameInB)
	Hsystem = chainVolterraSystems_opencl(HsourceInv, Hcombined)
	writeKernels(Hsystem, name=nameOut)
	print('Writing h'+nameOut+'.csv')


def addKernelFiles(nameInA, nameInB, nameOut):  # System A is followed by system B
	print('Adding h'+nameInA+'.csv and h'+nameInB+'.csv')
	sysA = readKernels(name=nameInA)
	sysB = readKernels(name=nameInB)
	sysC = addVolterraSystems(sysA, sysB)
	writeKernels(sysC, name=nameOut)
	print('Writing h'+nameOut+'.csv')

# System A is followed by system B


def subtractKernelFiles(nameInA, nameInB, nameOut):
	print('Subtracting h'+nameInA+'.csv from h'+nameInB+'.csv')
	sysA = readKernels(name=nameInA)
	sysB = readKernels(name=nameInB)
	sysC = subtractVolterraSystems(sysA, sysB)
	writeKernels(sysC, name=nameOut)
	print('Writing h'+nameOut+'.csv')


def multiplyKernelFiles(nameInA, factor, nameOut):  # System A is followed by system B
	print('Multiplying h'+nameInA+'.csv')
	sysA = readKernels(name=nameInA)
	sysC = multiplyVolterraSystem(sysA, factor)
	writeKernels(sysC, name=nameOut)
	print('Writing h'+nameOut+'.csv')

# --------------------------------------------------------
# old version, using tuples

# def invertKernelFiles(nameIn, nameOut, skew=0):
#	 print('Inverting h'+nameIn+'.csv')
#	 h0, h1, h2, h3 = readKernels(name=nameIn)
#	 Hinv = invertKernels(h0, h1, h2, h3, skew)
#	 writeKernels(*Hinv, name=nameOut)
#	 print('Writing h'+nameOut+'.csv')


# def combineKernelFiles(nameInA, nameInB, nameOut):  # System A is followed by system B
#	 print('Combining h'+nameInA+'.csv and h'+nameInB+'.csv')
#	 HsourceInv = readKernels(name=nameInA)
#	 Hcombined = readKernels(name=nameInB)
#	 Hsystem = chainVolterraSystems(*HsourceInv, *Hcombined)
#	 writeKernels(*Hsystem, name=nameOut)
#	 print('Writing h'+nameOut+'.csv')


# def addKernelFiles(nameInA, nameInB, nameOut):  # System A is followed by system B
#	 print('Adding h'+nameInA+'.csv and h'+nameInB+'.csv')
#	 sysA = readKernels(name=nameInA)
#	 sysB = readKernels(name=nameInB)
#	 sysC = addVolterraSystems(*sysA, *sysB)
#	 writeKernels(*sysC, name=nameOut)
#	 print('Writing h'+nameOut+'.csv')


# # System A is followed by system B
# def subtractKernelFiles(nameInA, nameInB, nameOut):
#	 print('Subtracting h'+nameInA+'.csv from h'+nameInB+'.csv')
#	 sysA = readKernels(name=nameInA)
#	 sysB = readKernels(name=nameInB)
#	 sysC = subtractVolterraSystems(*sysA, *sysB)
#	 writeKernels(*sysC, name=nameOut)
#	 print('Writing h'+nameOut+'.csv')


# def multiplyKernelFiles(nameInA, factor, nameOut):  # System A is followed by system B
#	 print('Multiplying h'+nameInA+'.csv')
#	 sysA = readKernels(name=nameInA)
#	 sysC = multiplyVolterraSystem(*sysA, factor)
#	 writeKernels(*sysC, name=nameOut)
#	 print('Writing h'+nameOut+'.csv')


# --------------------------------------------------------
if __name__ == '__main__':
	# invertKernelFiles(nameIn='DUT', nameOut='invDUT', skew=5)
	# combineKernelFiles(nameInA='invDUT', nameInB='DUT', nameOut='test')
	# addKernelFiles(nameInA='pos', nameInB='neg', nameOut='add')
	# subtractKernelFiles(nameInA='pos', nameInB='neg', nameOut='sub')
	# multiplyKernelFiles(nameInA='invDUT', nameInB='DUT', nameOut='test')
	# combineKernelFiles(nameInA='invSource', nameInB='overall', nameOut='DUT')
	
	from itertools import combinations_with_replacement, permutations
	
	GPT=readKernels(name='GPT', folderName='')
	
	GPTout=VolterraResponse(use_opencl=True, kernelName="GPTout", kernelPath="", profiling_data=False)
	GPTout.h0 = 0
	GPTout.h1 = np.zeros(80)
	GPTout.h2 = np.zeros(shape=(40, 40))
	GPTout.h3 = np.zeros(shape=(40, 40, 40))
	GPTout.h4 = np.zeros(shape=(20, 20, 20, 20))
	GPTout.h5 = np.zeros(shape=(10, 10, 10, 10, 10))

	GPTout.len1 = len(GPTout.h1)
	GPTout.len2 = len(GPTout.h2)
	GPTout.len3 = len(GPTout.h3)
	GPTout.len4 = len(GPTout.h4)
	GPTout.len5 = len(GPTout.h5)

	
	for tau1 in np.arange(40):
		for tau2 in np.arange(tau1, 40):
			for t1,t2,t3 in list(set(permutations([0,tau1,tau2]))):
				GPTout.h3[t1,t2,t3] = GPT.h2[tau1,tau2]/len(set(permutations([t1,t2,t3])))
			
			
			
	writeKernels(GPTout, name='GPTout')
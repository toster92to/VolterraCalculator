import numpy as np
import pandas as pd
import csv

#Handles saving/reading of Volterra System files, that contain the kernels.


def readKernels(name='', folderName=''):
	wdin='Volterra_Kernels'+folderName
	# df=pd.read_csv(wdin+'/'+'gn_inv.csv', sep='\t', header=None, dtype=str)
	# gn_invraw=np.transpose(df.values[1:]).astype(float)
	# gnoise_inverted=gn_invraw

	df=pd.read_csv(wdin+'/'+'h0'+name+'.csv', sep='\t', header=None, dtype=str)
	h0raw=(df.values[0:]).astype(float)

	df=pd.read_csv(wdin+'/'+'h1'+name+'.csv', sep='\t', header=None, dtype=str)
	h1raw=(df.values[0:]).astype(float)

	df=pd.read_csv(wdin+'/'+'h2'+name+'.csv', sep='\t', header=None, dtype=str)
	h2raw=(df.values[0:]).astype(float)

	df=pd.read_csv(wdin+'/'+'h3'+name+'.csv', sep='\t', header=None, dtype=str)
	h3raw=(df.values[0:]).astype(float)



	#Fill Kernel arrays with input data from kernel files.
	#gnoise_inverted=gn_invraw

	#0th order kernel
	h0raw=h0raw[0:]
	h0=h0raw

	#1st order kernel
	length1=int(h1raw[0])#Get length of kernel
	h1raw=h1raw[1:]
	h1=h1raw
		
	#2nd order kernel	
	length2=int(h2raw[0])
	h2raw=h2raw[1:]	
	h2=np.zeros(shape=(length2,length2))
	for i in np.arange(length2):
		for j in np.arange(length2):
			h2[i,j]=h2raw[i+length2*j]

	#3rd order kernel
	length3=int(h3raw[0])
	h3raw=h3raw[1:]
	h3=np.zeros(shape=(length3, length3, length3))
	for i in np.arange(length3):
		for j in np.arange(length3):
			for k in np.arange(length3):
				h3[i,j,k]=h3raw[i+length3*j+length3*length3*k]
		

	h0=h0[0][0]	

	h1=h1[:,0]
	H=h0,h1,h2,h3
	return H

def writeKernels(h0, h1, h2, h3, name=''):
	#prepare raw data for write
	wdout='Volterra_Kernels'
	#gn_invraw=np.zeros(length1+1)
	length1=len(h1)
	length2=len(h2)
	length3=len(h3)

	h1raw=np.zeros(length1+1)
	h2raw=np.zeros(length2*length2+1)
	h3raw=np.zeros(length3*length3*length3+1)

	#gn_invraw[0]=length1
	h1raw[0]=length1
	h2raw[0]=length2
	h3raw[0]=length3

	#for n in np.arange(length1):
	#	gn_invraw[n+1]=gnoise_inverted[n]

	h0raw=h0

	for n in np.arange(length1):
		h1raw[n+1]=h1[n]

	for n in np.arange(length2*length2):
		h2raw[n+1]=h2[n%length2, int(n/length2)]

	for n in np.arange(length3*length3*length3):
		h3raw[n+1]=h3[n%length3, int(n/length3)%length3, int(int(n/length3)/length3)]
		
	#wd='Volterra_Kernels'
	#write to files
	# with open(wdout+'/'+'gn_inv.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		# csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		# for n in np.arange(len(gn_invraw)):
			# csv_writer.writerow([gn_invraw[n]])

	with open(wdout+'/'+'h0'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow([h0raw])

	with open(wdout+'/'+'h1'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h1raw)):
			csv_writer.writerow([h1raw[n]])

	with open(wdout+'/'+'h2'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h2raw)):
			csv_writer.writerow([h2raw[n]])

	with open(wdout+'/'+'h3'+name+'.csv', mode='w', encoding="utf-8", newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for n in np.arange(len(h3raw)):
			csv_writer.writerow([h3raw[n]])	

#------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math

from functools import lru_cache

from itertools import combinations_with_replacement
from itertools import permutations



@lru_cache(maxsize=128)
def expValXpowN(sigma, N):
	#https://math.stackexchange.com/questions/2751719/expected-value-of-xn-for-normal-distribution
	#returns the expected value of E[X^N], where X is a Normal(0,sigma^2) distributet random variable. 
	if N%2!=0: 
		return(0)
	
	if N==0:
		return(1)
	else: 
		return((N-1)*sigma**2 * expValXpowN(sigma, N-2))
		

#New, more general implementation of the expected value of a product of multiple time-shifted X. The time shifts are given by tauList = (tau1,tau2,...)
def pyExpVal(tauList, sigma):
	dictionary = dict((i, tauList.count(i)) for i in tauList)

	product=1
	for key in dictionary:
		product=product * expValXpowN(sigma, dictionary[key])
	return(product)


def calcPermutations(tauList):#Calculates the number of permutations at tauList = (tau1,tau2,...)
	dictionary = dict((i, tauList.count(i)) for i in tauList)

	product=math.factorial(len(tauList))
	for key in dictionary:
		product=product/math.factorial(dictionary[key])
	return(product)


def calcIndividualIndices(order, length):#Calculates the number of non-redundant kernel entries. See Alae Mrani's master thesis, p.18
	result=math.factorial(order+length-1)//math.factorial(length-1)//math.factorial(order)
	return(result)
	
	
	
def index2taus(order, length):#Generate vectors of index[i] and tau1[i],taus[i],...=taus[:,i] for all possible non-redundant combinations of indices. Could be useful for vectorized GPU calculations, where taus are given by vectors
	listLength=calcIndividualIndices(order, length)
	
	taus=np.zeros(shape=(listLength, order), dtype='int')
	index=np.zeros(listLength, dtype='int')
	
	for i in np.arange(1,listLength):
		taus[i,:]=taus[i-1,:]
		index[i]=i
		taus[i,order-1]+=1
		
		
		for j in np.arange(order-1):#go through all taus, from the right
			if taus[i,order-j-1] >= length:
				taus[i,order-j-2]+=1
				
				if j==0:
					taus[i,order-j-1] = taus[i,order-j-2]
				else:
					taus[i,order-j-1:] = np.ones(j+1)*taus[i,order-j-2]
	
	taus=np.transpose(taus)
	return index,taus
	
# index, ([tau1,tau2,tau3]) = index2Tau(order=3, length=5)
# print(tau1)
# print(tau2)
# print(tau3)

#Kernel data format conversions-----------------------------
def normToRed(c, order, length):#Converts a normal (symmetrical) Kernel ndarray into a reduced kernel vector, which only contains non-redundant information. Permutations are not evaluated.
	#Also, returns an array containing all the combinations auf taus at vector index i, which can be accessed by tau1[i],tau2[i],... = taus[:,i]
	index, taus = index2taus(order,length)
	cred=np.zeros(len(index))
	
	
	for i in index:
		cred[i]=c[tuple(taus[:,i])]
	return(cred, taus)
		

def redToRedPerm(cred, order, length):#Converts a reduced kernel vector into a reduced nPerm kernel vector, taking into account the number of permutations. 
	credPerm=np.zeros(len(cred))
	index,taus=index2taus(order,length)
	
	for i in np.arange(len(cred)):
		credPerm[i]=cred[i]*calcPermutations(taus[:,i].tolist())
	return(credPerm)
	
	
def redPermToNorm(credPerm, order, length):#Converts a reduced nPerm kernel vector into a normal/symmetrical kernel ndarray.
	index,taus=index2taus(order,length)
	
	for i in index:
		nPerms=calcPermutations(taus[:,i].tolist())
		for tauPerms in list(set(permutations(tuple(taus[:,i])))):
			c[tuple(tauPerms)]=credPerm[i]/nPerms
	return c
	
#Some test cases for kernel conversions
# c=np.zeros(shape=(5,5,5))
# c[0,1,1]=1
# c[1,0,1]=1
# c[1,1,0]=1
# cred, taus=normToRed(c,order=3,length=5)
# credPerm=redToRedPerm(cred, order=3, length=5)
# cBar=redPermToNorm(credPerm, order=3, length=5)#Should be equal to c, if c was symmetrical
# print(cBar)

#-------------------------------
	

def printLengths(length1,length2,length3,length4,length5,length6,length7):
	print('Matrix edge length info')
	print('A1:' + str(calcIndividualIndices(1, length1)))
	print('A2:' + str(calcIndividualIndices(2, length2)))
	print('A3:' + str(calcIndividualIndices(3, length3)))
	print('A4:' + str(calcIndividualIndices(4, length4)))
	print('A5:' + str(calcIndividualIndices(5, length5)))
	print('A6:' + str(calcIndividualIndices(6, length6)))
	print('A7:' + str(calcIndividualIndices(7, length7)))
	
	


def calcExpectedSensitivity(vList, tauList):#calculate d cxxx(vList) / d h(tauList); tau are the indices of the kernel causing the correlation. 
	newDivider=0
	for i in np.arange(0,1+len(np.atleast_1d(vList))):
		if i==0: 
			vListTemp=np.atleast_1d(vList)
			tauListTemp=np.atleast_1d(tauList)	
			joinedList=np.concatenate((vListTemp,tauListTemp)).tolist()
		else:
			vListTemp=np.atleast_1d(vList)
			tauListTemp=np.atleast_1d(tauList)	
			
			tauListTemp+=vListTemp[i-1]
			vListTemp[i-1]=0
			
			joinedList=np.concatenate((vListTemp,tauListTemp)).tolist()
			
		#print(joinedList)
		newDivider+=pyExpVal(joinedList,sigma=1)
	return newDivider		
	
	
	
def calcHtoCMatrix(length, orderH, orderC):#Calculates the complete sensitivity matrix, transforming from orderH kernel h to orderC covariance C. Both kernel and covariance are given in the "reduced" data format
	#kernel h number of permutations is accounted for. 
	indexV, vaus = index2taus(orderC, length)	
	indexTau, taus = index2taus(orderH, length)
	
	print(taus)
	A=np.zeros(shape=(len(indexV), len(indexTau)))
	
	for iV in indexV:
		for iTau in indexTau:
			tausList=np.atleast_1d(taus[:,iTau]).tolist()
			vausList=np.atleast_1d(vaus[:,iV]).tolist()
			A[iV,iTau]= calcExpectedSensitivity(vausList, tausList)* calcPermutations(tausList)
	return (A)
	
	
# np.set_printoptions(threshold=500000, linewidth=3000, precision=5, suppress=True)
# print(calcHtoCMatrix(length=3, orderH=1, orderC=3))

# while(1):
	# pass


#--------------------------------------------------------------------	


def getOrthoMatrices(length, write=True):
	print('Getting forward matrices')
	
	wdin = 'Ortho_Matrices'
	wdout= 'Ortho_Matrices'

	
	length1=length[0]
	length2=length[1]
	length3=length[2]
	length4=length[3]
	length5=length[4]
	length6=length[5]
	length7=length[6]

	try:#A1 already there? 
		df = pd.read_csv(wdin+'/'+'A1-'+str(length1)+'.csv', sep='\t', header=None, dtype=str)
		A1 = (df.values[0:]).astype(float)
		A1=np.reshape(A1, (math.isqrt(len(A1)),math.isqrt(len(A1))))
		print('A1 already there')
	except:#Not there
		print('A1 is calculated...')
		#A1=	calcHtoCMatrix1(length1)
		A1= calcHtoCMatrix(length=length1, orderH=1, orderC=1)
	
	if(write):
		A1f=A1.flatten()
		np.savetxt(wdout+'/'+'A1-'+str(length1)+'.csv', A1f, delimiter='\r\n', fmt='%1.4g')
		
		
	try:#A2 already there? 
		df = pd.read_csv(wdin+'/'+'A2-'+str(length2)+'.csv', sep='\t', header=None, dtype=str)
		A2 = (df.values[0:]).astype(float)
		A2=np.reshape(A2, (math.isqrt(len(A2)),math.isqrt(len(A2))))
		print('A2 already there')
	except:#Not there
		print('A2 is calculated...')
		A2= calcHtoCMatrix(length=length2, orderH=2, orderC=2)
		
	if(write):
		A2f=A2.flatten()
		np.savetxt(wdout+'/'+'A2-'+str(length2)+'.csv', A2f, delimiter='\r\n', fmt='%1.4g')
		
		
	try:#A3 already there? 
		df = pd.read_csv(wdin+'/'+'A3-'+str(length3)+'.csv', sep='\t', header=None, dtype=str)
		A3 = (df.values[0:]).astype(float)
		A3=np.reshape(A3, (math.isqrt(len(A3)),math.isqrt(len(A3))))
		print('A3 already there')
	except:#Not there
		print('A3 is calculated...')
		A3= calcHtoCMatrix(length=length3, orderH=3, orderC=3)	
		
	if(write):
		A3f=A3.flatten()
		np.savetxt(wdout+'/'+'A3-'+str(length3)+'.csv', A3f, delimiter='\r\n', fmt='%1.4g')
	
	
	try:#A4 already there? 
		df = pd.read_csv(wdin+'/'+'A4-'+str(length4)+'.csv', sep='\t', header=None, dtype=str)
		A4 = (df.values[0:]).astype(float)
		A4=np.reshape(A4, (math.isqrt(len(A4)),math.isqrt(len(A4))))
		print('A4 already there')
	except:#Not there
		print('A4 is calculated...')
		A4= calcHtoCMatrix(length=length4, orderH=4, orderC=4)	
	
	if(write):
		A4f=A4.flatten()
		np.savetxt(wdout+'/'+'A4-'+str(length4)+'.csv', A4f, delimiter='\r\n', fmt='%1.4g')			
		
	
	try:#A5 already there? 
		df = pd.read_csv(wdin+'/'+'A5-'+str(length5)+'.csv', sep='\t', header=None, dtype=str)
		A5 = (df.values[0:]).astype(float)
		A5=np.reshape(A5, (math.isqrt(len(A5)),math.isqrt(len(A5))))
		print('A5 already there')
	except:#Not there
		print('A5 is calculated...')
		A5= calcHtoCMatrix(length=length5, orderH=5, orderC=5)	

	if(write):
		A5f=A5.flatten()
		np.savetxt(wdout+'/'+'A5-'+str(length5)+'.csv', A5f, delimiter='\r\n', fmt='%1.4g')		
				
				
	try:#A6 already there? 
		df = pd.read_csv(wdin+'/'+'A6-'+str(length6)+'.csv', sep='\t', header=None, dtype=str)
		A6 = (df.values[0:]).astype(float)
		A6=np.reshape(A6, (math.isqrt(len(A6)),math.isqrt(len(A6))))
		print('A6 already there')
	except:#Not there
		print('A6 is calculated...')
		A6= calcHtoCMatrix(length=length6, orderH=6, orderC=6)

	if(write):
		A6f=A6.flatten()
		np.savetxt(wdout+'/'+'A6-'+str(length6)+'.csv', A6f, delimiter='\r\n', fmt='%1.4g')		
		

	try:#A7 already there? 
		df = pd.read_csv(wdin+'/'+'A7-'+str(length7)+'.csv', sep='\t', header=None, dtype=str)
		A7 = (df.values[0:]).astype(float)
		A7=np.reshape(A7, (math.isqrt(len(A7)),math.isqrt(len(A7))))
		print('A7 already there')
	except:#Not there
		print('A7 is calculated...')
		A7= calcHtoCMatrix(length=length7, orderH=7, orderC=7)	

	if(write):
		A7f=A7.flatten()
		np.savetxt(wdout+'/'+'A7-'+str(length7)+'.csv', A7f, delimiter='\r\n', fmt='%1.4g')


				
	return(A1,A2,A3,A4,A5,A6,A7)
	
	
def getInvOrthoMatrices(length, write=True):
	print('Getting inverse orthogonal matrices...')
	wdout= 'Ortho_Matrices'

	
	length1=length[0]
	length2=length[1]
	length3=length[2]
	length4=length[3]
	length5=length[4]
	length6=length[5]
	length7=length[6]	
	
	#print('Getting forward matrices...')
	A1,A2,A3,A4,A5,A6,A7=getOrthoMatrices(([length1,length2,length3,length4,length5,length6,length7]))
	
	print('Inverting...')
	A1inv=np.linalg.inv(A1)	

	A2inv=np.linalg.inv(A2)	
		
	A3inv=np.linalg.inv(A3)

	A4inv=np.linalg.inv(A4)

	A5inv=np.linalg.inv(A5)

	A6inv=np.linalg.inv(A6)

	A7inv=np.linalg.inv(A7)
	
	if(write):
		
		A1invf=A1inv.flatten()
		np.savetxt(wdout+'/'+'A1inv-'+str(length1)+'.csv', A1invf, delimiter='\r\n', fmt='%1.4g')	
		
		A2invf=A2inv.flatten()
		np.savetxt(wdout+'/'+'A2inv-'+str(length2)+'.csv', A2invf, delimiter='\r\n', fmt='%1.4g')	
		
		A3invf=A3inv.flatten()
		np.savetxt(wdout+'/'+'A3inv-'+str(length3)+'.csv', A3invf, delimiter='\r\n', fmt='%1.4g')	
		
		A4invf=A4inv.flatten()
		np.savetxt(wdout+'/'+'A4inv-'+str(length4)+'.csv', A4invf, delimiter='\r\n', fmt='%1.4g')	
		
		A5invf=A5inv.flatten()
		np.savetxt(wdout+'/'+'A5inv-'+str(length5)+'.csv', A5invf, delimiter='\r\n', fmt='%1.4g')	
		
		A6invf=A6inv.flatten()
		np.savetxt(wdout+'/'+'A6inv-'+str(length6)+'.csv', A6invf, delimiter='\r\n', fmt='%1.4g')	
		
		A7invf=A7inv.flatten()
		np.savetxt(wdout+'/'+'A7inv-'+str(length7)+'.csv', A7invf, delimiter='\r\n', fmt='%1.4g')	

	return(A1inv,A2inv,A3inv,A4inv,A5inv,A6inv,A7inv)
#---------------------------------------	


if __name__ == '__main__':

	#def c2hkernel(c, A):
	
	
		
	length1=4#50
	length2=4#50
	length3=4#50
	length4=4#20
	length5=4#20
	length6=4#9
	length7=4#9

	printLengths(length1,length2,length3,length4,length5,length6,length7)

	A1,A2,A3,A4,A5,A6,A7=getOrthoMatrices(([length1,length2,length3,length4,length5,length6,length7]))
	A1inv,A2inv,A3inv,A4inv,A5inv,A6inv,A7inv=getInvOrthoMatrices(([length1,length2,length3,length4,length5,length6,length7]))

	#print(calcPermutations([0,0,0,0,1]))
	vaus=([0,0,0])
	taus=np.atleast_1d([0,0,0])
	print(calcExpectedSensitivity(vaus, taus)* calcPermutations(taus.tolist()))
	
	np.set_printoptions(threshold=500000, linewidth=3000, precision=5, suppress=True)
	#print(A3)
	#print(A3inv)



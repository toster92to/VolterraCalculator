import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris
from scipy.signal.windows import blackman
from scipy.signal.windows import gaussian
from scipy.signal.windows import kaiser
from scipy.signal.windows import hann
from scipy.stats import skewnorm



#component='NLResistor'
#component='Inductor'
component='Capacitor'
#component = 'GaussResistor'
#component = 'NLCore'

print('')
print('')
print('')
print('')
print('')
print('Volterra Component Creator')
print('Type: '+ component)

if component=='NLResistor':
	#u(i)= A(i) = a0 + a1 i + a2 i^2 + a3 i^3 + a4 i^4 +a5 i^5
	a=np.zeros(6)
	a[0]=0
	a[1]=1#Linear Resistance
	a[2]=0
	a[3]=0
	a[4]=0
	a[5]=0
	
	print('h0 ='+str(a[0]))
	print('h1[0] ='+str(a[1]))
	print('h2[0,0] ='+str(a[2]))
	print('h3[0,0,0] ='+str(a[3]))
	print('h4[0,0,0,0] ='+str(a[4]))
	print('h5[0,0,0,0,0] ='+str(a[5]))
	
	
if component=='NLCore':#Nonlinear core of an inductor/capacitor
	#L(i) = B(i) L0. Create ZL0 using the Inductor creator, and then compose ZL = ZL0 ° NLCore
	#C(u) = B(u) C0. Create YC0 using the Inductor creator, and then compose YC = YC0 ° NLCore
	a=np.zeros(6)
	a[0]=1
	a[1]=0
	a[2]=-0.1
	a[3]=0
	a[4]=0
	a[5]=0
	
	n=np.zeros(6)
	for i in np.arange(1,6):
		n[i]=1/i * a[i-1]
	
	
	
	print('h0 ='+str(a[0]))
	print('h1[0] ='+str(a[1]))
	print('h2[0,0] ='+str(a[2]))
	print('h3[0,0,0] ='+str(a[3]))
	print('h4[0,0,0,0] ='+str(a[4]))
	print('h5[0,0,0,0,0] ='+str(a[5]))
	
#--------------------------------------------------------------------
# L
if component=='Inductor':
	R = 0.001  # Rser
	L = 5e-9

	timestep = 1e-9

	length1 = 501 #2N+1
	times = np.arange(length1)*timestep
	freqs = np.arange(length1)/length1 * 1/timestep


	ZL = 1j*2*np.pi*freqs * L + R
	ZL[251:] = np.conjugate(ZL[250:0:-1])


	#Add real-valued part around the Nyquist frequency
	ZL[241:261]=ZL[241:261] + 1.3*np.abs(ZL[241:261])


	center = len(ZL) // 2
	bandwidth = 41
	window = gaussian(bandwidth, 12)
	window=window/np.sum(window)
	#plt.plot(window)
	#plt.show()

	#ZL[center-bandwidth//2:center+bandwidth//2] = np.convolve(ZL[center-bandwidth//2:center+bandwidth//2], window)
	ZL[center-bandwidth:center+bandwidth] = np.convolve(ZL, window, mode='same')[center-bandwidth:center+bandwidth]


	halfWindLength = 40

	ZLtime = np.real(np.fft.ifft(ZL))

	ZLtime = np.roll(ZLtime, halfWindLength)
	#print(np.fft.fft(ZLtime))
	#print(ZL)


	ZLtime = ZLtime[:2*halfWindLength] * \
		(0.54-0.46*np.cos(2*np.pi*np.arange(2*halfWindLength)/(2*halfWindLength-1)))


	#Shortening the response
	startEnergyFraction = 0.00006
	endEnergyFraction = 0.000001



	E_ges = 0
	m_start = 0
	m_end = 0
	E_min = 0
	E_max = 0
	for Z in ZLtime:
		E_ges += abs(Z) ** 2
	for k in range(len(ZLtime)):
		E_min += abs(ZLtime[k]) ** 2
		if E_min >= startEnergyFraction * E_ges:
			m_start = k
			break
	for k in range(len(ZLtime)-1, -1, -1):
		E_max += abs(ZLtime[k]) ** 2
		if E_max >= endEnergyFraction * E_ges:
			m_end = k
			break

	ZLtime_new = ZLtime[m_start:m_end+1]
	print('Original Length:' + str(len(ZLtime)))
	print('Shortened Length:' + str(len(ZLtime_new)))
	print('')
	print('Negative Length:' +str(m_start-halfWindLength))
	print('Positive Length:' +str(m_end-halfWindLength+1))

	#print(repr(ZLtime))

	print('')

	print('Code for shortened array:')
	print('h1['+str(m_start-halfWindLength)+':] = ' + 'np.' + repr(ZLtime_new[0:-m_start+halfWindLength]))
	print('h1[0:'+str(m_end-halfWindLength+1)+'] = ' + 'np.' + repr(ZLtime_new[-m_start+halfWindLength:]))




	# plt.plot(np.real(ZLtime))
	plt.plot(np.arange(m_start-halfWindLength, m_end-halfWindLength+1), np.real(ZLtime_new), color='lightcyan')
	plt.stem(np.arange(m_start-halfWindLength, m_end-halfWindLength+1), np.real(ZLtime_new), basefmt = 'black')

	plt.show()

	plt.plot(20*np.log10(np.abs(np.fft.fft(ZLtime))))
	plt.plot(np.angle(np.fft.fft(np.roll(ZLtime, halfWindLength))))
	plt.plot(np.convolve(np.angle(np.fft.fft(np.roll(ZLtime, halfWindLength))), [-0.5,0.5], mode='same'))
	plt.show()

	##Plot the shortened version
	# plt.plot(np.roll(ZLtime_new, m_start-halfWindLength))
	# plt.show()

	# plt.plot(20*np.log10(np.abs(np.fft.fft(ZLtime_new))))
	# plt.plot(np.angle(np.fft.fft(np.roll(ZLtime_new, m_start-halfWindLength))))
	# plt.show()



#Capacitor---------------------------------------------------
if component=='Capacitor':
	R=24#Rpar
	G=1/R

	C=116e-6

	timestep=0.1e-3

	length1=501
	times=np.arange(length1)*timestep
	freqs=np.arange(length1)/length1 * 1/timestep


	YC=1j*2*np.pi*freqs * C + G
	YC[251:]=np.conjugate(YC[250:0:-1])




	YC[241:261]=YC[241:261] + 1.3*np.abs(YC[241:261])


	center = len(YC) // 2
	bandwidth = 41
	window = gaussian(bandwidth, 12)
	window=window/np.sum(window)
	#plt.plot(window)
	#plt.show()

	#ZL[center-bandwidth//2:center+bandwidth//2] = np.convolve(ZL[center-bandwidth//2:center+bandwidth//2], window)
	YC[center-bandwidth:center+bandwidth] = np.convolve(YC, window, mode='same')[center-bandwidth:center+bandwidth]



	halfWindLength=40

	YCtime=np.real(np.fft.ifft(YC))
	YCtime=np.roll(YCtime, halfWindLength)
	#print(np.fft.fft(YCtime))
	#print(YC)


	YCtime=YCtime[:2*halfWindLength]*(0.54-0.46*np.cos(2*np.pi*np.arange(2*halfWindLength)/(2*halfWindLength-1)))

	#Shortening the response
	startEnergyFraction = 0.00006
	endEnergyFraction = 0.000001


	E_ges = 0
	m_start = 0
	m_end = 0
	E_min = 0
	E_max = 0
	for Y in YCtime:
		E_ges += abs(Y) ** 2
	for k in range(len(YCtime)):
		E_min += abs(YCtime[k]) ** 2
		if E_min >= startEnergyFraction * E_ges:
			m_start = k
			break
	for k in range(len(YCtime)-1, -1, -1):
		E_max += abs(YCtime[k]) ** 2
		if E_max >= endEnergyFraction * E_ges:
			m_end = k
			break

	YCtime_new = YCtime[m_start:m_end+1]
	print('Original Length:' + str(len(YCtime)))
	print('Shortened Length:' + str(len(YCtime_new)))
	print('')
	print('Negative Length:' +str(m_start-halfWindLength))
	print('Positive Length:' +str(m_end-halfWindLength+1))

	#print(repr(YCtime))

	print('')

	print('Code for shortened array:')
	print('h1['+str(m_start-halfWindLength)+':] = ' + 'np.' + repr(YCtime_new[0:-m_start+halfWindLength]))
	print('h1[0:'+str(m_end-halfWindLength+1)+'] = ' + 'np.' + repr(YCtime_new[-m_start+halfWindLength:]))



	# plt.plot(np.real(YCtime))
	plt.plot(np.arange(m_start-halfWindLength, m_end-halfWindLength+1), np.real(YCtime_new), color='lightcyan')
	plt.stem(np.arange(m_start-halfWindLength, m_end-halfWindLength+1), np.real(YCtime_new), basefmt = 'black')

	plt.show()

	plt.plot(20*np.log10(np.abs(np.fft.fft(YCtime))))
	plt.plot(np.angle(np.fft.fft(np.roll(YCtime, halfWindLength))))
	plt.plot(np.convolve(np.angle(np.fft.fft(np.roll(YCtime, halfWindLength))), [-0.5,0.5], mode='same'))
	plt.show()


	# plt.plot(20*np.log10(np.abs(np.fft.fft(YCtime_new))))
	# plt.plot(np.angle(np.fft.fft(np.roll(YCtime_new, -halfWindLength+m_start))))
	# plt.plot(np.convolve(np.angle(np.fft.fft(np.roll(YCtime_new, -halfWindLength+m_start))), [-0.5,0.5], mode='same'))
	# plt.show()


	#-----------------------


#Experimental: Instead of giving a fixed series resistance, we can introduce a frequency-dependent resistor
#that only works at low frequencies, and approaches a short circuit towards larger frequencies (or the 
#other way round with parallel admittance Y). This allows for higher Q-factor components at intermediate frequencies. 
if component=='GaussResistor':

	#Generalized mostly-causal gaussian
	Rser=1.0

	
	sigma = 5 #how far should the resistance be spread in time domain, and inversely, in frequency domain. A value of 1 approaches full frequency band;
	#a value of x will make the resistance usable up to fs/x.
	alpha = 3 #Skewness parameter - 0 is symmetric; and the higher, the more causality we get, comparatively.


	halfWindLength=20
	sample = np.arange(-halfWindLength, halfWindLength-1, 1)

	mu=-sigma * alpha/(np.sqrt(1 + alpha**2)) * np.sqrt(2/np.pi)#This is the correct µ, so that our window has "0 center of mass" -> 0 phase shift at low frequency. 


	gausswin = skewnorm.pdf(sample, a=alpha, loc=mu, scale=sigma)

	gausswin = gausswin/np.sum(gausswin) * Rser
	
	
	#Shortening the response
	startEnergyFraction = 0.00006
	endEnergyFraction = 0.000001


	E_ges = 0
	m_start = 0
	m_end = 0
	E_min = 0
	E_max = 0
	for spl in gausswin:
		E_ges += abs(spl) ** 2
	for k in range(len(gausswin)):
		E_min += abs(gausswin[k]) ** 2
		if E_min >= startEnergyFraction * E_ges:
			m_start = k
			break
	for k in range(len(gausswin)-1, -1, -1):
		E_max += abs(gausswin[k]) ** 2
		if E_max >= endEnergyFraction * E_ges:
			m_end = k
			break

	
	print('Code for shortened array:')
	print('h1['+str(m_start-halfWindLength)+':] = ' + 'np.' + repr(gausswin[m_start:halfWindLength]))
	print('h1[0:'+str(m_end-halfWindLength)+'] = ' + 'np.' + repr(gausswin[halfWindLength:m_end]))


	plt.plot(sample, gausswin)
	plt.show()
	
	plt.plot(20*np.log10(np.abs(np.fft.fft(gausswin))))
	plt.plot(np.angle(np.fft.fft(np.roll(gausswin, halfWindLength-1))))
	plt.show()
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager



Ibias=0.05

C=10e-12
Rfb=200
Z0=50
Ro=200

Vt=0.025

#Generate test input
Amplitude=0.07#V;




ssFactor2=0.5#Small Signal factor, small signal is this much lower than large signal.
ssFactor3=0.01

inputlength=100000#samples;
timestep=10e-12#s; Sample timestep of the volterra system
fstep=2*1/(inputlength*timestep)#
print('fstep:'+'{:.2e}'.format(fstep))

times=np.arange(inputlength)*timestep
input=np.zeros(inputlength)

output=np.zeros(inputlength)
output2=np.zeros(inputlength)
output3=np.zeros(inputlength)


numberFrequencies=1000
print('fmax: '+'{:.2e}'.format(fstep*numberFrequencies))
dBcHarmonic3=np.zeros(numberFrequencies)
dBcHarmonic5=np.zeros(numberFrequencies)
dBcHarmonic7=np.zeros(numberFrequencies)

for i in range(0,numberFrequencies,1):

	frequency=i*fstep
	normalizedFrequency = frequency*timestep

	#Sinewave	
	if(1):
		for n in np.arange(len(input)):
			input[n]=Amplitude*np.math.sin(2*np.pi*normalizedFrequency*n)	
		
		
	input2=input*ssFactor2
	input3=input*ssFactor3


	for n in np.arange(len(input)-1):
		output[n+1]=output[n]+timestep* 1/C*( output[n]*(-1/Ro) - (output[n]-input[n])/(Z0+Rfb) - Ibias*np.tanh(1/(2*Vt)*1/(Z0+Rfb)*(Z0*output[n]+Rfb*input[n])))

	#for n in np.arange(len(input)-1):
	#	output2[n+1]=output2[n]+timestep* 1/C*( output2[n]*(-1/Ro) - (output2[n]-input2[n])/(Z0+Rfb) - Ibias*np.tanh(1/(2*Vt)*1/(Z0+Rfb)*(Z0*output2[n]+Rfb*input2[n])))	
	#
	#for n in np.arange(len(input)-1):
	#	output3[n+1]=output3[n]+timestep* 1/C*( output3[n]*(-1/Ro) - (output3[n]-input3[n])/(Z0+Rfb) - Ibias*np.tanh(1/(2*Vt)*1/(Z0+Rfb)*(Z0*output3[n]+Rfb*input3[n])))
		
		
	outputFFT =20*np.log10(np.clip(np.abs(np.fft.fft(output[int(inputlength/2):]/int(inputlength/2))), a_min=1e-12, a_max=1e12) )
	#outputFFT2=20*np.log10(np.clip( np.abs(np.fft.fft(output[int(inputlength/2):]/int(inputlength/2)/ssFactor2)) , a_min=1e-9, a_max=1e9) )
	#outputFFT3=20*np.log10(np.abs(np.fft.fft(output3[int(inputlength/2):]/int(inputlength/2)/ssFactor3)))



	
	#print(str(outputFFT[harmonicNumber*np.rint(normalizedFrequency*inputlength/2).astype('int')])+'dBV')
	dBVCarrier=(outputFFT[1*np.rint(normalizedFrequency*inputlength/2).astype('int')])
	
	harmonicNumber=3
	dBVHarmonic3=(outputFFT[harmonicNumber*np.rint(normalizedFrequency*inputlength/2).astype('int')])
	dBcHarmonic3[i]=dBVHarmonic3-dBVCarrier
	
	harmonicNumber=5
	dBVHarmonic5=(outputFFT[harmonicNumber*np.rint(normalizedFrequency*inputlength/2).astype('int')])
	dBcHarmonic5[i]=dBVHarmonic5-dBVCarrier
	
	harmonicNumber=7
	dBVHarmonic7=(outputFFT[harmonicNumber*np.rint(normalizedFrequency*inputlength/2).astype('int')])
	dBcHarmonic7[i]=dBVHarmonic7-dBVCarrier
	
	
	
	
	#print(str(dBcHarmonic[i])+' dBc')




#Fonts used
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
#font.set_style('italic')
font.set_size('22')



#Subplots and figure size

fig, ax2 = plt.subplots(1,1, figsize=(6, 4), dpi=90)



#fig, ax2 = plt.subplots(1,1)

# #Plotting
# ax1.plot(times, input, color='lightblue', label='input')
# ax1.plot(times, output, color='lightsalmon', label='Large signal')
# ax1.plot(times, output2/ssFactor2, color='salmon', label='Medium signal')
# ax1.plot(times, output3/ssFactor3, color='tomato', label='Small signal')
# ax1.set_xlabel('Time [s]', fontproperties=font)
# ax1.set_ylabel('Voltage [V]', fontproperties=font)
# ax1.legend()
# ax1.grid()


# ax2.plot(outputFFT, color='lightsalmon', label='Large signal')
# ax2.plot(outputFFT2, color='salmon', label='Medium signal')
# ax2.plot(outputFFT3, color='tomato', label='Small signal')
# ax2.set_xlabel('Frequency', fontproperties=font)
# ax2.set_ylabel('log. Voltage [VdB]', fontproperties=font)
# ax2.legend()
# ax2.grid()


ax2.semilogx(np.arange(1,numberFrequencies,1)*fstep, dBcHarmonic3[1:], color='#600297', label='HD3', linewidth = 2)
ax2.semilogx(np.arange(1,numberFrequencies,1)*fstep, dBcHarmonic5[1:], color='#9702ed', label='HD5', linewidth = 2)
ax2.semilogx(np.arange(1,numberFrequencies,1)*fstep, dBcHarmonic7[1:], color='#cc77fd', label='HD7', linewidth = 2)
ax2.set_ylim(-150, -20)#dB
ax2.set_xlabel('Frequency [Hz]', fontproperties=font)
ax2.set_ylabel('Distortion [dBc]', fontproperties=font)
ax2.legend()
ax2.grid()




plt.savefig('Plots/FigureOutput/Theory_DifferentialAmpFBHD3.pdf', dpi=90, transparent=False, bbox_inches='tight')
plt.show()	
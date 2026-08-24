import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from scipy.special import comb
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from sklearn.decomposition import PCA
from Volterra_In_Out_6 import VolterraResponse



# General Settings
top = mpl.colormaps['Blues_r'].resampled(128)
bottom = mpl.colormaps['Oranges'].resampled(128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
					   bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')
colormap23 = newcmp

color7b = '#f5a300'  # medium orange
color7c = '#d28700'  # dark orange
color1c = '#004e8a'  # dark blue
color10c = '#951169'  # dark violet
color11a = '#804597'
color10a = '#c9308e'
color1a = '#5d85c3'

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
# font.set_style('italic')
font.set_size('12')





def plotOverview(systemOrig, Amplitude, timestep):
	
	offset1=-systemOrig.nLen1
	offset2=-systemOrig.nLen2
	offset3=-systemOrig.nLen3
	system=systemOrig#wraparoundToOrdered(systemOrig)
	
	
	
	# Log the sizes of arrays
	print(f"h1 size: {system.h1.size}")
	print(f"h2 size: {system.h2.size}")
	print(f"h3 size: {system.h3.size}")
	print(f"h4 size: {system.h4.size}")
	print(f"h5 size: {system.h5.size}")

	fig = plt.figure(figsize=(12, 8), dpi=90, constrained_layout=False)
	fig.suptitle('System Information', fontproperties=font)

	# Plot Nonlinear Step Response
	axh1 = plt.subplot(2, 2, 1)
	
	inStep=np.zeros(systemOrig.len1+1*systemOrig.nLen1)
	
	inStep[systemOrig.nLen1:] = Amplitude * np.ones(systemOrig.len1)	
	#inStep[systemOrig.nLen1]=Amplitude
	
	#inStep[systemOrig.nLen1+1]=Amplitude
	#inStep[systemOrig.nLen1+2]=Amplitude
	#inStep[systemOrig.nLen1+3]=Amplitude
	#inStep[systemOrig.nLen1+4]=Amplitude
	
	inTimes = timestep * (np.arange(systemOrig.len1+1*systemOrig.nLen1)+offset1)


	#print(systemOrig.H(inStep, neg_len=np.array([systemOrig.nLen1, systemOrig.nLen2, systemOrig.nLen3, systemOrig.nLen4, systemOrig.nLen5]))[:-systemOrig.nLen1])

	#Samples or Times as xaxis
	#xaxis=inTimes
	#axh1.set_xlabel('Time $[s]$', fontproperties=font)
	
	xaxis=np.arange(-systemOrig.nLen1, systemOrig.len1)
	axh1.set_xlabel('Sample #', fontproperties=font)
	
	#systemOrig.use_opencl=False
	axh1.plot(xaxis, systemOrig.H(inStep, neg_len=np.array([systemOrig.nLen1, systemOrig.nLen2, systemOrig.nLen3, systemOrig.nLen4, systemOrig.nLen5])), color='black', marker='D', markersize=8, linewidth=2.5, label='H')
	#axh1.plot(xaxis, systemOrig.H(inStep), color='black', marker='D', markersize=8, linewidth=2.5, label='H')
	axh1.plot(xaxis, systemOrig.H0(inStep), color='firebrick', marker='o', linewidth=1.5, label='H0')
	axh1.plot(xaxis, systemOrig.H1(inStep, neg_len=systemOrig.nLen1), color='royalblue', marker='o', linewidth=1.5, label='H1')
	axh1.fill_between(xaxis, systemOrig.H1(inStep, neg_len=systemOrig.nLen1), color='royalblue', alpha=0.15, interpolate=True, linewidth=1.5)
	axh1.plot(xaxis, systemOrig.H2(inStep, neg_len=systemOrig.nLen2), color='salmon', marker='o', linewidth=1.5, label='H2')
	axh1.fill_between(xaxis, systemOrig.H2(inStep, neg_len=systemOrig.nLen1), color='salmon', alpha=0.15, interpolate=True, linewidth=1.5)
	axh1.plot(xaxis, systemOrig.H3(inStep, neg_len=systemOrig.nLen3), color='cornflowerblue', marker='o', linewidth=1.5, label='H3')
	axh1.fill_between(xaxis, systemOrig.H3(inStep, neg_len=systemOrig.nLen1), color='cornflowerblue', alpha=0.15, interpolate=True, linewidth=1.5)
	axh1.plot(xaxis, systemOrig.H4(inStep, neg_len=systemOrig.nLen4), color='lightsalmon', marker='o', linewidth=1.5, label='H4')
	axh1.fill_between(xaxis, systemOrig.H4(inStep, neg_len=systemOrig.nLen1), color='lightsalmon', alpha=0.15, interpolate=True, linewidth=1.5)
	axh1.plot(xaxis, systemOrig.H5(inStep, neg_len=systemOrig.nLen5), color='lightskyblue', marker='o', linewidth=1.5, label='H5')
	axh1.fill_between(xaxis, systemOrig.H5(inStep, neg_len=systemOrig.nLen1), color='lightskyblue', alpha=0.15, interpolate=True, linewidth=1.5)
	

	print('Max SR: ' + str(np.max(system.H(inStep)
		  [1:] - system.H(inStep)[:-1]) / timestep / 1e9))

	axh1.set_title('Nonlinear Step Response', fontproperties=font)
	axh1.legend()
	axh1.grid(visible=True)
	
	# #Limit Axes
	#axh1.set_ylim((-0.05, 0.35))
	#axh1.set_xlim((-2, 20))
	
	
	# Plot Frequency Response
	axh2 = plt.subplot(2, 2, 2)
	axh2_ang = axh2.twinx()
	referenceDirac=np.zeros(system.len1+system.nLen1)
	referenceDirac[-offset1]=1
	
	# print(offset1)
	# plt.plot(system.h1)
	# plt.show()
	
	axh2_ang.plot(np.unwrap(np.angle(np.fft.fft(system.h1)/np.fft.fft(referenceDirac))),
				  label='ang(h1)', linestyle='dotted', color='firebrick')
	axh2.plot(20 * np.log10(np.abs(np.fft.fft(system.h1))),
			  label='mag(h1)', color='royalblue')
	axh2.set_xlabel('Frequency bin', fontproperties=font)
	axh2.set_ylabel('Magnitude (dB)', fontproperties=font)
	axh2_ang.set_ylabel('Angle (rad)', fontproperties=font)
	axh2.set_title('Frequency Response', fontproperties=font)
	axh2.grid(visible=True)
	axh2.legend(loc='lower left')
	axh2_ang.legend(loc='lower right')

	# Plot Kernels
	# 1st order kernel
	axh3 = plt.subplot(2, 3, 4)
	axh3.plot(np.arange(system.len1+system.nLen1)+offset1, system.h1, color='royalblue', marker='o')
	axh3.set_xlabel('Index', fontproperties=font)
	axh3.set_ylabel('Value', fontproperties=font)
	axh3.set_title('1st Order Kernel', fontproperties=font)
	axh3.grid(visible=True)

	# 2nd order kernel
	axh4 = plt.subplot(2, 3, 5)
	start, stop = 0, system.len2+system.nLen2
	h2slice = system.h2[start:stop, start:stop]
	x, y = np.meshgrid(np.arange(start, stop)+offset2, np.arange(start, stop)+offset2)
	factor2 = 400 / np.max(np.abs(h2slice) ** 3)
	im2 = axh4.scatter(x.flatten(), y.flatten(), c=h2slice.flatten(), s=factor2 * abs(h2slice.flatten()) ** 3, cmap='seismic',
					   vmin=-1.2 * np.max(np.abs(h2slice)), vmax=1.2 * np.max(np.abs(h2slice)))
	markers = np.zeros(shape=(stop - start, stop - start))
	for i in range(stop - start):
		markers[i, i] = 1
	axh4.scatter(x.flatten(), y.flatten(), c='black',
				 s=10 * markers.flatten(), marker='x')
	axh4.set_xlabel('x Index', fontproperties=font)
	axh4.set_ylabel('y Index', fontproperties=font)
	axh4.set_title('2nd Order Kernel', fontproperties=font)
	axh4.grid(visible=True)
	clb = fig.colorbar(im2, ax=axh4, fraction=0.02, pad=0.1, location='left')
	clb.ax.tick_params(labelsize=8)
	clb.ax.set_title(r'$h_2(\tau_1, \tau_2)$', fontsize=12, rotation=0)

	# 3rd order kernel
	axh5 = plt.subplot(2, 3, 6, projection='3d')
	start, stop = 0, system.len3+system.nLen3
	h3slice = system.h3[start:stop, start:stop, start:stop]
	x, y, z = np.meshgrid(np.arange(start, stop)+offset3, np.arange(
		start, stop)+offset3, np.arange(start, stop)+offset3)
	factor3 = 200 / np.max(np.abs(h3slice) ** 3)
	im3 = axh5.scatter(x.flatten(), y.flatten(), z.flatten(), c=h3slice.flatten(), s=factor3 * abs(h3slice.flatten()) ** 3, cmap='seismic',
					   vmin=-1.2 * np.max(np.abs(h3slice)), vmax=1.2 * np.max(np.abs(h3slice)), alpha=0.9)
	markers = np.zeros(shape=(stop - start, stop - start, stop - start))
	for i in np.arange(stop - start):
		markers[i, i, i] = 1
	axh5.scatter(x.flatten(), y.flatten(), z.flatten(),
				 c='black', s=10 * markers.flatten(), marker='x')
	axh5.set_xlabel('x Index', fontproperties=font)
	axh5.set_ylabel('y Index', fontproperties=font)
	axh5.set_zlabel('z Index', fontproperties=font)
	axh5.set_title('3rd Order Kernel', fontproperties=font)
	axh5.grid(visible=True)
	clb = fig.colorbar(im3, ax=axh5, fraction=0.02, pad=0.1, location='left')
	clb.ax.tick_params(labelsize=8)
	clb.ax.set_title(r'$h_3(\tau_1, \tau_2, \tau_3)$', fontsize=12, rotation=0)

	output_dir = 'Plots/FigureOutput'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	plt.savefig(f'{output_dir}/Results_Kernelsh2.png',
				dpi=90, transparent=False)
	plt.savefig(f'{output_dir}/Results_Kernelsh3.png',
				dpi=90, transparent=False)

	plt.show()


def plot1(system, start, stop):
	offset1=-system.nLen1
	#system=wraparoundToOrdered(system)
	h1slice = system.h1[start:stop]
	fig = plt.figure(figsize=(6, 4.5), dpi=90)
	axh = fig.add_axes(rect=(.1, .1, .8, .8))
	# , zorder=1)#,  marker='o', markeredgecolor='darkorange', markerfacecolor='darkorange')
	axh.stem(np.arange(start, stop)+offset1, h1slice,
			 linefmt='black', markerfmt=' ', basefmt='black')
	im1 = axh.scatter(np.arange(start, stop)+offset1, h1slice, c=h1slice, s=50, edgecolors='black',
					  cmap=colormap23, vmin=1.2*-np.max(np.abs(h1slice)), vmax=1.2*np.max(np.abs(h1slice)), zorder=2)
	axh.set_xlabel(r'$\tau_1$', fontproperties=font)
	axh.set_ylim((1.2*-np.max(np.abs(h1slice)), 1.2*np.max(np.abs(h1slice))))
	axh.set_title('', fontproperties=font)
	axh.grid(visible=False)
	clb = fig.colorbar(im1, ax=axh, fraction=0.031,
					   pad=0.1, location='left', aspect=30)
	clb.ax.tick_params(labelsize=8)
	clb.ax.set_title(r'$h_1(\tau_1)$', fontsize=12, rotation=0)
	clb.set_ticks([])

	output_dir = 'Plots/FigureOutput'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	plt.savefig(f'{output_dir}/Results_Kernelsh1.png',
				dpi=90, transparent=False)


def plot2(system, start, stop):
	offset2=-system.nLen2
	#system=wraparoundToOrdered(system)
	h2slice = system.h2[start:stop, start:stop]
	fig = plt.figure(figsize=(6, 4.5), dpi=90)
	axh = fig.add_axes(rect=(.1, .1, .8, .8))
	x, y = np.meshgrid(np.arange(start, stop)+offset2, np.arange(start, stop)+offset2)

	factor2 = 400/np.max(np.abs(h2slice)**3)

	# Flattened the arrays
	im2 = axh.scatter(x.flatten(), y.flatten(), c=h2slice.flatten(), s=factor2*abs(h2slice.flatten())**3, cmap=colormap23,
					  vmin=-1.2*np.max(np.abs(h2slice)), vmax=1.2*np.max(np.abs(h2slice)))
	markers = np.zeros(shape=(stop-start, stop-start))
	for i in range(stop-start):
		markers[i, i] = 1
	axh.scatter(x.flatten(), y.flatten(), c='black',
				s=10*markers.flatten(), marker='x')
	axh.set_xlabel(r'$\tau_1$', fontproperties=font)
	axh.yaxis.tick_right()
	axh.yaxis.set_label_position("right")
	axh.set_ylabel(r'$\tau_2$', fontproperties=font)
	axh.set_title('', fontproperties=font)
	axh.grid(visible=False)
	clb = fig.colorbar(im2, ax=axh, fraction=0.02, pad=0.1, location='left')
	clb.ax.tick_params(labelsize=8)
	clb.ax.set_title(r'$h_2(\tau_1, \tau_2)$', fontsize=12, rotation=0)

	output_dir = 'Plots/FigureOutput'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	plt.savefig(f'{output_dir}/Results_Kernelsh2.png',
				dpi=90, transparent=False)


def plot3(system, start, stop):
	offset3=-system.nLen3
	#system=wraparoundToOrdered(system)
	
	h3slice = system.h3[start:stop, start:stop, start:stop]
	fig = plt.figure(figsize=(6, 4.5), dpi=90, constrained_layout=False)
	axh = fig.add_axes(rect=(.1, .1, 0.8, 0.8), projection='3d')
	x, y, z = np.meshgrid(np.arange(start, stop)+offset3, np.arange(
		start, stop)+offset3, np.arange(start, stop)+offset3)

	factor3 = 200/np.max(np.abs(h3slice)**3)

	# Flattened the arrays
	im3 = axh.scatter(x.flatten(), y.flatten(), z.flatten(), c=h3slice.flatten(), s=factor3*abs(h3slice.flatten())**3, cmap=colormap23,
					  vmin=-1.2*np.max(np.abs(h3slice)), vmax=1.2*np.max(np.abs(h3slice)), alpha=0.9)
	axh.set_box_aspect(None, zoom=1.2)
	markers = np.zeros(shape=(stop-start, stop-start, stop-start))
	for i in np.arange(stop-start):
		markers[i, i, i] = 1
	axh.scatter(x.flatten(), y.flatten(), z.flatten(),
				c='black', s=10*markers.flatten(), marker='x')
	axh.set_xlabel(r'$\tau_1$', fontproperties=font)
	axh.set_ylabel(r'$\tau_2$', fontproperties=font)
	axh.set_zlabel(r'$\tau_3$', fontproperties=font)
	axh.set_title('', fontproperties=font)
	axh.grid(visible=False)
	clb = fig.colorbar(im3, ax=axh, fraction=0.02, pad=0.1, location='left')
	clb.ax.tick_params(labelsize=8)
	clb.ax.set_title(r'$h_3(\tau_1, \tau_2, \tau_3)$', fontsize=12, rotation=0)

	output_dir = 'Plots/FigureOutput'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	plt.savefig(f'{output_dir}/Results_Kernelsh3.png',
				dpi=90, transparent=False)


def plot4(fig, ax, system, start, stop, tau4, cax, vmin, vmax):
	try:
		if tau4 < 0 or tau4 >= system.h4.shape[3]:
			raise IndexError(
				f"tau4 index {tau4} is out of bounds for axis 3 with size {system.h4.shape[3]}")

		h4slice = system.h4[start:stop, start:stop, start:stop, tau4]
		ax.clear()
		ax.set_xlabel(r'$\tau_1$', fontproperties=font)
		ax.set_ylabel(r'$\tau_2$', fontproperties=font)
		ax.set_zlabel(r'$\tau_3$', fontproperties=font)
		ax.set_title(r'$h_4(\tau_1, \tau_2, \tau_3, \tau_4)$',
					 fontproperties=font)

		x, y, z = np.meshgrid(np.arange(start, stop), np.arange(
			start, stop), np.arange(start, stop))
		factor4 = 200 / np.max(np.abs(h4slice) **
							   3) if np.max(np.abs(h4slice)**3) != 0 else 1
							   
		
		im = ax.scatter(x, y, z, c=h4slice, s=factor4 * np.abs(h4slice)**3, cmap=colormap23,
						vmin=vmin, vmax=vmax)
		if cax:
			cax.clear()
			fig.colorbar(im, cax=cax)
	except IndexError as e:
		print(f"Error in plot4: {e}")
	except Exception as e:
		print(f"Unexpected error in plot4: {e}")


def plot5(fig, ax, system, start, stop, tau4, tau5, cax, vmin, vmax):
	try:
		if tau4 < 0 or tau4 >= system.h5.shape[3]:
			raise IndexError(
				f"tau4 index {tau4} is out of bounds for axis 3 with size {system.h5.shape[3]}")
		if tau5 < 0 or tau5 >= system.h5.shape[4]:
			raise IndexError(
				f"tau5 index {tau5} is out of bounds for axis 4 with size {system.h5.shape[4]}")

		h5slice = system.h5[start:stop, start:stop, start:stop, tau4, tau5]
		ax.clear()
		ax.set_xlabel(r'$\tau_1$', fontproperties=font)
		ax.set_ylabel(r'$\tau_2$', fontproperties=font)
		ax.set_zlabel(r'$\tau_3$', fontproperties=font)
		ax.set_title(
			r'$h_5(\tau_1, \tau_2, \tau_3, \tau_4, \tau_5)$', fontproperties=font)

		x, y, z = np.meshgrid(np.arange(start, stop), np.arange(
			start, stop), np.arange(start, stop))
		factor5 = 200 / np.max(np.abs(h5slice) **
							   3) if np.max(np.abs(h5slice)**3) != 0 else 1
		im = ax.scatter(x, y, z, c=h5slice, s=factor5 * np.abs(h5slice)**3, cmap=colormap23,
						vmin=vmin, vmax=vmax)
		if cax:
			cax.clear()
			fig.colorbar(im, cax=cax)
	except IndexError as e:
		print(f"Error in plot5: {e}")
	except Exception as e:
		print(f"Unexpected error in plot5: {e}")


def interactive_plot_h4(system):
	fig_h4 = plt.figure(figsize=(10, 10), dpi=100)
	ax1 = fig_h4.add_subplot(111, projection='3d')
	plt.subplots_adjust(left=0.1, bottom=0.25, right=0.85,
						top=0.9, wspace=0.4, hspace=0.4)

	# Create colorbar axes
	cax1 = fig_h4.add_axes([0.87, 0.25, 0.02, 0.6])

	# Find global min and max for h4
	global_min_h4 = np.min(system.h4)
	global_max_h4 = np.max(system.h4)
	# print the index pf the global min and max
	print(np.where(system.h4 == global_min_h4), global_min_h4)
	print(np.where(system.h4 == global_max_h4), global_max_h4)

	# Initial plot
	tau4 = 0
	plot4(fig_h4, ax1, system, 0, system.len4, tau4, cax1, global_min_h4, global_max_h4)

	# Slider for tau4
	ax_tau4_slider_h4 = plt.axes(
		[0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
	tau4_slider_h4 = Slider(ax_tau4_slider_h4, r'$\tau_4$',
							0, system.h4.shape[3]-1, valinit=0, valstep=1)

	def update_h4(val):
		tau4 = int(tau4_slider_h4.val)
		plot4(fig_h4, ax1, system, 0, system.len4, tau4,
			  cax1, global_min_h4, global_max_h4)
		fig_h4.canvas.draw_idle()

	tau4_slider_h4.on_changed(update_h4)
	plt.show()


def interactive_plot_h5(system):
	fig_h5 = plt.figure(figsize=(10, 10), dpi=100)
	ax2 = fig_h5.add_subplot(111, projection='3d')
	plt.subplots_adjust(left=0.1, bottom=0.25, right=0.85,
						top=0.9, wspace=0.4, hspace=0.4)

	# Create colorbar axes
	cax2 = fig_h5.add_axes([0.87, 0.25, 0.02, 0.6])

	# Find global min and max for h5
	global_min_h5 = np.min(system.h5)
	global_max_h5 = np.max(system.h5)

	# Initial plot
	tau4 = 0
	tau5 = 0
	plot5(fig_h5, ax2, system, 0, system.len5, tau4, tau5,
		  cax2, global_min_h5, global_max_h5)

	# Sliders for tau4 and tau5
	ax_tau4_slider_h5 = plt.axes(
		[0.1, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
	tau4_slider_h5 = Slider(ax_tau4_slider_h5, r'$\tau_4$',
							0, system.h5.shape[3]-1, valinit=0, valstep=1)

	ax_tau5_slider_h5 = plt.axes(
		[0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
	tau5_slider_h5 = Slider(ax_tau5_slider_h5, r'$\tau_5$',
							0, system.h5.shape[4]-1, valinit=0, valstep=1)

	def update_h5(val):
		tau4 = int(tau4_slider_h5.val)
		tau5 = int(tau5_slider_h5.val)
		plot5(fig_h5, ax2, system, 0, system.len5, tau4, tau5,
			  cax2, global_min_h5, global_max_h5)
		fig_h5.canvas.draw_idle()

	tau4_slider_h5.on_changed(update_h5)
	tau5_slider_h5.on_changed(update_h5)
	plt.show()


# Principlae Component Analysis
if (0):
	def plot4_pca(system, start, stop):
		h4slice = system.h4[start:stop, start:stop, start:stop, :]
		h4shape = h4slice.shape
		h4flat = h4slice.reshape(-1, h4shape[-1])

		pca = PCA(n_components=3)
		h4_pca = pca.fit_transform(h4flat)

		fig = plt.figure(figsize=(10, 10), dpi=100)
		ax = fig.add_subplot(111, projection='3d')

		sc = ax.scatter(h4_pca[:, 0], h4_pca[:, 1],
						h4_pca[:, 2], c=h4_pca[:, 2], cmap=colormap23)
		ax.set_xlabel('PC 1', fontproperties=font)
		ax.set_ylabel('PC 2', fontproperties=font)
		ax.set_zlabel('PC 3', fontproperties=font)
		ax.set_title('PCA of 4th Order Kernel', fontproperties=font)

		# Add colorbar
		cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
		cbar.ax.tick_params(labelsize=8)
		cbar.ax.set_title(r'$PC 3$', fontsize=12, rotation=0)

		plt.show()

	def plot5_pca(system, start, stop):
		h5slice = system.h5[start:stop, start:stop, start:stop, :, :]
		h5shape = h5slice.shape
		h5flat = h5slice.reshape(-1, h5shape[-1])

		pca = PCA(n_components=3)
		h5_pca = pca.fit_transform(h5flat)

		fig = plt.figure(figsize=(10, 10), dpi=100)
		ax = fig.add_subplot(111, projection='3d')

		sc = ax.scatter(h5_pca[:, 0], h5_pca[:, 1],
						h5_pca[:, 2], c=h5_pca[:, 2], cmap=colormap23)
		ax.set_xlabel('PC 1', fontproperties=font)
		ax.set_ylabel('PC 2', fontproperties=font)
		ax.set_zlabel('PC 3', fontproperties=font)
		ax.set_title('PCA of 5th Order Kernel', fontproperties=font)

		# Add colorbar
		cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
		cbar.ax.tick_params(labelsize=8)
		cbar.ax.set_title(r'$PC 3$', fontsize=12, rotation=0)

		plt.show()

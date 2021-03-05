#Necessary Python packages and dependencies
import numpy as np
from numpy import load 
import matplotlib.pyplot as plt

def myspecto_new(data,window,nfft,shift,m):
	print('In function myspecto_new...')
	N = np.int(np.floor( (data.shape[1]- window- 1) / shift))
	out1 = np.zeros((m,N), dtype=complex)

	for i in range(N):
		#print(i+1)
		tmp1 = data[0,(i)*shift+1 : (i)*shift+window+1];
		tmp2 = np.transpose(tmp1).reshape(128,1)
		tmp3 = np.hamming(window).reshape(128,1)
		tmp4 = (tmp2 * tmp3).reshape(128,1)
		tmp = np.fft.fft(tmp4, n=nfft, axis=0)
		tmp = tmp.reshape(4096)
		out1[:,i] = tmp;
	
	return out1
	
def main():
	#Open Data save in file
	DATAPATH = '/test_collect_data'
	ASLDATA = '/asl_hello_will'
	FILENAME = '..' + DATAPATH + ASLDATA + '.npz'
	print('Loading data from: ' + FILENAME)

	#Load in the necessary parts of the radar data
	data = load(FILENAME)
	sensor_data = data['data']
	sample_times = data['sample_times']
	sensor_config_dump = data['sensor_config_dump']
	processing_config_dump	= data['processing_config_dump']
	data_info = data['data_info']

	n = sensor_data.shape[2]
	m = sensor_data.shape[0]
	print('The data has FT/ST dimensions: '+ str(m) + 'x' + str(n))

	#Code to generate the data_cube
	stack_cube = np.reshape(sensor_data, (m,n))
	range_profile = np.fft.fft(stack_cube)
	nfft = 2**12
	window=128
	noverlap = 100
	shift = window - noverlap

	#Here is where the errors start 
	tmp = range_profile[60:100,:]
	window_data = tmp.sum(axis=0)
	spec_input = np.transpose(window_data).reshape(1,1096)
	#print(spec_input.shape)
	sx = myspecto_new(spec_input, window, nfft, shift, m)
	#print(sx.shape)
	sx2 = np.abs(np.flipud(np.fft.fftshift(sx, axes=(0,))))
	x = np.double(np.divide(sx2,np.max(sx2)))
	y = (np.abs(20 * np.log10(x)))
	print('Generating Spectogram...')


	fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,10))

	ax1.imshow(np.abs(np.transpose(stack_cube)), cmap='jet', extent=[0,150,0,1000], aspect='auto')
	ax1.set_title('Radar Cube Plotted')
	ax2.imshow(y, cmap='jet', extent=[0,150,0,1000], aspect='auto')
	ax2.set_title('Auto-scaled Aspect') 
	plt.tight_layout()
	plt.savefig("tmp.png")
	plt.show()

if __name__ == "__main__":
    
	main()

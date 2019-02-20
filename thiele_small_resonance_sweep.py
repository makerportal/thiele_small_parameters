import pyaudio
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def fft_calc(data_vec):
    fft_data_raw = (np.fft.fft(data_vec))
    fft_data = (fft_data_raw[0:int(np.floor(len(data_vec)/2))])/len(data_vec)
    fft_data[1:] = 2*fft_data[1:]
    return fft_data

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 2 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
record_secs = 10 # seconds to record
dev_index = 2 # device index found by p.get_device_info_by_index(ii)
chunk = 44100*record_secs # 2^12 samples for buffer
R_1 = 9.8 # measured resistance from voltage divider resistor
R_dc = 3.3 # measured dc resistance of loudspeaker

freq_sweep_range = (40.0,200.0) # range of frequencies in sweep (plus or minus a few to avoid noise at the ends)

audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)

stream.stop_stream() # pause stream so that user can control the recording
input("Click to Record")

data,chan0_raw,chan1_raw = [],[],[]

# loop through stream and append audio chunks to frame array
stream.start_stream() # start recording
for ii in range(0,int((samp_rate/chunk)*record_secs)):
    data.append(np.fromstring(stream.read(chunk),dtype=np.int16))

# stop, close stream, and terminate pyaudio instantiation
stream.stop_stream()
stream.close() 
audio.terminate()

print("finished recording\n------------------")

# loop through recorded data to extract each channel
for qq in range(0,(np.shape(data))[0]):
    curr_dat = data[qq]
    chan0_raw.append(curr_dat[::2])
    chan1_raw.append((curr_dat[1:])[::2])

# conversion from bits
chan0_raw = np.divide(chan0_raw,(2.0**15)-1.0)
chan1_raw = np.divide(chan1_raw,(2.0**15)-1.0)

# Calculating FFTs and phases
spec_array_0_noise,spec_array_1_noise,phase_array,Z_array = [],[],[],[]
for mm in range(0,(np.shape(chan0_raw))[0]):
    Z_0 = ((fft_calc(chan0_raw[mm])))
    Z_1 = ((fft_calc(chan1_raw[mm])))
    
    phase_array.append(np.subtract(np.angle(Z_0,deg=True),
                                   np.angle(Z_1,deg=True)))

    spec_array_0_noise.append(((np.abs(Z_0[1:]))))
    spec_array_1_noise.append(((np.abs(Z_1[1:]))))
    
# frequency values for FFT
f_vec = samp_rate*np.arange(chunk/2)/chunk # frequency vector
plot_freq = f_vec[1:] # avoiding f = 0 for logarithmic plotting

# calculating Z
Z_mean = np.divide(R_1*np.nanmean(spec_array_0_noise,0),
           np.subtract(np.nanmean(spec_array_1_noise,0),np.nanmean(spec_array_0_noise,0)))

# setting minimum frequency locations based on frequency sweep
f_min_loc = np.argmin(np.abs(plot_freq-freq_sweep_range[0]))
f_max_loc = np.argmin(np.abs(plot_freq-freq_sweep_range[1]))
max_f_loc = np.argmax(Z_mean[f_min_loc:f_max_loc])+f_min_loc
f_max = plot_freq[max_f_loc]

# print out impedance found from phase zero-crossing
print('Resonance at Z-based Maximum:')
print('f = {0:2.1f}, Z = {1:2.1f}'.format(f_max,np.max(Z_mean[f_min_loc:f_max_loc])))
print('------------------')

# smoothing out the phase data by averaging large spikes

smooth_width = 10 # width of smoothing window for phase
phase_trimmed = (phase_array[0])[f_min_loc:f_max_loc]
phase_diff = np.append(0,np.diff(phase_trimmed))
for yy in range(smooth_width-1,len(phase_diff)-smooth_width):
    for mm in range(0,smooth_width):
        if np.abs(phase_diff[yy]) > 100.0:
            phase_trimmed[yy] = (phase_trimmed[yy-mm]+phase_trimmed[yy+mm])/2.0
            phase_diff[yy] = (phase_diff[yy-mm]+phase_diff[yy+mm])/2.0
        if np.abs(phase_diff[yy]) > 100.0:
            continue
        else:
            break
    if np.abs(phase_diff[yy]) > 100.0:
        phase_trimmed[yy] = np.nan

##### plotting algorithms for impedance and phase ####
        
fig,ax = plt.subplots()
fig.set_size_inches(12,8)

# Logarithm plots in x-axis
p1, = ax.semilogx(plot_freq[f_min_loc:f_max_loc],Z_mean[f_min_loc:f_max_loc],label='$Z$',color='#7CAE00')
ax2 = ax.twinx() # mirror axis for phase
p2, = ax2.semilogx(plot_freq[f_min_loc:f_max_loc],phase_trimmed,label='$\phi$',color='#F8766D')

# plot formatting
subplot_vec = [p1,p2]
ax2.legend(subplot_vec,[l.get_label() for l in subplot_vec],fontsize=20)
ax.yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p2.get_color())
ax.set_ylabel('Impedance [$\Omega$]',fontsize=16)
ax2.set_ylabel('Phase [Degrees]',fontsize=16)
ax2.grid(False)
ax.spines["right"].set_edgecolor(p1.get_color())
ax2.spines["right"].set_edgecolor(p2.get_color())
ax.tick_params(axis='y', colors=p1.get_color())
ax2.tick_params(axis='y', colors=p2.get_color())
ax.set_xlabel('Frequency [Hz]',fontsize=16)

peak_width = 70.0 # approx width of peak in Hz
ax.set_xlim([f_max-(peak_width/2.0),f_max+(peak_width/2.0)])
ax.set_ylim([np.min(Z_mean[f_min_loc:f_max_loc]),np.max(Z_mean[f_min_loc:f_max_loc])+0.5])
ax2.set_ylim([-45,45])
ax.set_xticks([])
ax.set_xticklabels([])
ax2.set_xticks(np.arange(f_max-(peak_width/2.0),f_max+(peak_width/2.0),10))

# locating phase and Z maximums to annotate the figure
Z_max_text = ' = {0:2.1f} $\Omega$'.format(np.max(Z_mean[f_min_loc:f_max_loc]))
f_max_text = ' = {0:2.1f} Hz'.format(plot_freq[np.argmax(Z_mean[f_min_loc:f_max_loc])+f_min_loc])
ax.annotate('$f_{max}$'+f_max_text+', $Z_{max}$'+Z_max_text,xy=(plot_freq[np.argmax(Z_mean[f_min_loc:f_max_loc])+f_min_loc],
                                                                np.max(Z_mean[f_min_loc:f_max_loc])),\
            xycoords='data',xytext=(-300,-50),size=14,textcoords='offset points',
                   arrowprops=dict(arrowstyle='simple',
                                   fc='0.6',ec='none'))
# from phase
phase_f_min = np.argmin(np.abs(np.subtract(f_max-(peak_width/2.0),plot_freq)))
phase_f_max = np.argmin(np.abs(np.subtract(f_max+(peak_width/2.0),plot_freq)))
phase_min_loc = np.argmin(np.abs(phase_array[0][phase_f_min:phase_f_max]))+phase_f_min

Z_max_text_phase = ' = {0:2.1f} $\Omega$'.format(Z_mean[phase_min_loc])
f_max_text_phase = ' = {0:2.1f} Hz'.format(plot_freq[phase_min_loc])
ax2.annotate('$\phi_{min}$'+' ={0:2.1f}$^\circ$'.format(np.abs(phase_array[0][phase_min_loc]))+', $f_{max}$ = '+f_max_text_phase+\
             '\n$Z_{max}$'+Z_max_text_phase,
             xy=(plot_freq[phase_min_loc],phase_array[0][phase_min_loc]),\
            xycoords='data',xytext=(-120,-150),size=14,textcoords='offset points',arrowprops=dict(arrowstyle='simple',
                                   fc='0.6',ec='none'))
# print out impedance found from phase zero-crossing
print('Resonance at Phase Zero-Crossing:')
print('f = {0:2.1f}, Z = {1:2.1f}'.format(plot_freq[phase_min_loc],Z_mean[phase_min_loc]))

# uncomment to save plot
##plt.savefig('Z_sweep_with_phase.png',dpi=300,facecolor=[252/255,252/255,252/255])

plt.show()

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
dev_index = 2 # device index found by p.get_device_info_by_index(ii)
chunk = 44100 # 2^12 samples for buffer
R_1 = 9.8 # measured resistance from voltage divider resistor

audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
stream.stop_stream()
Z_mean_array,max_freq_array,phase_array = [],[],[]
prev_freq = 0.0

try:
    while True:
        input("Click to Record Frequency")
        stream.start_stream()
        data = []
        chan0_raw,chan1_raw = [],[]
        # loop through stream and append audio chunks to frame array
        data.append(np.fromstring(stream.read(chunk),dtype=np.int16))
        stream.stop_stream()
        # separate data into both channels
        chan0_raw.append(data[0][::2])
        chan1_raw.append((data[0][1:])[::2])

        record_secs = (np.shape(chan0_raw))[0]*(chunk/samp_rate)

        # conversion from bits
        chan0_raw = np.divide(chan0_raw,(2.0**15)-1.0)
        chan1_raw = np.divide(chan1_raw,(2.0**15)-1.0)

        # plotting prep

        f_vec = samp_rate*np.arange(chunk/2)/chunk
        fft_chan0 = fft_calc(chan0_raw[0])
        fft_chan1 = fft_calc(chan1_raw[0])
        
        Z_mean = np.divide(R_1*np.sqrt(np.mean(np.power(chan0_raw[0],2.0))),
                   np.subtract(np.sqrt(np.mean(np.power(chan1_raw[0],2.0))),np.sqrt(np.mean(np.power(chan0_raw[0],2.0)))))
        
        max_loc = np.argmax(np.abs(fft_chan0))
        max_freq = f_vec[max_loc]

        phase = (np.angle(fft_chan0[max_loc],deg=True))-(np.angle(fft_chan1[max_loc],deg=True))
                
        end_pt = int(np.ceil(samp_rate/max_freq))
        print('f = {1:2.1f}, phase = {2:2.3f}, Z = {0:2.1f}'.format(Z_mean,max_freq,phase))

        # if a frequency is redone - remove the previous
        # this is great for getting rid of phase jumps
        if prev_freq == max_freq:
            Z_mean_array[-1] = float(Z_mean)
            max_freq_array[-1] = float(max_freq)
            phase_array[-1] = float(phase)
        else:
            Z_mean_array.append(float(Z_mean))
            max_freq_array.append(float(max_freq))
            phase_array.append(float(phase))
            
        prev_freq = float(max_freq)
        
# interrupt with ctrl+C and plot resulting data

except KeyboardInterrupt:
    # stop stream and terminate audio instantiation
    stream.close()
    audio.terminate()

    # plotting routine
    fig,ax = plt.subplots()
    fig.set_size_inches(12,8)
    p_color = ['#7CAE00','#F8766D']
    p1 = ax.scatter(max_freq_array,Z_mean_array,label='$Z$',c=p_color[0])
    # comment out the plot below if the frequency selection is not sequential
    ax.plot(max_freq_array,Z_mean_array,label='$Z$',c=p_color[0])
    ax2 = ax.twinx()
    p2 = ax2.scatter(max_freq_array,phase_array,label='$\phi$',c=p_color[1])
    # comment out below for non-sequential frequencies
    ax2.plot(max_freq_array,phase_array,label='$\phi$',c=p_color[1])
    subplot_vec = [p1,p2]
    ax2.legend(subplot_vec,[l.get_label() for l in subplot_vec],fontsize=20)
    ax.yaxis.label.set_color(p_color[0])
    ax2.yaxis.label.set_color(p_color[1])
    ax.set_ylabel('Impedance [$\Omega$]',fontsize=16)
    ax2.set_ylabel('Phase [Degrees]',fontsize=16)
    ax2.grid(False)
    ax.spines["right"].set_edgecolor(p_color[0])
    ax2.spines["right"].set_edgecolor(p_color[1])
    ax.tick_params(axis='y', colors=p_color[0])
    ax2.tick_params(axis='y', colors=p_color[1])
    ax.set_xlabel('Frequency [Hz]',fontsize=16)
    Z_max_text = ' = {0:2.1f} $\Omega$'.format(np.max(Z_mean_array))
    f_max_text = ' = {0:2.1f} Hz'.format(max_freq_array[np.argmax(Z_mean_array)])
    ax.annotate('$f_{max}$'+f_max_text+', $Z_{max}$'+Z_max_text,xy=(max_freq_array[np.argmax(Z_mean_array)],np.max(Z_mean_array)),\
                xycoords='data',xytext=(50,-5),size=14,textcoords='offset points',
                       arrowprops=dict(arrowstyle='simple',
                                       fc='0.6',ec='none'))
    phase_min_loc = np.argmin(np.abs(phase_array))
    f_max_phase = '= {0:2.1f} Hz'.format(max_freq_array[phase_min_loc])
    Z_max_phase = ' = {0:2.1f} $\Omega$'.format(Z_mean_array[phase_min_loc])
    ax2.annotate('$f_{max}$'+f_max_phase+', $Z_{max}$'+Z_max_phase,xy=(max_freq_array[phase_min_loc],phase_array[phase_min_loc]),\
                xycoords='data',xytext=(60,-5),size=14,textcoords='offset points',
                       arrowprops=dict(arrowstyle='simple',
                                       fc='0.6',ec='none'))
    plt.savefig('Z_phi_manual_resonance.png',dpi=300,facecolor=[252/255,252/255,252/255])
    plt.show()

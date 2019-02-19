# thiele_small_parameters
Calculation of Thiele-Small parameters using Python's PyAudio library and a stereo USB audio device

The user will need to wire up a USB audio device in a voltage divider configuration to measure voltage across a loudspeaker. This will allow the python algorithm to approximate the impedance of the speaker driver based on the FFT of each channel on the USB audio device. It also uses the phase difference to approximate the resonance frequency.

TWO FILES ARE PRESENT AND FUNCTION AS FOLLOWS:

1. 

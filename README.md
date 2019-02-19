# thiele_small_parameters
Calculation of Thiele-Small parameters using Python's PyAudio library and a stereo USB audio device

The user will need to wire up a USB audio device in a voltage divider configuration to measure voltage across a loudspeaker. This will allow the python algorithm to approximate the impedance of the speaker driver based on the FFT of each channel on the USB audio device. It also uses the phase difference to approximate the resonance frequency.

TWO FILES ARE PRESENT AND FUNCTION AS FOLLOWS:

1. thiele_small_resonance_sweep.py - This file uses an input frequency sweep to approximate the frequency and impedance of the resonance of a loudspeaker
  
2. thiele_small_manual.py - This file lets the user input frequencies manually. This creates a more realistic approximation of both the impedance and phase at resonance. It's more accurate because it uses the RMS value of a full sine wave (assuming the user inputs a constant sine wave). Therefore, once the resonance region is approximated using the code above, the user should move on to the manual method to approximate the actual impedance value.

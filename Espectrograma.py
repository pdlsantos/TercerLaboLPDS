# -*- coding: utf-8 -*-
"""
### Espectrograma

@author: Paulo
"""
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sign
from Funciones import *

Fs, data = wavfile.read("GoodHeart/a0001.wav")


minS1 = 5.34
maxS1 = 5.44

minS2 = 5.73
maxS2 = 5.77

Fs = 1.0*Fs
data = 1.0*data#[:,1]

data = data/np.max(data)

order = 4
cutoff = 200.0 #Hz

b, a = sign.butter(order, 2*cutoff/Fs, "lowpass")

out_butter = sign.filtfilt(b,a, 10000.0*data, padtype = None)

N = data.shape[0]
length = N / Fs
time = np.linspace(0., length, data.shape[0])
Freq = np.linspace(-Fs/2, Fs/2, data.shape[0])

plt.figure()
plt.subplot(2,1,1)
plt.plot(time, out_butter)
plt.xlim(time[0], time[-1])
plt.ylabel("Amplitud")
plt.xlim(4,6.5)

plt.subplot(2,1,2)
plt.specgram(data, NFFT=1024, Fs=Fs, noverlap=256, mode='psd', scale = 'dB')
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.ylim(0,1000)
plt.xlim(4,6.5)
plt.show()


plt.figure()

plt.subplot(2,1,1)
plt.title("Tono S2")
plt.plot(time, out_butter)
plt.ylabel("Amplitud")
plt.xlim(minS2, maxS2)

plt.subplot(2,1,2)
plt.specgram(out_butter, NFFT=32, noverlap = 16, Fs=Fs, mode='magnitude', scale = 'dB')
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.ylim(0,300)
plt.xlim(minS2, maxS2)
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.title("Tono S1")
plt.plot(time, out_butter)
plt.ylabel("Amplitud")
plt.xlim(minS1, maxS1)

plt.subplot(2,1,2)
plt.specgram(out_butter, NFFT=32, noverlap = 16,  Fs=Fs, mode='magnitude', scale = 'dB')
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.ylim(0,300)
plt.xlim(minS1, maxS1)
plt.show()

out_butter_hilbert = sign.hilbert(out_butter)

out_butter_a = out_butter + out_butter_hilbert #Data analítica

amplitude_envelope = np.abs(out_butter_a)
b, a = sign.butter(order, 2*20/Fs, "lowpass")
Ampli_butter = sign.filtfilt(b,a, amplitude_envelope, padtype = None)

instantaneous_phase = np.unwrap(np.angle(out_butter_a))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * Fs)

plt.figure()
#plt.title("Espectrograma 70 BPM")
plt.subplot(2,1,1)
plt.plot(time, out_butter)
plt.plot(time, Ampli_butter)
plt.xlim(time[0], time[-1])
plt.ylabel("Amplitud")
plt.xlim(4,6.5)

plt.subplot(2,1,2)
plt.plot(time[:-1], instantaneous_frequency)
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.ylim(0,1000)
plt.xlim(4,6.5)


plt.show()

plt.figure()
#plt.title("Espectrograma 70 BPM")
plt.subplot(2,1,1)
plt.plot(time, out_butter)
plt.plot(time, Ampli_butter)
plt.ylabel("Amplitud")
plt.xlim(minS1, maxS1)

plt.subplot(2,1,2)
plt.plot(time[:-1], instantaneous_frequency)
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.xlim(minS1, maxS1)
plt.ylim(0,400)

plt.show()


plt.figure()
#plt.title("Espectrograma 70 BPM")
plt.subplot(2,1,1)
plt.plot(time, out_butter)
plt.plot(time, Ampli_butter)
plt.ylabel("Amplitud")
plt.xlim(minS2, maxS2)

plt.subplot(2,1,2)
plt.plot(time[:-1], instantaneous_frequency)
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.ylim(0,400)
plt.xlim(minS2, maxS2)


plt.show()


out_butter

out_mio = np.convolve(out_butter, 1/(np.pi*time))[:out_butter.size]

out_a = out_butter + 1j*out_mio
i_a = np.abs(out_a)
i_p = np.unwrap(np.angle(out_a))
i_f = (np.diff(i_p) / (2.0*np.pi) * Fs)


plt.figure()
#plt.title("Espectrograma 70 BPM")
plt.subplot(2,1,1)
plt.plot(time, out_butter)
plt.plot(time, i_a)
plt.xlim(time[0], time[-1])
plt.ylabel("Amplitud")
plt.xlim(4,6.5)

plt.subplot(2,1,2)
plt.plot(time[:-1], i_f)
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.ylim(0,1000)
plt.xlim(4,6.5)


plt.show()

plt.figure()
#plt.title("Espectrograma 70 BPM")
plt.subplot(2,1,1)
plt.plot(time, out_butter)
plt.plot(time, Ampli_butter)
plt.ylabel("Amplitud")
plt.xlim(minS1, maxS1)

plt.subplot(2,1,2)
plt.plot(time[:-1], instantaneous_frequency)
plt.xlabel("Time [s]")
plt.ylabel("Freq [Hz]")
plt.xlim(minS1, maxS1)
plt.ylim(0,400)

plt.show()


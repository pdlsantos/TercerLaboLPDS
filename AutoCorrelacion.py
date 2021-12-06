# -*- coding: utf-8 -*-
"""
### AUTOCORRELACION

@author: Paulo
"""

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sign
from Funciones import *

Fs, data = wavfile.read("GoodHeart/a0001.wav")

# Tipos de datos
# Data son todos enteros
# Fs es entero

Fs = 1.0*Fs
data = 1.0*data#[:,0]

#data = data/np.max(data)
BPM_MAX = 250.0
BPM_MIN = 30.0

N = data.shape[0]
length = N / Fs
time = np.linspace(0., length, data.shape[0])

ind_max = np.where(np.min(np.abs(time - 60.0/BPM_MAX)) == np.abs(time - 60.0/BPM_MAX))
ind_min = np.where(np.min(np.abs(time - 60.0/BPM_MIN)) == np.abs(time - 60.0/BPM_MIN))


Energia = np.abs(data)**2


## Filtro Butterworth
order = 4
cutoff = 10.0 #Hz
b, a = sign.butter(order, 2*cutoff/Fs, "lowpass")
out_Butter = sign.filtfilt(b,a, 10000.0*Energia, padtype = None)


AutoCorrelacion = sign.correlate(out_Butter, out_Butter)

AutoCorrelacion = AutoCorrelacion / np.max(AutoCorrelacion)
plt.figure()
plt.plot(time, AutoCorrelacion[int(AutoCorrelacion.shape[0]/2):])
plt.xlabel("Tiempo de Retardo [s]")
plt.ylabel("Amplitud [ua]")
plt.grid()
plt.show()

plt.figure()
plt.plot(time, AutoCorrelacion[int(AutoCorrelacion.shape[0]/2):])
plt.xlabel("Tiempo de Retardo [s]")
plt.ylabel("Amplitud [ua]")
plt.grid()
plt.xlim(0,2)
plt.show()

Interleaving = 10
Auto = AutoCorrelacion[int(AutoCorrelacion.shape[0]/2):int(AutoCorrelacion.shape[0]*8/14):Interleaving]

Auto = [Auto[i] if (time[i*Interleaving]>time[ind_max] and time[i*Interleaving]<time[ind_min]) else 0 for i in range(Auto.size)]
Auto = np.array(Auto)

Auto = Auto/np.max(Auto)

B = np.array(np.where(Auto > 0.95*np.max(Auto)))[0]
Pico = np.argmax(Auto) if np.size(np.where(Auto > 0.9*np.max(Auto))) == 0 else B[-1]

print("El Ritmo Cardíaco es aproximadamente ", 60.0/(1.0*np.array(time[Pico*Interleaving])))





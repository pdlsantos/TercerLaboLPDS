# -*- coding: utf-8 -*-
"""
### Analisis Temporal

@author: Paulo.

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
data = 1.0*data#[:,1]

data = data/np.max(data)

N = data.shape[0]
length = N / Fs
time = np.linspace(0., length, data.shape[0])

plt.figure()
plt.plot(time, data)
plt.grid()
plt.xlim(4,6.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [ua]')
plt.title('Señal Cruda')
plt.text(5.45,0.24,"S1", fontsize = 12, bbox ={'facecolor':'white', 'pad':5})
plt.text(5.82,0.54,"S2", fontsize = 12,  bbox ={'facecolor':'white', 'pad':5})
plt.show()
 

energia = np.abs(data)**2
energia  = energia / np.max(energia)


# 760 - 917
# 1489 - 1594
#  0.063965656466875

Energia_media = np.sum(energia)/energia.size
nivel = [Energia_media for a in time]

plt.figure()
plt.plot(time, energia)
plt.plot(time, nivel)
plt.grid()
plt.xlim(4,6.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Energía normalizada')
plt.show()

order = 4
cutoff = 10.0 #Hz
b, a = sign.butter(order, 2*cutoff/Fs, "lowpass")

E_B = sign.filtfilt(b,a, energia, padtype = None)

plt.figure()
plt.plot(time, E_B, label = 'Energía Filtrada')
plt.plot(time, nivel, label = 'Energía media')
plt.grid()
plt.xlim(4,6.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Energía filtrada [ua]')
plt.legend()
plt.show()

Salto = [1 if a > Energia_media else 0 for a in E_B]

plt.figure()
plt.plot(time, Salto)
plt.grid()
plt.xlim(4,6.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Pulsos detectados')
plt.show()

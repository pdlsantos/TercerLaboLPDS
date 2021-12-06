# -*- coding: utf-8 -*-
"""
## Analisis de Fourier

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
 


plt.figure()
plt.plot(time, data)
plt.grid()
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [ua]')
plt.title('Señal Cruda')
plt.show()
 
## Filtro Butterworth

order = 4
cutoff = 120.0 #Hz

b, a = sign.butter(order, 2*cutoff/Fs, "lowpass")

out_Butter = sign.filtfilt(b,a, 10000.0*data, padtype = None)

plt.figure()
plt.plot(time, out_Butter)
plt.grid()
plt.xlim(4,6.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [ua]')
plt.title('Señal con Filtro Butterworth')
#plt.text(5.45,2400,"S1", fontsize = 12, bbox ={'facecolor':'white', 'pad':5})
#plt.text(5.82,5400,"S2", fontsize = 12,  bbox ={'facecolor':'white', 'pad':5})
plt.show()


## Analisis en Frecuencia

Freq = np.linspace(-Fs/2, Fs/2, data.shape[0])*60.0

data_FFT = np.fft.fft(data)
data_FFT = np.fft.fftshift(data_FFT)

out_Butter_FFT = np.fft.fft(out_Butter)
out_Butter_FFT = np.fft.fftshift(out_Butter_FFT)

plt.figure()
plt.plot(Freq, np.abs(data_FFT)**2, label="Señal Cruda")
plt.plot(Freq, np.abs(out_Butter_FFT)**2, label="Señal Filtrada")
plt.grid()
plt.xlim(-150,150)
plt.legend()
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [ua]')
plt.title('Señal con Filtro Butterworth')
#plt.text(5.45,2400,"S1", fontsize = 12, bbox ={'facecolor':'white', 'pad':5})
#plt.text(5.82,5400,"S2", fontsize = 12,  bbox ={'facecolor':'white', 'pad':5})
plt.show()

## Filtro Más Agresivo

order = 4
cutoff = 10.0 #Hz = 15*60 = 900 bpm

b, a = sign.butter(order, 2*cutoff/Fs, "lowpass")

D2 = np.abs(data)**2

out2_Butter = sign.filtfilt(b,a, D2, padtype = None)

plt.figure()
plt.plot(time, out2_Butter)
plt.grid()
plt.xlim(4,6.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [ua]')
plt.title('Señal con Filtro Butterworth')
#plt.text(5.45,2400,"S1", fontsize = 12, bbox ={'facecolor':'white', 'pad':5})
#plt.text(5.82,5400,"S2", fontsize = 12,  bbox ={'facecolor':'white', 'pad':5})
plt.show()


out2_Butter_FFT = np.fft.fft(out2_Butter)

plt.figure()
#plt.plot(Freq, np.abs(data_FFT)**2, label="Señal Cruda")
plt.plot(Freq, np.abs(np.fft.fftshift(out2_Butter_FFT))**2, label="Señal Filtrada")
plt.grid()
plt.xlim(-150,150)
plt.legend()
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [ua]')
#plt.text(5.45,2400,"S1", fontsize = 12, bbox ={'facecolor':'white', 'pad':5})
#plt.text(5.82,5400,"S2", fontsize = 12,  bbox ={'facecolor':'white', 'pad':5})
plt.show()


Ind_max = int(150.0*data.shape[0]/(60.0*Fs))+1

Rango_Busqueda = 10.0*out2_Butter_FFT[:Ind_max]

Umbral = CFAR(1e-6, 24, 1e-10, Rango_Busqueda, Guarda = 1)

plt.figure()
plt.plot(60*(np.arange(Rango_Busqueda.size))*Fs/data.shape[0], np.abs(Rango_Busqueda)**2*1e-7)
plt.plot(60*(np.arange(Rango_Busqueda.size))*Fs/data.shape[0], Umbral*1e-7)
plt.xlabel("Frecuencia [BPM]")
plt.ylabel("Amplitud [ua]")
plt.show()

Salida_FINAL = Detector1(Rango_Busqueda, Umbral)

plt.figure()
plt.scatter(np.arange(Salida_FINAL.size), Salida_FINAL)
plt.show()


Frecuencia = 60*(np.argmax(Salida_FINAL[3:])+3)*Fs/data.shape[0]

print("La Frecuencia Cardíaca es ", np.round(Frecuencia, 2), "  BPM")

Rango_Busqueda  = Rango_Busqueda / np.max(np.abs(Rango_Busqueda))

Comp = np.abs(Rango_Busqueda)**2
output = np.zeros((Rango_Busqueda.size,))
    
for r in range(Rango_Busqueda.size):
    output[r] = np.exp(-r) if Comp[r]  > 0.1 else 0
        
Frecuencia = 60*(np.argmax(output[3:])+3)*Fs/data.shape[0]

print("La Frecuencia Cardíaca es ", np.round(Frecuencia, 2), "  BPM")




"""
@author: Paulo
"""
import numpy as np
import math

def rect(x):
    return np.where(np.abs(x)<=0.5, 1 + 0j, 0j)

def ref_chirp(signo, W, tp, fm):
    K = signo*W*(1j)*np.pi/tp
    t = np.linspace(-tp/2, tp/2, int(tp*fm), dtype = np.complex128)
    t2 = t*t
    out = rect(t/tp)*np.exp(K*t2)
    return out

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def fastconv(A,B):
    lengthC = np.size(A) + np.size(B) - 1
    sizefft = next_power_of_2(lengthC)
    fftA = np.fft.fft(A, n = sizefft, norm = "ortho")
    fftA = np.fft.fftshift(fftA)
    fftB = np.fft.fft(B, n = sizefft, norm = "ortho")
    fftB = np.fft.fftshift(fftB)
    fftY = np.sqrt(sizefft)*fftA*fftB
    print(np.shape(fftA))
    print(np.shape(fftB))
    print(np.shape(fftY))
    y = np.fft.ifft(fftY, n = sizefft, norm = "ortho")
    return y[:lengthC-1]

# Filtro adaptado para una un canal
def filtroAdaptado(y,h):
    shy = np.shape(y)
    output = np.zeros((shy[0] + np.size(h)-1, shy[1]), dtype = np.complex128)
    
    for i in range(shy[1]):
        output[:,i] =  np.convolve(h, y[:,i])
    return output 

# Filtro adaptado para una un datacube
def DC_filtroAdaptado(y,h):
    # Depende de la forma de y
    
    shy = np.shape(y)
    output = np.zeros((shy[0], shy[1] + np.size(h)-1, shy[2]))
    
    # shy[0] canales
    # shy[1] tiempo rapido
    # shy[2] tiempo lento
    
    for j in range(shy[0]):
        for i in range(shy[1]):
            output[j, :, i] =  np.convolve(h, y[j, :, i])
        return output 

def cancelador2pulsos(y,h):
    s2 = np.convolve(h, y[:,1])
    s1 = np.convolve(h, y[:,0])
    return s2 - s1

def cancelador2pulsosMAX(y,h):
    s = filtroAdaptado(y,h)
    lens = np.size(s[0,:])
    S = s[:,:lens-1] - s[:,1:]
    return S

def STI2(y,h):
    s = filtroAdaptado(y,h)
    lens = np.size(s[0,:])
    S = s[:,:lens-1] + s[:,1:]
    return S

def STI3(y,h):
    s = filtroAdaptado(y,h)
    lens = np.size(s[0,:])
    S = s[:,:lens-2] + 2*s[:,1:lens-1] + 2*s[:,2:] 
    return S

def cancelador3pulsos(y,h):
    s3 = np.convolve(h, y[:,2])
    s2 = np.convolve(h, y[:,1])
    s1 = np.convolve(h, y[:,0])
    return s3 -2*s2 + s1

def cancelador3pulsosMAX(y,h):
    s = filtroAdaptado(y,h)
    lens = np.size(s[0,:])
    S = s[:,:lens-2] - 2*s[:,1:lens-1] + 2*s[:,2:] 
    return S

def rango(tp,fm, T):
    c = 3e8
    Ro = T*c/2
    Cant = tp*fm-1
    x = np.arange(Cant)
    return (Ro + (x-1)*c/(fm*2))/1000 # En kilometros

def velocidad(PRF, fc, elems):
    Frec = np.linspace(PRF*(-1/2), PRF/2, elems)
    lda = 3e8/fc
    return Frec*lda/2

def filtroDoppler(y):
    shy = np.shape(y)
    TL = shy[1]
    TR = shy[0]
    
    output = np.zeros((TR,TL), dtype = np.complex128)
    
    for i in range(TR):
        output[i,:] = np.fft.fft(y[i,:])
    
    return output

def filtroDoppler2(y):
    
    shy = np.shape(y)
    TL = shy[1]
    TR = shy[0]
    
    MatrizFD = np.zeros((TL,TL), dtype = np.complex128)
    
    for r in range(TL):
        for l in range(TL):
            MatrizFD[r,l] = np.exp(-1j*2*np.pi*r*l/TL)
    
    return y@MatrizFD

def MatrizFiltrado(sh1,N,G):
    # N Tamano de cada ventana
    # G tamano de guardia
    
    if (2*(N+G)+1) > sh1:
        print("Los tamanos de N y G no machean con el tamano total")
        return
    
    MatrizFiltro = np.zeros((sh1,sh1))
    Pesos = np.zeros(N)+1                       # Son las ventanas
    for g in range(G+1):
        MatrizFiltro[g, (1+g+G):N+(1+g+G)] = Pesos
    # Relleno los primerios
    
    for n in range(1,N):
        MatrizFiltro[G+n, (2*G+1)+n:N+(2*G+1)+n] = Pesos
        MatrizFiltro[G+n,:n] = Pesos[-n:]
    
    Fila = np.zeros(sh1)
    Fila[:N] = Pesos
    Fila[(N+2*G+1):N+(N+2*G+1)] = Pesos
    
    for n in range((sh1-2*(N+G)-1)):
        MatrizFiltro[N+G+n,:] = np.roll(Fila, n)
    
    for g in range(G+1):
        MatrizFiltro[sh1-1-g, -N-(1+g+G):-(1+g+G)] = Pesos
    # Relleno los primerios
    
    
    for n in range(1,N+1):
        MatrizFiltro[sh1-1-n-G,-N-(2*G+1)-n:-(2*G+1)-n] = Pesos
        MatrizFiltro[sh1-1-n-G,-n:] = Pesos[:n]
    
    return MatrizFiltro

def ProbBinario(cant, total,M, SNR, alpha):
    p = (1+(alpha/(M*(1+SNR))))**(-M)
    elementos = [math.comb(total, r)*(p**r)*(1-p)**(total-r) for r in range(cant, total+1)]
    return sum(elementos)

def CFAR(Pfa, M, SNR, y, Guarda = 5): # N va a ser la mitad
    y = np.abs(y)
    N = int(M/2)
    
    # Constantes utiles
    
    y = np.abs(y)*np.abs(y)
    sh1 = y.shape[0]
    alpha = M*(Pfa**(-1/M)-1)
    
    MatrizFiltro = MatrizFiltrado(sh1, N, Guarda)
    output = MatrizFiltro.dot(y)
    return alpha*output/M
    

def Detector(y,Umbral, integBinaria = False, cant = 1):
    
    # Constantes utiles
    sh1 = y.shape[0]
    sh2 = y.shape[1]
    y = np.abs(y)*np.abs(y)
    output = np.zeros((sh1,sh2), dtype = np.complex128)
    
    for r in range(sh1):
        for l in range(sh2):
            output[r,l] = 1 if y[r, l] > Umbral[r, l] else 0
            
    if integBinaria == False:
        return output
    else:
        print("No usar con filtro Doppler")
        Integrador = np.zeros((sh2,1))+1
        output = output.dot(Integrador)
        for r in range(sh1):
            output[r] = 1 if output[r] >= cant else 0
        return output
    
def Detector1(y, Umbral):
    
    # Constantes utiles
    y = np.abs(y)**2
    
    output = np.zeros((y.size,))
    
    for r in range(y.size):
        output[r] = 1 if y[r] > np.real(Umbral[r]) else 0
            
    return output
    


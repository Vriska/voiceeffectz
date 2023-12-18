from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cmath
import math
samplerate, data = read(r'C:\Users\Vriska S\Desktop\wavtest2.wav')
freq = 2000
data = [1000*np.cos(2*np.pi*10*x/samplerate) for x in range(0,len(data))]
plt.subplot(4, 1, 1)
plt.plot(np.arange(0,samplerate,samplerate/len(data)),data)
#print(data)
fftdata = np.fft.fft(data)
#plt.subplot(4, 1, 1)
#plt.stem(np.arange(0,samplerate,samplerate/len(data)),np.abs(fftdata))
# plt.subplot(4, 1, 1)
# plt.plot(np.arange(0,samplerate,samplerate/len(h)),np.abs(np.fft.fft(h)))
timelen =len(data)/samplerate
t = np.arange(0,timelen,1/samplerate)
# (0,0) (timelen/2,1),(timelen,0)
xta= [0,timelen]
yta= [0,2]
xtf =[0,timelen/2,timelen]
ytf= [0,5,0]


def linear(x,a,b):
    return x*a +b


def fitting(xt,yt) :
    yo = []
    k=0
    for x in range(0,len(xt)-1) :
        xst = np.arange(xt[x],xt[x+1],1/samplerate)
        if k ==1 :
            xst= xst[1:]
        popt, pcov = curve_fit(linear, [xt[x],xt[x+1]], [yt[x],yt[x+1]])
        yo = np.append(yo,linear(xst,*popt))
        k= 1
    sum= 0
    integral =[]
    for x in yo :
        sum = sum+x
        integral.append(sum)
    integral = [1/len(yo)*x for x in integral]
    listinvert = []
    numbers = []
    for x in range(0,len(yo)-1):
        if yo[x] < yo[x+1] :
            listinvert.append(integral[x])
            numbers.append(x)
        else :
            listinvert=listinvert.reverse()
            for i in range(0,len(numbers)):
                integral[numbers[i]] = listinvert[i]
            listinvert=[]
            numbers=[]


    print(len(integral))
    return integral
plt.subplot(4, 1, 2)
plt.plot(np.arange(0,samplerate,samplerate/len(data)),fitting(xtf,ytf))
plt.show()

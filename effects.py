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
xta= [0,timelen/2,timelen]
yta= [2,0,2]
xtf =[0,timelen/2,timelen]
ytf= [0,5,5]


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
    #integral = [x/max(integral) for x in integral]



    print(len(integral))
    return integral

aenvelope=fitting(xta,yta)
fenvelope=fitting(xtf,ytf)
plt.subplot(4, 1, 2)
plt.plot(t,fenvelope)


def filter(data) :
    h =[] #Enter filter coefficients here
    #plt.subplot(4, 1, 1)
    #plt.plot(np.arange(0,samplerate,samplerate/len(h)),np.abs(np.fft.fft(h)))
    plt.show()
    data= np.convolve(h,data)
    return data


def ssb(data,fenvelope,k ) :
    #fenvelope =[0.2 for n in range(0,len(data))]
    ssbcos = []
    ssbsine =[]
    ssc=[]
    #fenvelope =[3 for x in range(0,len(data))]
    p=0
    l=np.arange(0,len(data),1)
    for n in range(0,len(data)):
        freq = fenvelope[n]
        ssbcos.append(np.cos((2*np.pi*(1+freq))/samplerate*l[n]))
        ssbsine.append(np.sin((2*np.pi*(1+freq))/samplerate*l[n]))
        p = p+1
    hilbertt = np.imag(signal.hilbert(data))
    for n in range(0,len(data)-1):
        if k==1 :
            ssc.append(data[n] * ssbcos[n] - hilbertt[n] * ssbsine[n])
        elif k==-1:
            ssc.append(data[n] * ssbcos[n] + hilbertt[n] * ssbsine[n])
    ssc.append(0)
    print(len(ssc))
    return ssc

def dsbsc(data, aenvelope) :
    dsbsc = [data[n]*aenvelope[n] for n in range(0,len(data))]
    return dsbsc

hilbertt= np.imag(signal.hilbert(data))


#new = np.convolve(dsbsc,hx)
#newmag = np.abs(np.fft.fft(new))
#plt.subplot(4, 1, 4)
#plt.plot(np.arange(0,samplerate,samplerate/len(ssc)),np.abs(np.fft.fft(ssc)))
l=np.arange(0,len(data),1)

#print(hilbertt)

plt.subplot(4, 1, 3)
plt.plot(np.arange(0,samplerate,samplerate/len(data)),(ssb(data,fenvelope,1)))
plt.show()
write(r'C:\Users\Vriska S\Desktop\wavtest3.wav', samplerate, np.int16(ssb(data,fenvelope,1)))


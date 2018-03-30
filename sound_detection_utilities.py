import wave, numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd

'''
Feature Prep Related
'''
def audio_prep(file_name,txt_name,plot=1,listen=1,noise=0,nperseg=256, noverlap=32):
    fs,x_wave = wavfile.read(file_name)
    interval = np.genfromtxt(txt_name, delimiter='\t')
  
    x = x_wave.astype(float)
    if(noise>0):
        x += np.random.normal(0, x.max()*noise,x.shape)
        
    x_play = x.astype(int)    
    # spectrogram
    if(x.shape==(x.shape[0],)):
        f, t, Sxx = signal.spectrogram(x, fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)
    else:
        f, t, Sxx = signal.spectrogram(x[:,0], fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)

    Sxx /= Sxx.max()
    labels = np.zeros(Sxx.shape[1])
    print("Sxx.shape=",Sxx.shape)
    
    print("\nRaw spectrogram")
    plt.pcolormesh(t, f, np.log(Sxx+1e-8))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    print("interval.shape=",interval.shape)       
    if(interval.shape == (2,)):     
        if(listen==1):
            sd.play(x_play,fs,blocking=True)
        bb,ee = int(interval[0]*fs),min(int(interval[1]*fs),(x.shape[0]))
        print("interval=",interval)
        if(plot == 1):
            
            print("fs=",fs)
            print(x.shape)
            plt.plot(np.linspace(start=0,stop=(x_play.shape[0]/fs),num=x_play.shape[0]),\
                    np.linspace(start=0,stop=0,num=x_play.shape[0]))
            plt.plot(np.linspace(start=interval[0],stop=interval[1],num=int((interval[1]-interval[0])*fs)),\
                   x_play[bb:ee,0])
            plt.plot(np.linspace(start=0,stop=interval[0],num=int((interval[0])*fs)),\
                   x_play[0:bb,0])
            if(ee<x.shape[0]):
                plt.plot(np.linspace(start=interval[1],stop=x.shape[0]/fs,num=int((x.shape[0]-ee-1))),\
                   x_play[ee+1:,0])
            #plt.plot(np.linspace(start=0,stop=x.shape[0]/fs,num=int(x.shape[0])),x[:,0])
            plt.show()

        #Sxx  = 1-Sxx
        t_bb= int(t.shape[0]*bb/x_play.shape[0])+1
        t_ee= int(t.shape[0]*ee/x_play.shape[0])
        print("t_begin=",t_bb,"t_end=",t_ee)
        
        # laybel_gen - 0 is everything else, 1 is scream class            
        for i in range(t_bb,t_ee):
            labels[i] = 1

        for i in range(0,len(labels)):
            labels[i] = int(labels[i])
        print("Raw. Sxx.shape=",Sxx.shape) 
        labels = np.hstack((labels[max(int(t_bb/2),int(t_bb-300)):t_ee],labels[max(int(t_bb/3),int(t_bb-100)):min(t_ee+10,Sxx.shape[1])]))
        Sxx    = np.hstack(( Sxx[:,max(int(t_bb/2),int(t_bb-300)):t_ee], Sxx[:,max(int(t_bb/3),int(t_bb-100)):min(t_ee+10,Sxx.shape[1])]))

        t = np.linspace(start=0,stop=Sxx.shape[1],num =Sxx.shape[1])
        
        if(plot == 1):
            plt.plot(labels)           
            plt.show()
            plt.plot(Sxx.T[t_bb-1])
            plt.show()
        print("Truncated. Sxx.shape=",Sxx.shape)     
        print("\nReturning spectrogram - single")
        plt.pcolormesh(t, f[0:int(Sxx.shape[0]*0.6)], np.log(Sxx[0:int(Sxx.shape[0]*0.6),:]+1e-8))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Number of frames')
        plt.show()
        return Sxx.T,labels,fs
    
    else:
        Sxx_prev = Sxx[:,0:2]
        labels_prev = np.zeros(Sxx_prev.shape[1])
        print("interval=",interval)
        print("fs=",fs)
        print(x.shape)
        prev_ee = 0;
        for j in range(0,interval.shape[0]):
            bb,ee = int(interval[j][0]*fs),int(interval[j][1]*fs)  

            # get the frame number of the starting and ending frames
            t_bb= int(t.shape[0]*bb/x_play.shape[0])+1
            t_ee= int(t.shape[0]*ee/x_play.shape[0])
            Sxx_prev = np.hstack((Sxx_prev, Sxx[:,max(prev_ee,int(t_bb-300)):min(Sxx.shape[1],t_ee+10)]))
            # laybel_gen - 0 is everything else, 1 is scream class            
            for i in range(t_bb,t_ee):
                labels[i] = 1
            for i in range(0,len(labels)):
                labels[i] = int(labels[i])
            
            labels_prev   = np.hstack((labels_prev,labels[max(prev_ee,int(t_bb-300)):min(Sxx.shape[1],t_ee+10)]))
            print("t_begin=",t_bb,"t_end=",t_ee)
            prev_ee = t_ee
            
        t_tmp = np.linspace(start=0,stop=Sxx_prev.shape[1],num =Sxx_prev.shape[1])
        #t_tmp = np.linspace(start=0,stop=Sxx_prev.shape[1]*x_play.shape[0]/t.shape[0]/fs,num =Sxx_prev.shape[1])
        print("\nReturning spectrogram - multiple")
        plt.pcolormesh(t_tmp, f[0:int(Sxx.shape[0]*0.6)],np.log(Sxx_prev[0:int(Sxx.shape[0]*0.6),:]+1e-8))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Number of frames')
        plt.show()
        
        return Sxx_prev.T,labels_prev,fs
    
def cross_train_prep(file_name,plot=1,listen=1,nperseg=256, noverlap=32):
    fs,x_wave = wavfile.read(file_name)
    
    x = x_wave.astype(float)
        
    x_play = x.astype(int)    
    # spectrogram
    if(x.shape==(x.shape[0],)):
        f, t, Sxx = signal.spectrogram(x, fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)
    else:
        f, t, Sxx = signal.spectrogram(x[:,0], fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)

    #Sxx /= Sxx.max()
    #Sxx = -np.log(Sxx+1e-9)
    Sxx /= Sxx.max()
    labels = np.zeros(Sxx.shape[1])
    #print("Sxx.shape=",Sxx.shape)
    #
    #print("\nRaw spectrogram")
    #plt.pcolormesh(t, f, np.log(Sxx+1e-8))
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
     
    if(listen==1):
        sd.play(x_play,fs,blocking=True)

    return Sxx.T,labels,fs    

'''
MFCC Related
'''

def get_mfcc_filter_bank(Sxx_in,num_bank,fs):
    m = np.linspace(start=1125 * np.log(1 + 10 / 700.0),stop=1125 * np.log(1 + fs*0.8 / 700.0),num=num_bank)
    # get list of freq centers
    print(m)
    f = np.zeros(len(m)+1)
    f[0:len(m)] = 700 * (np.exp(m / 1125) - 1)
    # get list of indices
    f = np.floor((Sxx_in.shape[1]+1)*f/fs)
    f[len(m)] = int(f[num_bank-1]*f[num_bank-1]/f[num_bank-2])
    print(f)
    # get coef for each filter bank for all time steps
    filter_bank = np.zeros((num_bank,Sxx_in.shape[1]))
    for i in range(0,num_bank):             
        for k in range(Sxx_in.shape[1]):        
            if(f[i] == f[i-1] or f[i] == f[i+1]): # at DC there might be large overlap 
                if(k == f[i]):
                    filter_bank[i][k] = 1;
                else:
                    filter_bank[i][k] = 0;
            elif(f[i-1] <= k and k <= f[i]):
                filter_bank[i][k] = (k-f[i-1]) / (f[i] - f[i-1])/(f[i+1] - f[i])
            elif(f[i] <= k and k <= f[i+1]):
                filter_bank[i][k] = (f[i+1]-k) / (f[i+1] - f[i])/(f[i+1] - f[i])
                
    return filter_bank;

def Sxxin_to_mfcc(Sxx_in,num_bank,fs,filter_bank):
    from scipy import fftpack
    mfcc_mat = np.zeros((Sxx_in.shape[0],num_bank)) # dim = (t,num_bank), same as Sxx_in

    for t in range(0,Sxx_in.shape[0]):
        mfcc_mat[t] = fftpack.dct(np.log(1e-15+np.dot(filter_bank,Sxx_in[t])))
    
    print(mfcc_mat.shape)
    return mfcc_mat;

def mfcc_difference(mfcc_mat):
    diff_mat = np.zeros_like(mfcc_mat);
    
    for t in range(1,mfcc_mat.shape[0]):
        diff_mat[t] = mfcc_mat[t] - mfcc_mat[t-1]
    return diff_mat
import yaml,os
import sys
import shutil
import librosa
import soundfile as sf, numpy as np, pandas as pd
import argparse, textwrap
import wave, numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile

from IPython import embed
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from tqdm import tqdm
import hashlib
import warnings


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
        


    #Sxx /= Sxx.max()
    #Sxx = -np.log(Sxx+1e-9)
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
        Sxx = np.hstack((Sxx[:,max(int(t_bb/2),int(t_bb-300)):t_ee],Sxx[:,max(int(t_bb/3),int(t_bb-100)):t_ee]))
        labels = np.hstack((labels[max(int(t_bb/2),int(t_bb-300)):t_ee],labels[max(int(t_bb/3),int(t_bb-100)):t_ee]))
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
            Sxx_prev = np.hstack((Sxx_prev, Sxx[:,max(prev_ee,int(t_bb-300)):min(Sxx.shape[1],t_ee)]))
            # laybel_gen - 0 is everything else, 1 is scream class            
            for i in range(t_bb,t_ee):
                labels[i] = 1
            for i in range(0,len(labels)):
                labels[i] = int(labels[i])
            
            labels_prev   = np.hstack((labels_prev,labels[max(prev_ee,int(t_bb-300)):min(Sxx.shape[1],t_ee)]))
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


''' DCASE prep '''
def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def prep_from_yaml(data,audio_path,listen,noise,trunc,nperseg, noverlap):
    
    x_wave,fs = sf.read(audio_path+"\\"+data['mixture_audio_filename'])

    x = x_wave.astype(float)
    if(noise>0):
        x += np.random.normal(0, x.max()*noise,x.shape)       
    x_play = x.astype(np.int32)    
    # spectrogram
    f, t, Sxx = signal.spectrogram(x, fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)
    #Sxx /= Sxx.max()
    labels = np.zeros(Sxx.shape[1])
    print("Sxx.shape=",Sxx.shape)
    
    #print("\nRaw spectrogram")
    #plt.pcolormesh(t, f, np.log(Sxx+1e-8))
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
  
    if(listen==1):
        sd.play(x_play,fs,blocking=True)
        
    if(data['event_present'] == True):
        
        bb = int(data['event_start_in_mixture_seconds']*fs)
        ee = min(int(bb+fs*data['event_length_seconds']),(x.shape[0]))

        #Sxx  = 1-Sxx
        t_bb= int(t.shape[0]*bb/x_play.shape[0])+1
        t_ee= int(t.shape[0]*ee/x_play.shape[0])
        print("t_begin=",t_bb,"t_end=",t_ee)

        # laybel_gen - 0 is everything else, 1 is scream class            
        for i in range(t_bb,t_ee):
            labels[i] = 1

    for i in range(0,len(labels)):
        labels[i] = int(labels[i])
        
    if(data['event_present'] == True and trunc == True):  
        if(t_bb > 1500):
            Sxx = Sxx.T
            Sxx = Sxx[t_bb-1500:]
            Sxx = Sxx.T
            labels = labels[t_bb-1500:]
    
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
    f = np.floor((Sxx_in.shape[1]+1)*f/fs*2)
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

def feature_for_sound(Sxx_mfcc_raw, num_bank,diff_by_time= True, diff = True,diff_diff = False, zero_cross_rate = False):
    
    Sxx_mfcc = Sxx_mfcc_raw[:,2:num_bank-4]
    if(diff_by_time == True):
        Sxx_mfcc_diff = mfcc_difference(Sxx_mfcc)
    else:
        Sxx_mfcc = Sxx_mfcc.T
        Sxx_mfcc_diff = mfcc_difference(Sxx_mfcc)
        Sxx_mfcc = Sxx_mfcc.T
        Sxx_mfcc_diff = Sxx_mfcc_diff.T
        Sxx_mfcc_diff = Sxx_mfcc_diff[:,1:]
    Feature = Sxx_mfcc;
    if(diff == True):
        Feature = np.hstack((Feature,Sxx_mfcc_diff))  
    if(zero_cross_rate == True):
        Sxx_zcr = Sxx_mfcc_diff; # dummy here. Needs to be implemented
        Feature = np.hstack((Feature,Sxx_zcr)) 
    if(diff_diff == True):
        if(diff_by_time == True):
            Sxx_mfcc_diff_diff = mfcc_difference(Sxx_mfcc_diff)
        else:
            Sxx_mfcc_diff = Sxx_mfcc_diff.T
            Sxx_mfcc_diff_diff = mfcc_difference(Sxx_mfcc_diff)
            Sxx_mfcc_diff = Sxx_mfcc_diff.T
            Sxx_mfcc_diff_diff = Sxx_mfcc_diff_diff.T
            Sxx_mfcc_diff_diff = Sxx_mfcc_diff_diff[:,1:]
        Feature = np.hstack((Feature,Sxx_mfcc_diff_diff)) 
    
    return Feature;

''' Normalization '''

def mean_var_normalization(Sxx_in, byRow = True):
    
    if(byRow == False):
        Sxx_in = Sxx_in.T
        
    for i in range(Sxx_in.shape[0]):
        Sxx_in[i] -= np.mean(Sxx_in[i]) 
        Sxx_in[i] /= np.std(Sxx_in[i]) 
        
    if(byRow == False):
        Sxx_in = Sxx_in.T
        
    return Sxx_in

''' Event wise performance metric '''
# stat for event wise metric

def event_metric(output_label, labels, margin):
    
    true_event_cnt = False
    true_event     = labels[0]
    tp_counted = False
    fp_counted = False
    fn_counted = False
    event_tpos, event_fpos, event_fneg = 0,0,0
    
    onsets = []
    offsets= []
    
    # prevent invalid read
    onsets.append(0)
    offsets.append(0)
    
    # find all onsets and offsets
    for i in range(1,output_label.shape[0]):            
        if(labels[i] == 1 and labels[i-1] == 0):
            onsets.append(i)
        if(labels[i] == 0 and labels[i-1] == 1):
            offsets.append(i)
            
    # prevent invalid read
    onsets.append(output_label.shape[0])
    offsets.append(output_label.shape[0]+1)
    
    # count true positive & and false negative
    onsets_count = 0
    offsets_count = 0
    for i in range(1,output_label.shape[0]):       
        if(labels[i] == 1 and labels[i-1] == 0):
            onsets_count += 1;
        if(labels[i] == 0 and labels[i-1] == 1):
            offsets_count += 1;
            tp_counted = False; # lower the flag of true_positive counted
        if(labels[i] == labels[i-1] and labels[i] == 1):
            # within the same TRUE event
            if(output_label[i] == 1):
                # if output is TP
                fn_counted = False; # lower the flag of false_negative counted
                if(tp_counted == False):
                    # if this TP isn't counted yet
                    event_tpos += 1;
                    tp_counted = True;
            else:
                # if output is FN
                
                if(fn_counted == False and tp_counted == False): 
                    # if this FN isn't counted yet
                    if(i > (onsets[onsets_count]+margin) and i < (offsets[onsets_count]-margin)):
                        # allow some margins
                        event_fneg += 1;
                        fn_counted = True;   
        else:
            tp_counted = False
            fn_counted = False
            
    # count false positive
    fp_counted = False
    onsets_count = 0
    for i in range(1,output_label.shape[0]):        
        if(labels[i] == 1 and labels[i-1] == 0):
            onsets_count += 1;        
        if(labels[i] == labels[i-1] and labels[i] == 0):
            # within the same event
            if(output_label[i] == 1):
                # if output is FP
                if(fp_counted == False):
                    # if this FP isn't counted yet
                    if(i > (offsets[onsets_count]+margin) and i < (onsets[onsets_count+1]-margin)):
                        # allow some margins
                        event_fpos += 1;
                        fp_counted = True;
            else:
                 fp_counted = False;
        else:
            fp_counted = False;   
            
    return event_tpos, event_fpos, event_fneg      


            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:21:16 2019

@author: sleek_eagle
"""

from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import cnn
from scipy import signal
import math
import cmath
from numpy import diff
from keras.models import load_model
import keras.backend as K
import scipy.io.wavfile as wav
import os


#hop (stride) used in FFT. Time of the recording = hop*length_of_array ms
fft_hop=10
#desired length of sample in ms
sample_len=2000
frame_len=fft_hop=int(sample_len/fft_hop)


path='/home/sleek_eagle/research/security/project/results/11.2_0.14/'

emotions=np.array(['W','L','E','A','F','T','N'])
def get_emo_num(file_name):
    emotion=file_name[5]
    emo_num=to_onehot(np.where(emotions==emotion)[0][0])
    return emo_num

def to_onehot(num):
    num_classes=emotions.shape[0]
    out = np.empty([0,num_classes])
    for x in np.nditer(num):
        onehot = np.zeros(num_classes)
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out

def get_file_list(file_path,ext):
    with open(file_path) as f:
        lines = f.readlines()
        files=[]
        for line in lines:
            file_name=line[1:-5]+ext
            files.append(file_name)
    return files

def get_data_list(path,files):
    data=[]
    labels=[]
    for file in files:
        if(len(file)<11):
            continue
        ar=np.load(path+file)
        data.append(ar)
        label=get_emo_num(file)
        labels.append(label)
    return data,labels


def get_spec_voice(voice_list,sr):
    #length of FFT window in ms
    fft_len = 25 
    window_len = int(sr*fft_len*0.001)
    #hop (stride) of FFT in ms
    hop_length = int(sr*fft_hop*0.001)
    #librosa.display.waveplot(voice, sr=sr);
    n_mels = 40
    spec_list=[]
    for voice in voice_list:
        S = librosa.feature.melspectrogram(voice, sr=sr, n_fft=window_len, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        spec_list.append(S_DB)
    return spec_list

def read_voices(files):
    voice_list=[]
    labels=[]
    for file in files:
        y,sr=librosa.load(path+file)
        voice,_=librosa.effects.trim(y)
        voice_list.append(voice)
        label=get_emo_num(file)
        labels.append(label)
    return voice_list,labels,sr

def get_derivatives(spec,voice):
    resampled=signal.resample(voice, spec.shape[1])
    (n_mels,spec_len)=spec.shape
    max_freq=8196
    freqs=list(range(int(max_freq/n_mels),max_freq,int(max_freq/n_mels)))
    dX_dt=np.zeros((n_mels,spec_len))
    for a in range(n_mels):
        for b in range(spec_len):
            val=-2*math.pi*b*freqs[a]
            v=cmath.exp(-val*1j)*resampled[b]
            dX_dt[a][b]=abs(v)
            
    diff_resampled=diff(resampled)
    diff_resampled=np.append(diff_resampled,values=[0],axis=0)
    return dX_dt,diff_resampled

def get_norm_data(ar_list):
    data=[]
    for ar in ar_list:
        #standardize data
        m=np.tile(mean_ar,ar.shape[1])
        s=np.tile(std_ar,ar.shape[1])
        norm_ar=(ar-m)/s
        diff=frame_len-ar.shape[1]
        if(ar.shape[1] < frame_len):
            norm_ar=np.pad(norm_ar,pad_width=((0,0),(0,diff)),mode='mean')
        else:
            max_index=-diff
            start=randint(0,max_index)
            norm_ar=norm_ar[:,start:(start+frame_len)]
        data.append(norm_ar)
    data=np.array(data)
    return data

sess = K.get_session()
def get_adv(labels,input_data):
    loss = 1*K.categorical_crossentropy(labels, model.output)
    grads = K.gradients(loss, model.input)
    res=sess.run(grads,{model.input:input_data})
    return res

#load saved trained model
model=load_model('/home/sleek_eagle/research/security/project/emodb.h5')

def save_files(path,files):
    spec_list=[]
    for file in files:
        y, sr = librosa.load(path+file)
        voice,_=librosa.effects.trim(y)
    
        fft_len = 25 
        window_len = int(sr*fft_len*0.001)
        #hop (stride) of FFT in ms
        hop = 10
        hop_length = int(sr*hop*0.001)
        #librosa.display.waveplot(voice, sr=sr);
        n_mels = 40
        S = librosa.feature.melspectrogram(voice, sr=sr, n_fft=window_len, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        spec_list.append(S_DB)
        print(file)
        file_name=file.split('.')[0]
    return spec_list

def spec_from_voice(voice_list):
    spec_list=[]
    for voice in voice_list:    
        fft_len = 25 
        window_len = int(sr*fft_len*0.001)
        #hop (stride) of FFT in ms
        hop = 10
        hop_length = int(sr*hop*0.001)
        #librosa.display.waveplot(voice, sr=sr);
        n_mels = 40
        S = librosa.feature.melspectrogram(voice, sr=sr, n_fft=window_len, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        spec_list.append(S_DB)
    return spec_list

def get_voice_data(files):
    global sr
    voice_list=[]
    for file in files:
        y, sr = librosa.load(path+file)
        voice,_=librosa.effects.trim(y)
        voice_list.append(voice)
    return voice_list

def get_data_from_voice(voice_list):
    spec_list=spec_from_voice(voice_list)
    test_data=get_norm_data(spec_list)
    test_data=np.expand_dims(test_data,axis=-1)
    return test_data


def save_wav_files(voice_list,res):    
    for i in range(0,len(voice_list)):
        file_name=test_files[i]
        dir_name=str(res[0])[0:4]+"_"+str(res[1])[0:4]
        dir_path='/home/sleek_eagle/research/security/project/results/'+dir_name+'/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        wav.write(dir_path+file_name,sr,voice_list[i])
    
    
#get train data for calculating statistics to normalize
train_files=get_file_list('/home/sleek_eagle/research/security/project/train.txt','npy')
train_list,train_labels = get_data_list('/home/sleek_eagle/research/security/project/data/np_arrays/',train_files)
test_files=get_file_list('/home/sleek_eagle/research/security/project/test.txt','npy')
_,test_labels = get_data_list('/home/sleek_eagle/research/security/project/data/emo_db_test_spectro/',test_files)
test_files=get_file_list('/home/sleek_eagle/research/security/project/test.txt','wav')
test_labels=np.array(test_labels)
test_labels=np.squeeze(test_labels)

#get fft coefficiant wise mean and std for normalization
big_ar=np.empty(shape=(40,1))
for ar in train_list:
    big_ar=np.append(big_ar,ar,axis=1)
big_ar=big_ar[:,1:]
mean_ar=np.mean(big_ar,axis=1)
std_ar=np.std(big_ar,axis=1)
mean_ar=np.reshape(mean_ar,newshape=(mean_ar.shape[0],1))
std_ar=np.reshape(std_ar,newshape=(std_ar.shape[0],1))



voice_list=get_voice_data(test_files)
test_data=get_data_from_voice(voice_list)
adv_voice_list=voice_list

res=model.evaluate(x=test_data,y=test_labels)
#use predicted labels as proxy for ground truth labels
targets=model.predict(x=test_data)
#save modified (adversarial) wav files so we can later wun speech-to-text on them
save_wav_files(adv_voice_list,res)

#perform multi step FGSM in the raw voice data domain
for k in range(0,1000):
    grad=get_adv(targets,test_data)[0]
    der_list=[]
    adv_voice_list_tmp=[]
    epsilon=0.004
    for i in range(0,len(test_data)):
        spec=np.squeeze(test_data[i])
        dX_dt,diff_resampled=get_derivatives(spec,adv_voice_list[i])
        diff_resampled=np.expand_dims(diff_resampled,axis=-1)
        diff_resampled=1/(diff_resampled+0.001)
        m1=np.matmul(dX_dt,diff_resampled)
        g= np.matrix.transpose(np.squeeze(grad[i]))
        mul=np.matmul(g,m1)
        grad_to_add=signal.resample(mul,adv_voice_list[i].shape[0])
        grad_sign=np.sign(grad_to_add)
        adv_voice=adv_voice_list[i]+epsilon*grad_sign[:,0]
        adv_voice_list_tmp.append(adv_voice)
    adv_voice_list=adv_voice_list_tmp
    test_data=get_data_from_voice(adv_voice_list)
    res=model.evaluate(x=test_data,y=test_labels)
    save_wav_files(adv_voice_list,res)
    print(res)

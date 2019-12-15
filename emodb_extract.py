#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:58:21 2019

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

def get_data_list(files):
    path='/home/sleek_eagle/research/security/project/data/np_arrays/'
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

def get_file_list(file_path,ext):
    with open(file_path) as f:
        lines = f.readlines()
        files=[]
        for line in lines:
            file_name=line[1:-5]+ext
            files.append(file_name)
    return files



'''
calculate features
'''''


path='/home/sleek_eagle/research/security/project/results/2.19_0.48/'
def get_spec(file):
    y, sr = librosa.load(path+file)
    #length of FFT window in ms
    fft_len = 25 
    window_len = int(sr*fft_len*0.001)
    #hop (stride) of FFT in ms
    hop = 10
    hop_length = int(sr*hop*0.001)
    voice,_=librosa.effects.trim(y)
    #librosa.display.waveplot(voice, sr=sr);
    n_mels = 40
    S = librosa.feature.melspectrogram(voice, sr=sr, n_fft=window_len, 
                                       hop_length=hop_length, 
                                       n_mels=n_mels)
    return S,voice

def get_spec_voice(voice,sr):
    #length of FFT window in ms
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
    return S


'''
*******************************
create spectrograms from .wav files
**********************************
'''
out_path='/home/sleek_eagle/research/security/project/data/emo_db_test_spectro/'

files = [f for f in listdir(path) if isfile(join(path, f))]

for file in test_files:
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
    '''
    librosa.display.specshow(S_DB, sr=sr, hop_length=220, 
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    '''
    print(file)
    file_name=file.split('.')[0]
    np.save(out_path+file_name,S_DB)
    

#find files names from text files
train_files=get_file_list('/home/sleek_eagle/research/security/project/train.txt','npy')
vali_files=get_file_list('/home/sleek_eagle/research/security/project/vali.txt','npy')
test_files=get_file_list('/home/sleek_eagle/research/security/project/test.txt','npy')
   
#hop (stride) used in FFT. Time of the recording = hop*length_of_array ms
fft_hop=10
#desired length of sample in ms
sample_len=2000
frame_len=fft_hop=int(sample_len/fft_hop)

train_list,train_labels = get_data_list(train_files)
test_list,test_labels = get_data_list(test_files)
vali_list,vali_labels = get_data_list(vali_files)

#get fft coefficiant wise mean and std for normalization
big_ar=np.empty(shape=(40,1))
for ar in train_list:
    big_ar=np.append(big_ar,ar,axis=1)
big_ar=big_ar[:,1:]
mean_ar=np.mean(big_ar,axis=1)
std_ar=np.std(big_ar,axis=1)
mean_ar=np.reshape(mean_ar,newshape=(mean_ar.shape[0],1))
std_ar=np.reshape(std_ar,newshape=(std_ar.shape[0],1))


train_data=get_norm_data(train_list)
train_data=np.expand_dims(train_data,axis=-1)

vali_data=get_norm_data(vali_list)
vali_data=np.expand_dims(vali_data,axis=-1)

test_data=get_norm_data(test_list)
test_data=np.expand_dims(test_data,axis=-1)

train_labels=np.array(train_labels)
train_labels=np.squeeze(train_labels)

vali_labels=np.array(vali_labels)
vali_labels=np.squeeze(vali_labels)

test_labels=np.array(test_labels)
test_labels=np.squeeze(test_labels)
 
#load saved model
from keras.models import load_model
import keras.backend as K
model=load_model('/home/sleek_eagle/research/security/project/emodb.h5')
    
model.evaluate(test_data,test_labels)

S_DB

epsilon=0.01

#create adversarial spectrograms
x=test_data
target=model.predict(x=x)
 
def get_adv(labels,input_data):
    loss = 1*K.categorical_crossentropy(labels, model.output)
    grads = K.gradients(loss, model.input)
    delta = K.sign(grads[0])
    out = input_data+epsilon*delta
    res=sess.run(out,{model.input:input_data})
    return res

get_adv(test_labels,test_data)
sess = K.get_session()

loss_list,acc_list=[],[]  

#target=test_labels
x=test_data

res = model.evaluate(x=x,y=test_labels)
loss_list.append(res[0])
acc_list.append(res[1]) 
#use prediction as a proxy for ground truth labels
target=model.predict(x=test_data)
#perform FGSM for all the data at one step
#range(0,10). So 10 is the number of steps we do FGSM
for i in range(0,10):
    x=get_adv(target,x)
    #evaluate model under adversarial samples
    res=model.evaluate(x=x,y=test_labels)
    loss_list.append(res[0])
    acc_list.append(res[1]) 
    print(res)


#plot results
plt.plot(acc_list)
plt.title('Accuracy vs FGSM steps')
plt.xlabel('FGSM steps')
plt.ylabel('accuracy')
r=list(range(0,21))
plt.xticks(r)




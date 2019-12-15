#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:18:20 2019

@author: sleek_eagle
"""

import speech_recognition as sr
import scipy.io.wavfile as wav
from os import listdir
from os.path import isfile, join




text_dict={"a01":"Der Lappen liegt auf dem Eisschrank",
           "a02":"Das will sie am Mittwoch abgeben",
           "a04":"Heute abend könnte ich es ihm sagen",
           "a05":"Das schwarze Stück Papier befindet sich da oben neben dem Holzstück",
           "a07":"In sieben Stunden wird es soweit sein",
           "b01":"Was sind denn das für Tüten, die da unter dem Tisch stehen?",
           "b02":"Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter",
           "b03":"An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht",
           "b09":"Ich will das eben wegbringen und dann mit Karl was trinken gehen",
           "b10":"Die wird auf dem Platz sein, wo wir sie immer hinlegen"
        }

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def get_text(file):
    tmp_file='/home/sleek_eagle/research/security/project/results/test.wav'
    data=wav.read(file)
    track=data[1]
    #track_scaled=(track-np.min(track))/(np.max(track)-np.min(track))*255
    track_scaled=2/(np.max(track)-np.min(track))*track + (np.max(track)+np.min(track))/(np.min(track)-np.max(track))
    track_scaled*=32767
    track_scaled=track_scaled.astype('int16')
    data=wav.write(tmp_file,data[0],track_scaled)
    harvard = sr.AudioFile(tmp_file)
    with harvard as source:
        audio = r.record(source)
    text=r.recognize_google(audio,language='de')
    return text

def get_wer(file):
    idx=file.split('/')[-1][2:5]
    reference=text_dict[idx].lower()
    
    text=get_text(file).lower()
    
    er=wer(reference,text)
    return er

'''
do speech recognition and calaulate WER for each resutl
do this for all .wav files in the input directoty
'''
r = sr.Recognizer()
path='/home/sleek_eagle/research/security/project/results/12.0_0.14/'
files = [f for f in listdir(path) if isfile(join(path, f))]
wer_list=[]
for i,file in enumerate(files):
    er=get_wer(path+file)
    wer_list.append(er)
    print(i)

np.save('/home/sleek_eagle/research/security/project/results/wer/12.0_0.14',np.array(wer_list))
np.mean(np.load('/home/sleek_eagle/research/security/project/results/wer/12.0_0.14.npy'))


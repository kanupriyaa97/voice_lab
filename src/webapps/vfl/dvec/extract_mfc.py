'''
Description: This program simply extracts the
mfcc of the file given a wav path
'''

import os
import numpy as np
from scipy.io import wavfile
from mfcc import MFCC
import datetime
np.set_printoptions(threshold=np.nan)
import copy

def normalize(M):
    w_size = M.shape[0]
    new_array = []
    target = copy.deepcopy(M)
    mean = target.mean(0)
    std = target.std(0)
    new_array = (target-mean)/std
    new_array = np.array(new_array)
    return new_array

def extend(M, max_dim):
    new_array = np.copy(M)
    dim = new_array.shape[0]
    while_flag = False
    while(dim < max_dim):
        while_flag = True
        new_array = np.concatenate((new_array,M))
        dim = new_array.shape[0]
    if(while_flag):
        return np.array(new_array[:max_dim,:])
    return new_array

def extract_mel_spectra(wav_path,out_dir=None):
    
    mfcObj = MFCC(srate=8000,lowerf=20., upperf=3600., nfilt=63, nfft=1024)
    
    try:
        fs, data = wavfile.read(wav_path)
        assert fs==8000
        mbk = mfcObj.compute_logspec(data).astype('float16')
        ashape = mbk.shape
        index = np.argwhere(mbk.mean(axis=1)<-11.)
        mbk = np.delete(mbk, index, axis=0)
        bshape = mbk.shape
        mbk = np.reshape(mbk, (-1,63))
        mbk = mbk.astype(np.float64)
        mbk = normalize(mbk)
        mbk = extend(mbk, 500)
        
        mbk = mbk.astype(np.float32)
        return mbk

    except:
        print("Unable to extract mel-spectra from",wav_path)


'''
Created on Mon Apr 9 04:50:47 2018
Modified on November 30th 2018
@author: Shahan Ali Memon <samemon@cs.cmu.edu>  

Mentors: Rita Singh, Bhiksha Raj 
Copyright: Language Technologies Institute, CMU

Description: This program is used for calculating jitter
and shimmer from the voice

'''

import scipy.io.wavfile as wav


import os
import sys
#from vad import voiceVAD

def extract_jitter(attributes,values):
    for i in range(len(values)):
        a = attributes[i]
        v = values[i]
        if('jitterLocal_sma_amean' in a):
            return v

def extract_shimmer(attributes,values):
    for i in range(len(values)):
        a = attributes[i]
        v = values[i]
        if('shimmerLocal_sma_amean' in a):
            return v

def compute_jitter_shimmer(audio_path):
    root_name = audio_path.split("/")[-1]
    this_folder = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(this_folder+'/'+'temp_jitshim/'):
        os.makedirs(this_folder+'/'+'temp_jitshim/')

    newfile_path = this_folder+'/'+'temp_jitshim/'+root_name
    os.system("sox "+audio_path+" -t .wav -r 16000 -R -b 16 "\
                  "-c 1 "+newfile_path)
    
    
    
    #Now we extract the jitter and shimmer using IS10 paralinguistics
    #using openSmile

    feature = 'IS10_paraling_compat.arff'
    conf_file = 'IS10_paraling_compat.conf'
    
    lst = audio_path.split("/")
    primaryName = lst[-1].split(".wav")[0]
    outputFileName = this_folder+"/temp_jitshim/"+primaryName+"."+ feature
    
    os.system(this_folder+'/SMILExtract -C /Users/samemon/downloads/openSMILE-2.1.0/config/' + conf_file + ' ' + '-I ' + newfile_path + ' -O ' + outputFileName)
    all_lines = list(map(lambda l: l.rstrip(), open(outputFileName).readlines()))
    attributes = []
    values = all_lines[-1].split(",")
    for l in all_lines:
        if('@attribute' in l):
            attributes.append(l)
    if(len(values) != len(attributes)):
        return -1,-1
    
    jitter = extract_jitter(attributes,values)
    shimmer = extract_shimmer(attributes,values)
    os.remove(outputFileName)
    os.remove(newfile_path)
    return round(float(jitter),8), round(float(shimmer),8)

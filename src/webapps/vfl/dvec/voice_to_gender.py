'''
Created on Mon Apr 9 04:50:47 2018

@author: Shahan Ali Memon <samemon@cs.cmu.edu>  

Mentors: Rita Singh, Bhiksha Raj 
Copyright: Language Technologies Institute, CMU


Description: This program is for gender classification                                                                                     using voice. It takes in path to a wav file, and then                                                                                      return back 0/1 based on 
0 ==> if voice is of a male person                                                                                                         
1 ==> if voice is of a female person

'''

import scipy.io.wavfile as wav

'''
Description:
This function will be used to extract
the gender from a wav file using the
pre-trained gender classifier.

It requires:
a) path to wav file

Returns:
the gender of the wav file
'''

import os
from extract_mfc import extract_mel_spectra
import classify as dvec_extract
from sklearn.externals import joblib
import sys

this_folder = os.path.dirname(os.path.abspath(__file__))
gender_classifier_1 = this_folder+'/gender_dvec_both.pkl'
gender_classifier_2 = this_folder+'/gender_dvec.pkl'

def return_classifier_info():
    return gender_classifier

def extract_gender(audio_path):
    root_name = audio_path.split("/")[-1]
    
    if not os.path.exists(this_folder+'/temp_gender/'):
        os.makedirs(this_folder+'/temp_gender/')

    if not os.path.exists(this_folder+'/temp_wavs/'):
        os.makedirs(this_folder+'/temp_wavs/')

    newfile_path = this_folder+'/temp_gender/'+root_name
    os.system("sox "+audio_path+" -t .wav -r 8000 -R -b 16 "\
                  "-c 1 "+newfile_path) #trim 0.1 10.0")
    
    
    mbk = extract_mel_spectra(newfile_path)
    
    #Now we extract the d-vectors
    mbk_file = this_folder+"/temp_gender/"+root_name.split(".wav")[0]+".mbk"
    mbk.tofile(mbk_file)
    dvector = dvec_extract.extract_dvec(this_folder+"/temp_gender/")
    clf = joblib.load(gender_classifier_2)
    gender = clf.predict(dvector)[0]
    
    #Removing the two files it creates
    os.remove(mbk_file)
    os.remove(newfile_path)
    
    return ("f" if gender == 1 else "m")
    
    
if __name__ == "__main__":
    argv = sys.argv[1:]
    if(len(argv) != 1):
        print("Usage: python voice_to_gender.py <path to wav>")
        sys.exit()
    else:
        gen = extract_gender(argv[0])
        s = "Female" if gen == "f" else "Male"
        print("Gender == ",s)

'''
Created on Mon Apr 9 04:50:47 2018
Modified on Nov 30th 2018
@author: Shahan Ali Memon <samemon@cs.cmu.edu>  

Mentors: Rita Singh, Bhiksha Raj 
Copyright: Language Technologies Institute, CMU


Description: This program is for gender classification                                                                                     using voice. It takes in path to a wav file, and then                                                                                      return back 0/1 based on 
0 ==> if voice is of a male person                                                                                                         
1 ==> if voice is of a female person

'''

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
from sklearn.externals import joblib
import sys

this_folder = os.path.dirname(os.path.abspath(__file__))
gender_classifier_1 = this_folder+'/gender_dvec_both.pkl'
gender_classifier_2 = this_folder+'/gender_dvec.pkl'

def return_classifier_info():
    return gender_classifier

def extract_gender(dvector):
    clf = joblib.load(gender_classifier_2)
    gender = clf.predict(dvector)[0]
    
    return ("f" if gender == 1 else "m")


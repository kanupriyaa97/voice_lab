import time
import os
from scipy.io import loadmat, savemat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import sys, getopt, traceback
import math
import time
from torch.utils.data import Dataset, DataLoader
from ResNet import ResNet
global NET_IN_FILE

this_folder = os.path.dirname(os.path.abspath(__file__))
NET_IN_FILE = this_folder+"/SpkNet-CNN-ResNet-f128-x-entropy_Iter_70"

class MelBankDataset(Dataset):
    def __init__(self, file_list, target_list, size_list, crop_size, dtype):
        self.file_list = file_list
        self.target_list = target_list
        self.size_list = size_list
        self.crop_size = crop_size
        self.dtype = dtype
        self.unit = np.dtype(dtype).itemsize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        with open(self.file_list[index], 'r') as f:
            mbk = np.frombuffer(f.read(),
                                dtype=self.dtype).reshape((-1,63))[np.newaxis, ...]
        mbk, label = mbk.astype('float32'), self.target_list[index]

        return mbk, label

'''
Requires: The path to the data directory
Returns: 
wavfile_list: Which is the list of all the files
Id_list: Which is the list of all the Ids for each file

Note: We won't need Id list for now, but maybe later 
'''

def parse_data(data_dir):
    #Initializing the wavlist and Id list
    wavfile_list = []
    ID_list = []
    #Going through each path in the data directory
    for root, directories, filenames in os.walk(data_dir):
        #Goign through each file
        for filename in filenames:
            #Whichever file ends with .mbk
            if filename.endswith('.mbk'):
                #We create a wav list and a id list for each file
                wav_file = os.path.join(root, filename)
                wavfile_list.append(wav_file)
                ID_list.append(filename)
    return wavfile_list, ID_list

def read_labels():
    labels={}
    l_file = open('labels.lst')
    l_files_lines=l_file.readlines()
    for ins in l_files_lines:
        ins = ins.split(' ')
        u_id = ins[1]
        l = ins[2]
        if u_id not in labels.keys():
            labels[u_id] = int(l)-1 # labels need to be zero indexed!
    return labels

def classify_batch():
    class_n = 3788
   # print 'class_n should be 3788 but it is', class_n
    model = ResNet(1, 4, 16, 64, 256, class_n)
    net = nn.DataParallel(model, device_ids=[0])
    ff = net
    ff.load_state_dict(torch.load(NET_IN_FILE,map_location=lambda storage, loc: storage))
    ff.eval()
    feat_M = []
    all_ids = []
    iteration = 0
    dict_M = {}
    for i, (data, label) in enumerate(dataloader):
        print i, data.shape
        inputs = Variable(data)
	(feature) = ff(inputs)
        o = feature.cpu().data.numpy()
	feat_M.extend(w for w in o)
        all_ids.extend(l for l in label)

    feat_M_np = np.array(feat_M)
    return feat_M_np


def extract_dvec(data_dir):
    global dataloader
    dtype = np.float32
    batchsize = 1
    crop_size = [6000, 63]
    size_list = []
    mbkfile_list, label_list = parse_data(data_dir)
    testset = MelBankDataset(mbkfile_list,
                             label_list, size_list,
                             crop_size, dtype)
    dataloader = DataLoader(testset, batch_size=batchsize,
                            shuffle=True, num_workers=1,
                            drop_last=False)
    return classify_batch()

import glob, sys

import os.path
import os
import time 
import numpy as np
from tqdm import tqdm
import scipy.stats
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import tensorflow as tf
from tensorflow import keras
import utils
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


def get_test_results(BIN_DIR, data, test_list, modelfile, results_file1, results_file2):
    print('testing...', modelfile)
    model = load_model(modelfile)
    model.summary()
    MOS_Predict=np.zeros([len(test_list),])
    MOS_true   =np.zeros([len(test_list),])
    df = pd.DataFrame(columns=['audio', 'true_mos','predict_mos','system_ID','speaker_ID'])

    for i in tqdm(range(len(test_list))):

        filepath=test_list[i].split(',')
        speakerid = filepath[0]
        sysid = filepath[1]
        filename=filepath[2].split('.')[0]
        mos=float(filepath[3])

        _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
        _mag = _feat['mag_sgram']    
        [Average_score, Frame_score]=model.predict(_mag, verbose=0, batch_size=1)
        MOS_Predict[i]=Average_score
        MOS_true[i]   =mos

        df = df.append({'audio': filename, 
                        'true_mos': MOS_true[i], 
                        'predict_mos': MOS_Predict[i], 
                        'system_ID': sysid, 
                        'speaker_ID': speakerid}, 
                       ignore_index=True)
        
    df.to_pickle(results_file1)
    df.to_csv(results_file2, index=False)
    return



##################################################
model_name = "LA_model"


if model_name == "LA_model":
    folder = './LA_best_model/'
    model = folder+"/mosnet.h5"
    feats = "orig"
    
##################################################


data = "LA"
flist = glob.glob("data/"+feats+"/*.h5")
fnames = [f.split("/")[-1].split(".")[0] for f in flist]
spks = [f.split("_")[0] for f in fnames]
testlist = [spk+",original,"+fname+".wav,0" for spk,fname in zip(spks, fnames)]
print(testlist)
            
bin_dir = "data/"+feats
results_file1 = "data/results.pkl"
results_file2 = "data/results.txt"
logname = "data/test_list.txt"

# apply the trained model and save results
get_test_results(bin_dir, data, testlist, model, results_file1, results_file2)




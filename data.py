import os
import  glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from tqdm import tqdm





def init_data():
    base_url = './inputs/ICBHI_final_database'
    label_url = './inputs/ICBHI_Challenge_diagnosis.txt'
    trn_tst_url = './inputs/ICBHI_challenge_train_test.txt'

    df1 = pd.read_csv(trn_tst_url,sep="\t", header=None,names=['labelid','class'])
    df2 = pd.read_csv(label_url,sep="\t", header=None, names = ['labelid','diagnosis'])

    df2['Class'] = df2['diagnosis']
    for i in range(len(df2)):
        if df2['diagnosis'][i] == 'Healthy':
            df2.iloc[i,-1] = 1
        else:
            df2.iloc[i,-1] = 0

    df1['output'] = df1['class']
    for i in range(len(df1)):
        x = df1['labelid'][i]
        t = x[0:3]
        cls = df2.loc[df2['labelid'] == int(t),'Class'].values[0]
        df1['output'][i] = cls


    # print(df1.head())
    # print(df2.head())

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    files = tqdm(glob.glob(os.path.join(base_url, '*.wav')))
    for filename in files:
        try:
            data,sample_rate = librosa.load(filename)
            mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)    
            lbl = filename[-26:-4]
            #print(df1.loc[df1['labelid'] == str(lbl),'class'].values[0])
            if df1.loc[df1['labelid'] == str(lbl),'class'].values[0] == 'test':
                X_test.append(mfccs_scaled_features)
                y_test.append(df1.loc[df1['labelid'] == str(lbl),'output'].values[0])
            else:
                X_train.append(mfccs_scaled_features)
                y_train.append(df1.loc[df1['labelid'] == str(lbl),'output'].values[0])
        except:
            print("IndexError!")  

    X_test = np.asarray(X_test)
    print("X-test shape: {}".format(X_test.shape))

    y_test = np.asarray(y_test)
    print("Y-test shape: {}".format(y_test.shape))

    X_train = np.asarray(X_train)
    print("X-train shape:{}".format(X_train.shape))

    y_train = np.asarray(y_train)
    print("Y-train shape:{}".format(y_train.shape))
    return X_train, X_test, y_train, y_test
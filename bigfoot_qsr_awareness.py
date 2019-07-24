# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:46:52 2019

@author: payam.bagheri
"""
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from difflib import SequenceMatcher
import time
import re

dir_path = path.dirname(path.dirname(path.abspath(__file__)))
dic = pd.read_csv(dir_path + '/0_input_data/Bigfoot Restaurant Awareness Dictionary.csv', low_memory=False)
unspec = pd.read_excel(dir_path + '/0_input_data/unspecified.xlsx')
unspec = list(unspec.Unspecified)

data = pd.read_excel(dir_path + '/0_input_data/unaided_qsr_awareness.xlsx')
data = data.head(20)

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


raw_data_cols = ['unaided_qsr_awareness_r0', 'unaided_qsr_awareness_r1',
       'unaided_qsr_awareness_r2', 'unaided_qsr_awareness_r3',
       'unaided_qsr_awareness_r4']

my_codes = pd.DataFrame(index = range(data.shape[0]), columns = raw_data_cols)
maxes = pd.DataFrame(index = range(data.shape[0]), columns = ['max values'])

start = time.time()
for col in tqdm(raw_data_cols):
    didbreak = False
    for ind in data[col].index:
        #print(data[col].loc[ind])
        meansims = pd.DataFrame(index = dic.index, columns = ['maxs', 'code'])
        word = data[col].loc[ind]
        didbreak = False
        if word is not np.nan:
            for i in dic.index:
                word = re.sub('[^0-9a-zA-Z]+', '', str(word))
                sims = [similar(word,re.sub('[^0-9a-zA-Z]+', '', x)) for x in list(dic.loc[i][dic.loc[i].notnull()])]
                meansims['maxs'][i] = max(sims)
                meansims['code'][i] = i-1
                if max(sims) == 1:
                     my_codes[col].loc[ind] = i-1
                     didbreak = True
                     break               
            if didbreak == False:
                maxims = meansims['maxs'].astype('float64')
                idmax = maxims.idxmax(axis = 0)
                maxval = maxims.loc[idmax]
                maxes['max values'].loc[ind] = maxims.loc[idmax]
                #print('maxval', maxval, word)
                if maxval > 0.9:
                    my_codes[col].loc[ind] = meansims['code'].loc[idmax]
                else:
                    word = re.sub('[^0-9a-zA-Z]+', '', str(word))
                    sims = [similar(word,re.sub('[^0-9a-zA-Z]+', '', x)) for x in unspec]
                    maxunspec = max(sims)
                    if maxunspec > 0.85:
                        my_codes[col].loc[ind] = 153
                    else:
                        my_codes[col].loc[ind] = 1059
        else:
            my_codes[col].loc[ind] = np.nan
end = time.time()
print('total time is :', (end - start)/3600)

my_codes[(my_codes.isnull().sum(axis=1)>0) & (my_codes.isnull().sum(axis=1)<5)] = my_codes[(my_codes.isnull().sum(axis=1)>0) & (my_codes.isnull().sum(axis=1)<5)].replace(np.nan,0)

my_codes.to_csv(dir_path + '/0_output/bigfoot_qsr_my_codes.csv')
maxes.to_csv(dir_path + '/0_output/bigfoot_qsr_maxes.csv')

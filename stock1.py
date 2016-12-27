#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:19:58 2016

@author: frank
"""
stock_nb = '002624'
import neurolab as nl
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd

def get_nomorlize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return (x, x_min, x_max)


def get_input_output(df):
    df_y = df[['close']]
    df_x = df[['open', 'high', 'low', 'volume']]
    input_num = len(df_x.columns)
    lasttrade_name = df_x.iloc[0,:].name
    firsttrade_name = df_x.iloc[-1,:].name
    df_input = df_x.drop([lasttrade_name], axis=0)
    df_y = df_y.drop([firsttrade_name], axis=0)
    np_input = np.array(df_input.values)
    np_output = np.array(df_y.values)
    return (np_input, np_output, input_num)


def get_unnomorlize(x, x_min, x_max):
    x = x * (x_max - x_min) + x_min
    return x


#create train samples
df = ts.get_hist_data(stock_nb,start= '2015-11-01', end='2016-12-01')
(np_input, np_output, input_num) = get_input_output(df)
tar = get_nomorlize(np_output)[0]
for i in range(0,input_num):    
    np_input[:,i] = get_nomorlize(np_input[:,i])[0]

netminmax = zip(np.min(np_input, 0), np.max(np_input, 0))
netminmax = [list(e) for e in netminmax]
net = nl.net.newff(netminmax, [input_num,1])
net.trainf = nl.train.train_gdx
np_input= np.fliplr(np_input)
tar = np.flipud(tar)
error = net.train(np_input, tar, epochs=10000, show=100, goal = 0.001)
out = net.sim(np_input)

df_val = ts.get_hist_data(stock_nb,start= '2016-12-02')
(val_input, val_output, val_num) = get_input_output(df_val)
(val_tar, val_min, val_max) = get_nomorlize(val_output)
val_index = df_val.index[:-1]
val_index = np.flipud(val_index)
for i in range(0,val_num):    
    val_input[:,i] = get_nomorlize(val_input[:,i])[0]
np.fliplr(val_input)
val_out =  net.sim(val_input)
val_out = get_unnomorlize(val_out, val_min, val_max)

df_result = pd.DataFrame(val_out, index=val_index, columns = \
                         ['Predicted Price'])
np_real_price = df_val['close'].values[:-1]
np_real_price = np.flipud(np_real_price)
df_result['Real Price'] = np_real_price

plt.subplot(211)
plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('error (default SSE)')
print df_result
csv_filename = stock_nb + '.csv'
df_result.to_csv(csv_filename)
plt.figure()
df_result.plot()
plt.legend(loc='best')
fig_name = stock_nb + '.pdf'
plt.savefig(fig_name, format = 'pdf')
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 08:40:15 2019

@author: kanno
"""
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
from sklearn.model_selection import train_test_split

n_in = 20
#設定を入力
dna = [3, 60, 333, 0.08, 9, 4, 3]

n_hiddens = list()
for i in range(dna[0]):
    n_hiddens.append(dna[1])
n_centors = dna[0]

p_keep = dna[3]

epochs = dna[2]

maxlen= dna[4]

time_bias= dna[5]

layer_int = dna[6]

if layer_int==1:
    layer = 'SimpleRNN'
elif layer_int==2:
    layer = 'LSTM'
elif layer_int==3:
    layer = 'GRU'
    
isEs = False
BiDir = False
batch_size = 1000
ranseed=12345

#データ名
dataname = 'linear_data_FIR8_comAngle&AveStd_coor_s1_ntd2_26'
#モデル番号
model_no = '139'

#モデルを読み込み
filename = dataname+'_'+str(model_no)
model = tf.keras.models.load_model(filename+'.h5')
model.summary();

#ファイル読み込み
csv_input = pd.read_csv(filepath_or_buffer= dataname+".csv", encoding="ms932", sep=",")
array = csv_input.values

#入出力値を読み取り
X=array[:,1:n_in+1].astype(np.float32)
Y=array[:,n_in+1].astype(np.int)

#タイムスタンプを読み取り
TIME=array[:,0]

leng = len(Y)
data = []
target = []

for i in range(maxlen, leng):
	#入力データを参照秒数ごとにまとめる
	data.append(X[i-maxlen+1:i+1,:])
	#出力データをN秒ごとの作業に変換
	target.append(Y[i-time_bias])
    
#入出力データのshapeの調整
X = np.array(data).reshape(len(data), maxlen,n_in)
Y = np.array(target)

#タイムスタンプを入出力データと同期
TIME=TIME[maxlen-time_bias:leng-time_bias]

#学習データとテストデータの分割
x_train, x_test, y_train0, y_test0,time_train,time_test = train_test_split(X, Y,TIME, train_size=0.85,shuffle=False)

classes = model.predict_classes(x_test, batch_size=1)
prob = model.predict_proba(x_test, batch_size=1)

#結果を出力
im = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
ip = [0,0,0,0]
it = [0,0,0,0]
f = [0,0,0,0]

ia = [0,0,0,0]
ib = [0,0,0,0]

j = 0
for i in range(4):
    for j in range(y_test0.size):
        if y_test0[j]==i+1:
            it[i] += 1
                       
            if classes[j] == 1:
                im[i][0] += 1
            if classes[j] == 2:
                im[i][1] += 1
            if classes[j] == 3:
                im[i][2] += 1
            if classes[j] == 4:
                im[i][3] += 1
        else:
            pass
    
for i in range(4):        
    for k in range(y_test0.size):
        if classes[k]==i+1:
            ip[i]+=1
        else:
            pass

#再現率を導出        
for i in range(4):
    if it[i]==0:
        ia[i] = 0
    else:
        ia[i] = im[i][i]/it[i]

#適合率を導出    
for i in range(4):
    if ip[i]==0:
        ib[i] = 0
    else:
        ib[i] = im[i][i]/ip[i]

#F値を導出
for i in range(4):
    if ia[i]+ib[i]==0:
        f[i] = 0
    else:
        f[i] = 2*ia[i]*ib[i]/(ia[i]+ib[i])

it_sum = sum(it)
ip_sum = sum(ip)
ii = im[0][0]+im[1][1]+im[2][2]+im[3][3]#+i5

finname='GA_'+dataname+'_model_'+str(model_no)+'.csv'
fin = open(finname,'w')
fin.write('中間層,ドロップアウト率,earlystopping,参照数,ずらす時間,レイヤー,試行回数,bi-directional\n')
fin.write(str(n_hiddens)+','+str(p_keep)+','+str(isEs)+','+str(maxlen)+','+str(time_bias)+','+str(layer)+','+str(epochs)+','+str(BiDir)+'\n')
fin.write('\n')
fin.write('作業,溶接,移動,休憩,付帯,全体\n')
fin.write('作業時間(正解)(s),'+str(it[0])+','+str(it[1])+','+str(it[2])+','+str(it[3])+','+str(it_sum)+'\n')#+str(i5t)+','+str(it)+'\n')
fin.write('作業時間(予測)(s),'+str(ip[0])+','+str(ip[1])+','+str(ip[2])+','+str(ip[3])+','+str(ip_sum)+'\n')
fin.write('一致時間(s),'+str(im[0][0])+','+str(im[1][1])+','+str(im[2][2])+','+str(im[3][3])+','+str(ii)+'\n')
fin.write('再現率(%),'+str(ia[0]*100)+','+str(ia[1]*100)+','+str(ia[2]*100)+','+str(ia[3]*100)+','+str(np.float64(100*ii/it_sum))+'\n')
fin.write('適合率(%),'+str(ib[0]*100)+','+str(ib[1]*100)+','+str(ib[2]*100)+','+str(ib[3]*100)+','+str(np.float64(100*ii/ip_sum))+'\n')
fin.write('F値,'+str(f[0]*100)+','+str(f[1]*100)+','+str(f[2]*100)+','+str(f[3]*100)+','+str(np.float64(100*ii/ip_sum))+'\n')
fin.write('\n')
fin.write('比較,predict1,predict2,predict3,predict4,\n')

fin.write('test1,'+str(im[0][0]/ip[0])+','+str(im[0][1]/ip[1])+','+str(im[0][2]/ip[2])+','+str(im[0][3]/ip[3])+'\n')
fin.write('test2,'+str(im[1][0]/ip[0])+','+str(im[1][1]/ip[1])+','+str(im[1][2]/ip[2])+','+str(im[1][3]/ip[3])+'\n')
fin.write('test3,'+str(im[2][0]/ip[0])+','+str(im[2][1]/ip[1])+','+str(im[2][2]/ip[2])+','+str(im[2][3]/ip[3])+'\n')
fin.write('test4,'+str(im[3][0]/ip[0])+','+str(im[3][1]/ip[1])+','+str(im[3][2]/ip[2])+','+str(im[3][3]/ip[3])+'\n')

#重みのテンソルを出力
#for i in range(n_out):
#    for j in range(n_in):
#        fin.write(str(weight[i][j])+',')
#    fin.write('\n')
#fin.write('\n')

#正解データと予測の比較
for l in range(classes.size):
    fin.write(str(y_test0[l])+','+str(classes[l])+'\n')
fin.close()
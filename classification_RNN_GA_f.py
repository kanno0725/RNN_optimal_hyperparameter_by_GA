# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:50:38 2019

@author: kanno
"""
import numpy as np
import random as rn
import tensorflow as tf
import csv
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.layers.core import Dropout
#from keras.optimizers import Rmsprop
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import load_model 

from tensorflow.python.keras import initializers 
from tensorflow.python.keras.layers.recurrent import LSTM, SimpleRNN, GRU
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import CSVLogger
from sklearn import datasets
from sklearn.model_selection import train_test_split
from datetime import datetime

from tensorflow.python.keras import backend

def neuralnet(no_model,dnafull,dna0,dna1,dna2,dna3,dna4,dna5,dna6):
    
    """
    dna_temp[0] hid_layer_num INT 1~5
    dna_temp[1] hid_layer_node INT 16~128
    dna_temp[2] epoch INT 100~500
    dna_temp[3] dropout FLOAT 0.00~0.20
    dna_temp[4] maxlen INT　9~19
    dna_temp[5] time_bias INT 1~9
    dna_temp[6] layer INT 1~3
    """
    
    """
    パラメーター設定
    """
    #入力層次元
    n_in = 20
    #中間層次元、層数
    n_hiddens = list()
    for i in range(dna0):
        n_hiddens.append(dna1)
    n_centors = dna0
    #出力層次元、層数
    n_out = 5
    #活性化関数
    activation = 'relu'
    #ドロップアウト率
    p_keep = dna3
    #計算回数
    epochs = dna2
    #EarlyStoppingするか
    isEs= False
    #EarlyStoppingをするまでの回数
    es_patience= 60
    #ミニバッチ処理のサイズ
    batch_size = 1000
    #最適化アルゴリズム
    opt='rmsprop'
    #学習率(本プログラムでは未使用でデフォルト値を使用）
#    learning_rate=0.001
    #Adamのパラメータ(最適化アルゴリズムがAdamの時のみ使用・本プログラムでは未使用)
#    beta_1=0.9
#    beta_2=0.999
    #reccrentの参照数
    maxlen= dna4
    #Yを何秒ずらすか（=0だと過去maxlen秒参照、=maxlen/2だと前後maxlen/2秒参照、=maxlenだと未来maxlen秒参照になる)
    time_bias= dna5
    
    #RNNの種類(SimpleRNN,LSTM,GRU)
    layer_int = dna6
    
    #双方向性を使用するか
    BiDir= False
    
    #RNNの偶数層を逆向きにするか
    back= False
    
    #乱数の固定シード
#    ranseed= 12345
    
#    weight1 = 1
#    weight2 = 1
#    
    print('No_%d' % no_model)
    print(dna0,dna1,dna2,dna3,dna4,dna5,dna6)
    
    #乱数固定
    
    import os
    os.environ['PYTHONHASHSEED']='0'
#    np.random.seed(ranseed)
#    rn.seed(ranseed)
    
    #スレッド数等を１に固定(再現性に必要)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    
    from tensorflow.python.keras import backend as K
#    tf.compat.v1.set_random_seed()
    
     
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
    K.set_session(sess)
    
    
    #重み初期化
    init=initializers.TruncatedNormal()
    
    #ファイルの名前
    name = 'linear_data_FIR8_comAngle&AveStd_coor_s1_ntd2_26'
#    number = '-2'
    #ファイル読み込み
    csv_input = pd.read_csv(filepath_or_buffer= name+".csv", encoding="ms932", sep=",")
    array = csv_input.values
    
    #仮の入出力値を読み取り
    
    X=array[:,1:n_in+1].astype(np.float32)
    
    Y=array[:,n_in+1].astype(np.int)
    
    
    #タイムスタンプを読み取り
    TIME=array[:,0]
    
    leng = len(Y)
    data = []
    target = []
    
    i = 0

    for i in range(maxlen, leng):
    	#入力データを参照秒数ごとにまとめる
    	data.append(X[i-maxlen+1:i+1,:])
    	#出力データをN秒ごとの作業に変換
    	target.append(Y[i-time_bias])
    #入出力データのshapeの調整
    X = np.array(data).reshape(len(data), maxlen, n_in)
    Y = np.array(target)
    
    #タイムスタンプを入出力データと同期
    TIME=TIME[maxlen-time_bias:leng-time_bias]
    
    #学習データとテストデータの分割
    x_train, x_test, y_train0, y_test0,time_train,time_test = train_test_split(X, Y,TIME, train_size=0.85,shuffle=False)
    
    #学習データをtrainとvalidationに分割
    x_train, x_validation, y_train0, y_validation0 = train_test_split(x_train, y_train0,train_size=0.9,shuffle=False)
        
    #yを1ofKデータに変換(train,val,test)
    ntr=y_train0.size
    y_train=np.zeros(n_out*ntr).reshape(ntr,n_out).astype(np.float32)
    for i in range(ntr):
    	y_train[i,y_train0[i]]=1.0
    
    nte=y_test0.size
    y_test=np.zeros(n_out*nte).reshape(nte,n_out).astype(np.float32)
    for i in range(nte):
    	y_test[i,y_test0[i]]=1.0
    
    
    y_validation=np.eye(n_out)[(y_validation0.reshape(y_validation0.size))]
        
#    nrow=y_test0.size
   
    # モデル設定
    
    model = Sequential()
        
    for i in range(n_centors):
    	if(i==n_centors-1):
    		retSeq=False
    	else:
    		retSeq=True
    	if(i%2==1 and back):
    		gBack=True
    	else:
    		gBack=False
    	if(i==0):
    		in_dir=n_in
    	else:
    		in_dir=n_hiddens[i-1]
        
    	if (layer_int==1):
    		if(BiDir):
    			model.add(Bidirectional(SimpleRNN(n_hiddens[i],activation=activation,kernel_initializer=init,recurrent_initializer=init,dropout=p_keep,recurrent_dropout=p_keep, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) )))
    		else:
    #			model.add(SimpleRNN(n_hiddens[i],activation=activation,kernel_initializer=init,recurrent_initializer=init, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) ))
    			model.add(SimpleRNN(n_hiddens[i],activation=activation,kernel_initializer=init,recurrent_initializer=init,dropout=p_keep,recurrent_dropout=p_keep, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) ))
    
    	elif(layer_int==2):
    		if(BiDir):
    			model.add(Bidirectional(LSTM(n_hiddens[0],activation=activation,kernel_initializer=init,recurrent_initializer=init,dropout=p_keep,recurrent_dropout=p_keep, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) )))
    		else:
    			model.add(LSTM(n_hiddens[0],activation=activation,kernel_initializer=init,recurrent_initializer=init,dropout=p_keep,recurrent_dropout=p_keep, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) ))
    	
    	elif(layer_int==3):
    		if(BiDir):
    			model.add(Bidirectional(GRU(n_hiddens[0],activation=activation,kernel_initializer=init,recurrent_initializer=init,dropout=p_keep,recurrent_dropout=p_keep, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) )))
    		else:
    			model.add(GRU(n_hiddens[0],activation=activation,kernel_initializer=init,recurrent_initializer=init,dropout=p_keep,recurrent_dropout=p_keep, return_sequences=retSeq,go_backwards=gBack,  input_shape=(maxlen, in_dir) ))	
    
    model.add(Dense(n_out,kernel_initializer=init))
    model.add(Activation('softmax'))
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    early_stopping =EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)
    
#    now = datetime.now().strftime('%Y%m%d%H%M')
#    flog = name+number+'.log1.csv'
#    
#    csv_logger=CSVLogger(flog)
    
    if (isEs):
    	caBacks=[early_stopping]#,csv_logger]
    
    else:
    	caBacks=[]#csv_logger]
    
    #モデル学習
    
#    start = time.time()
    
    model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_validation,y_validation),callbacks=caBacks)
    #hist = model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,callbacks=caBacks)
    #,callbacks=[early_stopping]
    
#    slapsed_time=time.time() - start
#    
#    
#    val_acc = hist.history['val_acc']
#    acc = hist.history['acc']
#    val_loss = hist.history['val_loss']
#    loss = hist.history['loss']
#

#now = datetime.now().strftime('%Y%m%d%H%M')
#
#plt.rc('font',family='serif')
#fig = plt.figure()
#plt.plot(range(len(loss)), loss, label='loss', color='r')
#plt.plot(range(len(val_loss)), val_loss, label='val_loss', color='b')
#plt.xlabel('epochs')
#plt.legend()
#plt.show()
#plt.savefig(name+number+'.loss.png')
#
##plt.rc('font',family='serif')
##fig = plt.figure()
##plt.plot(range(len(val_acc)), val_acc, label='acc', color='b')
##plt.xlabel('epochs')
##plt.show()
##plt.savefig(name+number+'.val_acc.png')
       
    classes = model.predict_classes(x_test, batch_size=1)
#prob = model.predict_proba(x_test, batch_size=1)

#重みの出力
#L1 = model.get_weights()
#W1 = np.dot(L1[0],L1[1])+L1[2]
#W2 = np.dot(W1,L1[3])
#W3 = np.dot(W2,L1[4])
#W4 = W3+L1[5]
#weight1 = np.dot(W4,L1[6])+L1[7]
#weight = weight1.transpose()

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
    
#    it_sum = sum(it)
#    ip_sum = sum(ip)
#    ii = im[0][0]+im[1][1]+im[2][2]+im[3][3]#+i5
    
    if_ave = sum(f)/4
    
    model.save(name+'_'+str(no_model)+".h5")
#    model.save("kanno_"+str(no_model)+".model")
   
# =============================================================================
    backend.clear_session()
# =============================================================================
    
    
    return if_ave


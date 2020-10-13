# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:47:52 2019

@author: kanno
"""

#GA

import random as rn
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import pandas as pd
import classification_RNN_GA_f as RNN

"""
1. INITIAL POPULATION
2. FITNESS FUCTION
3. SELECTION
4. CROSSOVER
5. MUTATION
"""

"""
パラメータの設定
"""

N = 15 #遺伝子の個数,3以上 #15
G = 15 #初期を含まない世代数 #15

#エリートの個数,N以下
n = 2

"""
1. INITIAL POPULATION
"""
#test
#dna_temp = [[1,2],[1,2],[2,3],[0],[3,4],[1,3],[1,2]]
dna_temp = [[1,5],[16,128],[100,500],[0],[4,9],[1,4],[1,3]]

"""
dna_temp[0] hid_layer_num INT 1~5
dna_temp[1] hid_layer_node INT 16~128
dna_temp[2] epoch INT 100~500
dna_temp[3] dropout FLOAT 0.00~0.20
dna_temp[4] maxlen INT 9~19
dna_temp[5] time_bias　INT 1~9
dna_temp[6] layer　INT 1~3
"""

kotak = []

def initial(dna_temp):
    initial_ind = list()
    for i in range(N):
        ind0 = list()
        ind0.append(rn.randint(dna_temp[0][0],dna_temp[0][1])) #整数の乱数
        ind0.append(rn.randint(dna_temp[1][0],dna_temp[1][1]))
        ind0.append(rn.randint(dna_temp[2][0],dna_temp[2][1]))
        
        ind0.append(round(dna_temp[3][0] + rn.randrange(0,20,1)/100,2))
        
        ind0.append(rn.randint(dna_temp[4][0],dna_temp[4][1]))
        ind0.append(rn.randint(dna_temp[5][0],dna_temp[5][1]))
        ind0.append(rn.randint(dna_temp[6][0],dna_temp[6][1]))
       
        
        initial_ind.append(ind0)
    return initial_ind

initial_ind = initial(dna_temp) #最初の乱数を設定

"""
2. FITNESS FUCTION
"""
pon = {}
pon['ind_0'] = []
pon['fit_0'] = []

initial_fit = list()        
no_model = 1      
def fitten(no_model,initial_ind,initial_fit):
    i = 0
    for i in range(N):
        fit = RNN.neuralnet(no_model,i,initial_ind[i][0],initial_ind[i][1],initial_ind[i][2],initial_ind[i][3],initial_ind[i][4],initial_ind[i][5],initial_ind[i][6])
#    for i in initial_ind:
#        fit = RNN.neuralnet(no_model,i,i[0],i[1],i[2],i[3],i[4],i[5],i[6])
        initial_fit.append(fit)
        no_model += 1
        
    return initial_fit, no_model

initial_fit, no_model = fitten(no_model,initial_ind,initial_fit)

pon['ind_0'].extend(initial_ind)
pon['fit_0'].extend(initial_fit)
kotak.append(pon)

"""
3. SELECTION-A
"""
def selection_a(inital_ind,initial_fit):
    elite_ind = list()
    elite_fit = list()
    elite_ind.append(initial_ind[initial_fit.index(max(initial_fit))])
    elite_fit.append(max(initial_fit))
    return elite_ind, elite_fit
elite_ind, elite_fit = selection_a(initial_ind,initial_fit)

#==============================================================================
for genx in tqdm(range(1,1+G)):
            
    """
    4. CROSSOVER
    """
    
    gen0 = list()
    gen0_fit = list()
    initial_ind2 = initial_ind.copy()
    initial_fit2 = initial_fit.copy()
   
    for i in range(n):
        c = initial_fit2.index(max(initial_fit2))
        a = initial_ind2.pop(c)
        b = initial_fit2.pop(c)
        gen0.append(a)
        gen0_fit.append(b) #maxのfit,index
    
    border = len(dna_temp)//2
    kross1 = rn.randint(1,border)
    kross2 = rn.randint(border+1,len(dna_temp))
    
#    kross = rn.randint(1,len(dna_temp))
    xmen = 0.05
    
    gen1 = list()
    gen1_fit = list()
    
    index = [0,2]
    
    #shuffle#
    shuffle(initial_ind)
    
    for i in range(N//2):    
#    for i in range(int((N-n)/2)):
        x = initial_ind[index[0]:index[1]].copy()
        y1 = list()
        y2 = list()
        y1 = x[0][:kross1].copy()
        y1.extend(x[1][kross1:kross2])
        y1.extend(x[0][kross2:])
        y2 = x[1][:kross1].copy()
        y2.extend(x[0][kross1:kross2])
        y2.extend(x[0][kross2:])
        
        """
        5. MUTATION
        """
        if rn.random() <= xmen:
            if rn.random() <= 0.5:
                y1[0] = rn.randint(dna_temp[0][0],dna_temp[0][1])
                y2[0] = rn.randint(dna_temp[0][0],dna_temp[0][1])
                y1[1] = rn.randint(dna_temp[1][0],dna_temp[1][1])
                y2[1] = rn.randint(dna_temp[1][0],dna_temp[1][1])
                y1[2] = rn.randint(dna_temp[2][0],dna_temp[2][1])
                y2[2] = rn.randint(dna_temp[2][0],dna_temp[2][1])
            else:
                y1[3] = round(dna_temp[3][0] + rn.randrange(0,20,1)/100,2)
                y2[3] = round(dna_temp[3][0] + rn.randrange(0,20,1)/100,2)
                y1[4] = rn.randint(dna_temp[4][0],dna_temp[4][1])
                y2[4] = rn.randint(dna_temp[4][0],dna_temp[4][1])
                y1[5] = rn.randint(dna_temp[5][0],dna_temp[5][1])
                y2[5] = rn.randint(dna_temp[5][0],dna_temp[5][1])
                y1[6] = rn.randint(dna_temp[6][0],dna_temp[6][1])
                y2[6] = rn.randint(dna_temp[6][0],dna_temp[6][1])
            
        gen1.append(y1)
        gen1.append(y2)
        index[0] += 2
        index[1] += 2
    
    if N%2==0:
        pass
    else:
        x = initial_ind[N-2:N].copy()
        y1 = list()
        y2 = list()
        y1 = x[0][:kross1].copy()
        y1.extend(x[1][kross1:kross2])
        y1.extend(x[0][kross2:])
        y2 = x[1][:kross1].copy()
        y2.extend(x[0][kross1:kross2])
        y2.extend(x[0][kross2:])
        gen1.append(y2)
    #==============================================================================
            
    """
    2. FITNESS FUCTION
    """     
    pon = {}
    pon['ind_'+str(genx)] = []
    pon['fit_'+str(genx)] = []
    
    "NEW..."
    
    
    
    gen1_fit,no_model = fitten(no_model,gen1,gen1_fit)
    
    gen1.extend(gen0)
    gen1_fit.extend(gen0_fit)
    
    "NEW..."
    
    pon['ind_'+str(genx)].extend(gen1)
    pon['fit_'+str(genx)].extend(gen1_fit)
    kotak.append(pon)
    
    """
    3. SELECTION-B
    """    
    def selection_b(initial_ind,initial_fit,elite_ind,elite_fit):
        elite_ind.append(gen1[gen1_fit.index(max(gen1_fit))])
        elite_fit.append(max(gen1_fit))
        return elite_ind, elite_fit
    elite_ind, elite_fit = selection_b(initial_ind,initial_fit,elite_ind,elite_fit)
    ####↑
    initial_ind = gen1
    initial_fit = gen1_fit
# =============================================================================

"""
VISUALIZATION-A
"""

def plotted(elite_fit,elite_sp,ment,filename):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    plt.title('Genetic Algorithm')
    plt.ylabel('fitness')#**ffont)
    plt.xlabel('generation (n)',)#**ffont)
    
  
    plt.plot(elite_fit,'b-', linewidth=2)
#    plt.axis([0,len(elite_fit), 0.5,max(elite_fit)+0.5])
   
    
    plt.grid(which='major',color='r', linestyle='-', linewidth=0.5)
    plt.grid(which='minor',color='k', linestyle='-', linewidth=0.2)
    
    ax.xaxis.set_major_locator(MultipleLocator(ment))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig(filename,dpi=300)
    fig = plt.gcf()
    fig.set_size_inches(5, 2, forward=True)
    plt.show()
    print('best species :',end=" ")
    print('gen :',elite_fit.index(max(elite_fit)))
    print(elite_sp[ elite_fit.index(max(elite_fit)) ])
    print(max(elite_fit))

plotted(elite_fit,elite_ind,1,'test-0.png')

"""
VISUALIZATION-CONCLUDE
"""

#elitism_ind = list()
#for i in elite_ind:
#    if i not in elitism_ind:
#        elitism_ind.append(i)
#
#elitism_fit = list()
#for i in elitism_ind:
#    elitism_fit.append(elite_fit[elite_ind.index(i)])
#    
#plotted(elitism_fit,elitism_ind,1,'test-1.png')

with pd.ExcelWriter("GA_RESULT_s1_ntd2_f.xlsx") as writer:
    for c in range(len(kotak)):
        pd.DataFrame(kotak[c]).to_excel(writer, sheet_name="gen_"+str(c),index=False)
        
        

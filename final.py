
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from qpsolvers import solve_qp
import matplotlib.pyplot as plt


# In[16]:


# For deleting unnecessary prefix eg, "1:0.123432", "2:12.234" in each cell
def del_colname(x):
    x = str(x)
    x = x.split(":")
    return float(x[-1])


# In[73]:


path = "./Datasets/OHSUMED/QueryLevelNorm/Fold1/train.txt"
data = pd.read_csv(path, delimiter=" ", header=None)
data = pd.DataFrame(data)
data = data.drop([47, 48, 49],axis=1) # 47, 48, 49th cols are ["#docid", "=", docID]
data = data.apply(np.vectorize(del_colname))

data_q_min = int(data[1].min())
# data_q_max = int(data[1].max())
data_q_max = data_q_min+30
data_qnum = data_q_max - data_q_min + 1

w = pd.Series([0]*45)
slack = 0
SqChanged = 0

data_ls = [None] * data_qnum
y_hat_ls = [None] * data_qnum #change every element of this list every iteration

for i in range(data_q_min, data_q_max+1):
    # For each query id, store them as a separate dataframe in a list
    # In each query id dataframe, they are ranked by relevance
    data_ls[i-data_q_min] = data.loc[data[1] == i].sort_values(0, ascending = False)

    # After sorting, drop the original index
    data_ls[i-data_q_min].reset_index(drop = True)


# In[74]:


def feature_map(k, df_data, y=list()): #1*n
    [data_Row,data_Col] = df_data.shape
    docs_num = data_Row
        
    ones = np.ones(docs_num)
    zeros = np.zeros(docs_num)
    
    if(len(y)==0):
        y = np.arange(1,docs_num+1)
    
    A = (k*ones+1)-y
    A = np.row_stack((A, zeros))
    array_A = np.max(A, axis = 0)
#     matrix_A = np.expand_dims(array_A, axis=0)
    
    return np.dot(array_A, df_data.iloc[:,2:])


# In[75]:


def objF(k, df_data, y=list()): #1*1
    featureMap = feature_map(k, df_data, y)
    obj = np.dot(featureMap, w.T)
    return obj


# In[76]:


def NDCG(k, df_data, r): #1*1
    """ NDCG is the value function.
    It is made of a discount function of ranking D(r) and an increasing function phi(g) for relevance score.
    D(r) = 1 / log2(1+r), 
    phi(g) = np.power(2, g) - 1
    It favors higher ranking for highly relevant doc.
    """
    r = r[0:k]
    g = df_data.iloc[:,0].values[0:k]
    
    res = np.dot(1. / np.log2(1+r), (np.power(2,g) - 1))
    return res


# In[77]:


def kuhn_m(k, df_data):
    res = []
    
    [data_Row,data_Col] = df_data.shape
    docs_num = data_Row
        
    ones = np.ones(docs_num)
    zeros = np.zeros(docs_num)
    
    for i in range(docs_num):
        rank = i+1
        
        y = np.ones(docs_num)*rank
        A = (k*ones+1)-y
        A = np.row_stack((A, zeros))
        array_A = np.max(A, axis = 0)
    
        xw = np.dot(w, df_data.iloc[:,2:].T) #1*m
        awx = np.multiply(array_A, xw) #1*m
        
        g = df_data.iloc[:,0].values #1*m
        gd = np.dot(1. / np.log2(1+rank), (np.power(2,g) - 1)) #1*m
        
        res.append(awx-gd)
    
    cost = np.asarray(res) #m*m
    
    row_ind, col_ind = linear_sum_assignment(cost)
    y_pred = cost[row_ind,col_ind]
    
    return y_pred


# In[78]:


def constrain(k, df_data, y_hat): #return 1 for satisfy, 0 for violate
    [data_Row,data_Col] = df_data.shape
    docs_num = data_Row
    
    objYq = objF(k, df_data)
    
    objY = objF(k, df_data, y_hat)
    
    r = np.arange(1,docs_num+1)
    ndcg = NDCG(k, df_data, r)
    deltaQy = 1 - ndcg
    
    cons = objYq - objY - deltaQy + slack
#     print("cons:", cons)
    if(cons<0):
        return 0
    return 1


# In[79]:


def alg(k):
    global SqChanged
    for i in range(data_qnum):
        df_data = data_ls[i]
        y_hat = kuhn_m(k, df_data)
        y_hat_ls[i] = y_hat
        if(constrain(k, df_data, y_hat)==0):
            SqChanged = 1


# In[80]:


def opt(k):
    global w
    v = []
    b = []
    for i in range(data_qnum):
        df_data = data_ls[i]
        
        v.append(feature_map(k, data_ls[i], y_hat_ls[i]) - feature_map(k, data_ls[i]))
        
        [data_Row,data_Col] = df_data.shape
        docs_num = data_Row
        r = np.arange(1,docs_num+1)
        bq = 1-NDCG(k, df_data, r)
        b.append(bq)
        
    v_matrix = np.asarray(v) #q*n
    K = np.dot(v_matrix, v_matrix.T)
    
    b_matrix = np.asarray(b) #1*q
    
    G = -1*np.eye(data_qnum)
    h = np.zeros(data_qnum)
    A = np.ones(data_qnum)
    b = np.array([10])
#     print(K.shape, b_matrix.shape, G.shape, h.shape, A.shape, b.shape)
    
    try:
        alpha = solve_qp(K, b_matrix, G, h, A, b)
    except ValueError:
        print("ValueError")
        return 0
    w =  np.dot(alpha, v_matrix)
    


# In[81]:


test = pd.read_csv('./Datasets/OHSUMED/QueryLevelNorm/Fold1/test.txt', delimiter=" ", header=None)
test = pd.DataFrame(test)
test = test.drop([47, 48, 49],axis=1) # 47, 48, 49th cols are ["#docid", "=", docID]
test = test.apply(np.vectorize(del_colname))

test_q_min = int(test[1].min())
# test_q_max = q_min+30
test_q_max = int(test[1].max())
test_qnum = test_q_max - test_q_min + 1

test_ls = [None] * test_qnum

NDCG_Yq_ls = [] #i*k
NDCG_Y_hat_ls = [] #i*k

for i in range(test_q_min, test_q_max+1):
    test_ls[i-test_q_min] = test.loc[test[1] == i].sort_values(0, ascending = False)
    test_ls[i-test_q_min].reset_index(drop = True)


# In[82]:


def testw(k):
    
    NDCG_Yq = 0
    NDCG_Y_hat = 0
        
    for i in range(test_qnum):
        
        df_test = test_ls[i]
        [data_Row,data_Col] = df_test.shape
        docs_num = data_Row
        
        y_hat = np.dot(w, df_test.iloc[:,2:].T) #1*m
        r = np.argsort(y_hat)
        
        l = [0]*len(r)
        for i in range(len(r)):
            l[r[i]]=i+1
        l = pd.Series(l)
        
        NDCG_Yq = NDCG_Yq + NDCG(k, df_test, np.arange(1,docs_num+1))
        NDCG_Y_hat = NDCG_Y_hat + NDCG(k, df_test, l)
        
    NDCG_Yq_ls.append(NDCG_Yq)
    NDCG_Y_hat_ls.append(NDCG_Y_hat)


# In[83]:


def line():
    q = np.arange(k_range)+1
    plt.xlabel('k')
    plt.ylabel('NDCG')
    plt.plot(q, np.divide(NDCG_Y_hat_ls, NDCG_Yq_ls))


# In[ ]:


if __name__=="__main__":
    k_range = 10
    # print(w)
    for j in range(k_range):
        k = j+1
        print("k", k)
        for i in range(5):
            print("round", i)
            alg(k)
            if(SqChanged==1):
                opt(k)
            else:
                break
        testw(k)
    print(np.divide(NDCG_Y_hat_ls, NDCG_Yq_ls))


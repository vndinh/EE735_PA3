import pickle
import random
import numpy as np

from utils import get_data_set
'''
X, Y = get_data_set("train")
print(X.shape)
print(Y.shape)
Z = np.concatenate((X,Y),axis=1)
np.random.shuffle(Z)
print(Z.shape)
X_ = Z[:,0:3072]
Y_ = Z[:,3072:3082]
print(X_.shape)
print(Y_.shape)
'''
'''
idx = [i for i in range(50000)]
random.shuffle(idx)
idx1 = idx[0:4000]
idx2 = idx[4000:50000]
with open('..\\data\\svtrain.p', 'wb') as fp:
	pickle.dump(idx1, fp)

with open('..\\data\\usvtrain.p', 'wb') as fp:
	pickle.dump(idx2, fp)
X1 = X[idx1, :]
print(X1.shape)
X2 = X[idx2, :]
print(X2.shape)
'''
'''
arr = np.arange(9).reshape((3, 3))
print(arr)
np.random.shuffle(arr)
print(arr)
'''
train_x, train_y = get_data_set("train")
with open('..\\data\\svtrain.p', 'rb') as fp:
	idx = pickle.load(fp)
train_x = train_x[idx, :]
train_y = train_y[idx, :]
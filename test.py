import numpy as np
import one_NN
# a=np.loadtxt('/home/victorfang/Desktop/two datasets/sonar-train.txt',delimiter=',')
# b=np.matmul(a,a.T)
# # values,vectors=np.linalg.eig(b)
# val,vec=np.linalg.eig(np.array([[1,-3,3],[3,-5,3],[6,-6,4]]))
# val=val.real
# vec=vec.real
# vec=np.insert(vec,[0],val,0)
# print(vec)
# print(vec[0].argsort())
# vec=vec[:,np.argsort(-vec[0])]
# vec=np.delete(vec,[0],0)
# vec=vec[:,0:2]
# vec=np.delete(vec,-1,1)
# print(vec)

data_train=np.array([[1,2,3],[2,2,3],[3,2,3],[4,2,3]],dtype=np.float64)

mean=np.mean(data_train,axis=0)
data_train=data_train-mean
label_train=np.array([0,1,2,3])
data_test=np.array([[1,2,3],[2,1,3]],dtype=np.float64)
print(one_NN.one_NN(data_train,label_train,data_test))
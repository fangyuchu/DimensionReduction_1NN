import numpy as np
import MDS
import KNN_search
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

data_train=np.array([[0,2,3],[10,2,3],[0,2,3]],dtype=np.float64)

print(MDS.MDS(data_train,2))
# c=np.argsort(data_train)[:,1:]
# for i in range(data_train.shape[0]):
#     data_train[i,c[i]]=0
# print(data_train)
# a=np.array([[0,1],[1,-1],[-1,0]])
# print(np.cov(a,rowvar=False))
# c=np.mean(a,axis=1).reshape([3,1])
# # c=c.reshape([3,1])
# print(c)
# print(0.5*(a))
# b=np.array([0,3,4])
# print(np.diag(b))
# u,s,v=np.linalg.svd(a)
# print(v)
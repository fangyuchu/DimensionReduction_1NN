import numpy as np
# def CalDistance(vecs):
#     #vecs每行为一个向量
#     #返回向量之间的欧氏距离的矩阵
#     #行列的交点为这两个索引代表的向量的距离
#     dist=np.zeros([vecs.shape[0],vecs.shape[0]],dtype=np.float64)
#     for i in range(vecs.shape[0]):
#         for j in range(i+1,vecs.shape[0]):
#             dist[i][j]=np.linalg.norm(vecs[i]-vecs[j])                          #计算距离
#     for i in range(vecs.shape[0]):
#         for j in range(i):
#             dist[i][j]=dist[j][i]                                               #将对称的距离复制过去
#     return dist
def cal_distance(arr,obj_arr):
    #计算arr向量和obj_arr向量簇中各向量的距离，返回距离向量
    #obj_arr每行为一个向量
    dist=np.zeros(obj_arr.shape[0],dtype=np.float64)
    for i in range(obj_arr.shape[0]):
        dist[i]=np.linalg.norm(arr-obj_arr[i])
    return dist

def one_NN(data_train,label_train,data_test):
    label_test=np.zeros(data_test.shape[0])
    for i in range(data_test.shape[0]):
        d=cal_distance(data_test[i],data_train)
        index_min_distance=d.argsort()[0]
        label_test[i]=label_train[index_min_distance]
    return label_test

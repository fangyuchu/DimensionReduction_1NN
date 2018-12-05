#Multiple Dimensional Scaling
import numpy as np
def MDS(data,dimension,dist=0):
    #data一行为一个数据
    #dist为距离矩阵，dimension为约减后的维度
    #参考周志华机器学习第10章中介绍
    if dist is 0:
        dist=cal_distance(data)
    dist_2=np.square(dist)
    distI_2=np.mean(dist_2,axis=1).reshape([dist.shape[0],1])                     #对每行求平均值，并将结果转化为一列
    distJ_2=np.mean(dist_2,axis=0)                                                #对每列求平均值
    dist_all=np.mean(dist_2)/dist.shape[0]                                        #所有数据的平均值
    B=-0.5*(dist_2-distI_2-distJ_2+dist_all)
    val,vec=np.linalg.eig(B)
    vec=vec[:,np.argsort(-val)]
    vec=vec[:,:dimension]
    val=np.diag(-np.sort(-val)[:dimension])                                #生成特征值的对角矩阵
    return np.matmul(vec,np.sqrt(val))

def cal_distance(vecs):
    #vecs每行为一个向量
    #返回向量之间的欧氏距离的矩阵
    #行列的交点为这两个索引代表的向量的距离
    dist=np.zeros([vecs.shape[0],vecs.shape[0]],dtype=np.float64)
    for i in range(vecs.shape[0]):
        for j in range(i+1,vecs.shape[0]):
            dist[i][j]=np.linalg.norm(vecs[i]-vecs[j])                          #计算距离
    for i in range(vecs.shape[0]):
        for j in range(i):
            dist[i][j]=dist[j][i]                                               #将对称的距离复制过去
    return dist
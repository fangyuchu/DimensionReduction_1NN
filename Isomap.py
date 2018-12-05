import numpy as np
import MDS
import KNN_search
def Isomap(data,k,dimension):
    dist=MDS.cal_distance(data)
    c = np.argsort(dist)[:, k+1:]                               #取k近邻的索引
    for i in range(dist.shape[0]):
        dist[i, c[i]] = -1                                       #将k近邻以外的距离设为-1
    dist=floyd(dist)
    return MDS.MDS(data,dimension,dist)

def floyd(dist):
    for k in range(dist.shape[0]):
        for i in range(dist.shape[0]):
            for j in range(dist.shape[0]):
                if dist[i,k]!=-1 and dist[k,j]!=-1 and (dist[i,j]>dist[i,k]+dist[k,j] or dist[i,j]!=-1):
                    dist[i,j]=dist[i,k]+dist[k,j]
    return dist

def main(train_file,test_file,dimension,k):
    #用pca或svd降维，并接1nn
    data = np.loadtxt(train_file, delimiter=',')
    data_train = np.delete(data, -1, 1)                                #获得训练数据
    label_train=data[:,-1]                                             #获得训练数据的标签
    data=np.loadtxt(test_file,delimiter=',')
    data_test=np.delete(data,-1,1)                                      #获得训练数据
    label_test=data[:,-1]                                               #获得训练数据的标签

    data=np.vstack((data_train,data_test))                            #将测试数据和训练数据concat
    data=Isomap(data,k,dimension)
    data_train=data[0:data_train.shape[0]]
    data_test=data[data_train.shape[0]:,:]

    label_learned=KNN_search.KNN(1,data_train,label_train,data_test)       #获取训练所得的标签
    accuracy=np.mean(label_test==label_learned)                         #计算准确率
    print("the accuracy of the algorithm using ISOMAP is %f"%accuracy)

if __name__=='__main__':
    train_f='/home/victorfang/Desktop/two datasets/sonar-train.txt'
    test_f='/home/victorfang/Desktop/two datasets/sonar-test.txt'
    main(train_f,test_f,10,4)
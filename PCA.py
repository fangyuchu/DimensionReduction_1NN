import numpy as np
import one_NN
def PCA(data,dimension):
    #使用pca对data进行降维
    #data中一行为一个数据
    data=data-np.mean(data,axis=0)                                          #去中心化
    val,vec=np.linalg.eig(np.matmul(data.T,data))                           #求解特征值与特征向量，其中特征向量按列
    val=val.real
    vec=vec.real
    vec = vec[:,np.argsort(-val)]                                           #将矩阵以特征值从大到小排序
    projecting_matrix=vec[:,0:dimension]
    return projecting_matrix

def main(train_file,dimension,test_file):
    #用pca降维，并接1nn
    data = np.loadtxt(train_file, delimiter=',')
    data_train = np.delete(data, -1, 1)                                #获得训练数据
    label_train=data[:,-1]                                             #获得训练数据的标签
    projecting_matrix=PCA(data_train,dimension)                        #获得投影矩阵
    data=np.loadtxt(test_file,delimiter=',')                           #读取测试数据
    data_test=np.delete(data,-1,1)                                     #去除测试数据的标签
    label_test=data[:,-1]
    data_train=np.matmul(data_train,projecting_matrix)                  #训练数据进行投影
    data_test=np.matmul(data_test,projecting_matrix)                    #测试数据进行投影
    label_learned=one_NN.one_NN(data_train,label_train,data_test)       #获取训练所得的标签
    accuracy=np.mean(label_test==label_learned)                         #计算准确率
    print("the accuracy of the algorithm using PCA and 1nn is %f"%accuracy)

if __name__=='__main__':
    train_f='/home/victorfang/Desktop/two datasets/sonar-train.txt'
    test_f='/home/victorfang/Desktop/two datasets/sonar-test.txt'
    main(train_f,20,test_f)
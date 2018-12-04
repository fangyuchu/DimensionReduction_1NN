import numpy as np
import one_NN
def PCA(data,dimension):
    #使用pca对data进行降维
    #data中一行为一个数据
    data=data-np.mean(data,axis=0)                                          #去中心化
    val,vec=np.linalg.eig(np.matmul(data.T,data))                           #求解特征值与特征向量，其中特征向量按列
    val=val.real
    vec=vec.real
    vec = np.insert(vec, [0], val, 0)                                       #将特征值插入矩阵，与每一列的特征向量相对应
    vec = vec[:,np.argsort(-vec[0])]                                        #将矩阵以特征值从大到小排序
    vec = np.delete(vec, [0], 0)                                            #获得特征向量矩阵
    projecting_matrix=vec[:,0:dimension]
    return projecting_matrix
def main(train_file,dimension,test_file):
    #用pca降维，并接1nn
    data = np.loadtxt(train_file, delimiter=',')
    data_train = np.delete(data, -1, 1)  # 去掉标签
    label_train=data[:,-1]
    projecting_matrix=PCA(data_train,dimension)                        #获得投影矩阵
    data=np.loadtxt(test_file,delimiter=',')                           #读取测试数据
    data_test=np.delete(data,-1,1)                                     #去处测试数据的标签
    label_test=data[:,-1]
    data_train=np.matmul(data_train,projecting_matrix)
    data_test=np.matmul(data_test,projecting_matrix)
    label_learned=one_NN.one_NN(data_train,label_train,data_test)
    accuracy=np.mean(label_test==label_learned)
    print("the accuracy of the algorithm using PCA and 1nn is %f"%accuracy)

if __name__=='__main__':
    train_f='/home/victorfang/Desktop/two datasets/sonar-train.txt'
    test_f='/home/victorfang/Desktop/two datasets/sonar-test.txt'
    main(train_f,20,test_f)
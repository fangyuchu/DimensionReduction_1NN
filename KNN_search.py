import numpy as np

def cal_distance(arr,obj_arr):
    #计算arr向量和obj_arr向量簇中各向量的距离，返回距离向量
    #obj_arr每行为一个向量
    dist=np.zeros(obj_arr.shape[0],dtype=np.float64)
    for i in range(obj_arr.shape[0]):
        dist[i]=np.linalg.norm(arr-obj_arr[i])
    return dist

def KNN(k,data_train,label_train,data_test):
    label_test=np.zeros(data_test.shape[0])
    label_train=label_train.astype(int)
    for i in range(data_test.shape[0]):
        d=cal_distance(data_test[i],data_train)
        index_min_distance=d.argsort()[:k]
        label=np.argmax(np.bincount(label_train[index_min_distance]+1))-1   #查找近邻中哪个标签更多
        label_test[i]=label
    return label_test

import numpy as np
import MDS
import KNN_search
import time
import networkx as nx
import matplotlib.pyplot as plt
def Isomap(data,k,dimension):

    G=nx.Graph()
    for i in range(data.shape[0]):
        G.add_node(i)
    dist=MDS.cal_distance(data)
    c = np.argsort(dist)[:,0:k+1]  # 取k近邻的索引
    for i in range(dist.shape[0]):
        for j in range(c[i].shape[0]):
            G.add_weighted_edges_from([(i,c[i,j],dist[i,c[i,j]])])
    path=dict(nx.all_pairs_dijkstra_path_length(G))
    edge=connect_graph(data,path)
    G.add_weighted_edges_from(edge)
    path=dict(nx.all_pairs_dijkstra_path_length(G))
    dist=np.array([[0 for i in range(data.shape[0])]for j in range(data.shape[0])],dtype=np.float64)
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            dist[i,j]=path[i][j]
    return MDS.MDS(data,dimension,dist)

def floyd(dist):
    for k in range(dist.shape[0]):
        for i in range(dist.shape[0]):
            for j in range(dist.shape[0]):
                if dist[i,k]!=-1 and dist[k,j]!=-1 and (dist[i,j]>dist[i,k]+dist[k,j] or dist[i,j]==-1):
                    dist[i,j]=dist[i,k]+dist[k,j]
                    if i==0 and j==1:
                        print(k)
    return dist

def connect_graph(data,path):
    #找到不连通图中都有哪些连通的集合
    #将集合与集合之间最短的边连通
    #返回连通后的距离矩阵
    index=0
    arr=np.array([-1 for i in range(data.shape[0])])             #存储集合的连通，同个元素代表同个集合
    for i in range(data.shape[0]):
        if arr[i]!=-1:
            continue                                             #已加入集合了
        # ind=np.argwhere(dist[i]!=-1).flatten()
        ind=list(dict.keys(path[i]))

        arr[ind]=index
        index+=1
    sets=[0 for i in range(index)]
    for i in range(index):
        sets[i]=np.argwhere(arr==i).flatten()
    edge=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            n1,n2,d=min_connection_route(data[sets[i]],data[sets[j]],sets[i],sets[j])
            #dist[n1,n2]=dist[n2,n1]=d
            edge.append((n1,n2,d))
    return edge

def min_connection_route(data1,data2,set1,set2):
    #找到将两个集合连通起来的最短的路径及长度
    #data是数据，sets是数据在原集合中的索引
    min_dist=np.linalg.norm(data1[0]-data2[0])
    node1=0
    node2=0
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            dist=np.linalg.norm(data1[i]-data2[j])
            if dist<min_dist:
                min_dist=dist
                node1=i
                node2=j
    return set1[node1],set2[node2],min_dist

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
    print("k and accuracy using ISOMAP is %d, %f"%(k,accuracy))

if __name__=='__main__':
    dataset='splice'
    train_f='/home/victorfang/Desktop/two datasets/'+dataset+'-train.txt'
    test_f='/home/victorfang/Desktop/two datasets/'+dataset+'-test.txt'
    print("---------------------------------d=30------------------------------------------------------")
    for k in range(5):
        t_start = time.time()
        main(train_f, test_f, 30,36 + 2 * k)
        t_end = time.time()
        print('totally time cost:%d s' % (t_end - t_start))
    print("---------------------------------d=20------------------------------------------------------")
    for k in range(20):
        t_start=time.time()
        main(train_f,test_f,20,4+2*k)
        t_end=time.time()
        print('totally time cost:%d s'%(t_end-t_start))
    print("---------------------------------d=10------------------------------------------------------")
    for k in range(20):
        t_start=time.time()
        main(train_f,test_f,10,4+2*k)
        t_end=time.time()
        print('totally time cost:%d s'%(t_end-t_start))
import numpy as np
def ReadFile(s):
    data=np.loadtxt(s,delimiter=',')
    return data

if __name__=='__main__':
    d=ReadFile('/home/victorfang/Desktop/two datasets/sonar-train.txt')
    b=d[0,6]
    print(b)
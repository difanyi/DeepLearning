# -*- coding: utf-8 -*-
'''
Algorithm 1. 标准BP算法
----
    输入： 训练集 D，学习率 η.
    过程： 
        1. 随即初始化连接权与阈值 (ω，θ).
        2. Repeat：
        3.   for x_k，y_k in D:
        4.     根据当前参数计算出样本误差 E_k.
        5.     根据公式计算出随机梯度项 g_k.
        6.     根据公式更新 (ω，θ).
        7.   end for
        8. until 达到停止条件
    输出：(ω，θ) - 即相应的多层前馈神经网络.
'''
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):#tanh导数
    return 1.0-np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))
    
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))
    
class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):#几层，每层里面的单元数；激活函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        
        self.weights=[]
        for i in range(1,len(layers)-1):#随机初始化权重
            for i in range(1, len(layers) - 1):
                self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
                self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
                
    def fit(self,X,y,learning_rate=0.2,epochs=10000):#BP算法
        X=np.atleast_2d(X)    #判断输入训练集是否为二维
        temp=np.ones([X.shape[0],X.shape[1]+1]) #创建一个与X形状相同的全1数组
        temp[:,0:-1]=X
        X=temp
        y=np.array(y)
            
        for k in range(epochs):
            i=np.random.randint(X.shape[0])
            a=[X[i]]
                
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))#计算权重，dot点积
            error=y[i]-a[-1]#计算误差
            deltas=[error*self.activation_deriv(a[-1])]
                
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
        
    def predict(self, x):#预测
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a     

nn = NeuralNetwork([2,2,1], 'logistic')
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([1,0,0,1])
nn.fit(x,y,0.1,10000)
for i in [[0,0], [0,1], [1,0], [1,1]]:
    print(i, nn.predict(i))

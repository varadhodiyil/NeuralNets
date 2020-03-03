import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, inputs, labels, input_nodes, hidden_nodes):
        self.inputs = inputs
        self.labels = labels
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        #self.bias = []
        self.weights_in = np.random.randn(self.inputs.shape[1],input_nodes)*np.sqrt(2/input_nodes)
        self.bias_in=np.zeros((1,input_nodes))
        # self.weights_hidden = np.array([[np.random.uniform(
        #     0.1, 0.9) for i in range(hidden_nodes)]])
        self.weights_hidden= np.random.randn(hidden_nodes,1)*np.sqrt(2/hidden_nodes)
        self.bias_hidden= np.zeros((1,1))
        self.predictions = np.zeros(labels.shape)
        self.alpha=0.01
        self.alpha_bias=0.05
        self.steps=1
        print(self.weights_in)
        print(self.weights_hidden)

    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0) 

    def relu(self, prediction):
        return np.maximum(0, prediction)

    def derivative_relu(self, prediction):
        return np.heaviside(prediction, 0)

    def forward(self):
        self.hl = self.relu(np.dot(self.inputs, self.weights_in)+self.bias_in)
        self.predictions = self.relu(np.dot(self.hl, self.weights_hidden)+self.bias_hidden)

    def backprop(self):
        # d_hidden = np.dot(
        #     self.hl.T, (2*(self.labels-self.predictions))*self.derivative_relu(self.predictions))
        # in_2=(self.labels - self.predictions)
        # inner=np.dot(2*(self.labels - self.predictions) * self.derivative_relu(
        #     self.predictions), self.weights_hidden.T)
        # d_in = np.dot(self.inputs.T,  (np.dot(2*(self.labels - self.predictions) * self.derivative_relu(
        #     self.predictions), self.weights_hidden.T) * self.derivative_relu(self.hl)))
        del_hidden=2*(self.labels-self.predictions)*self.derivative_relu(self.predictions)
        d_hidden = np.dot(
            self.hl.T, del_hidden)
        del_in=np.dot(2*(self.labels - self.predictions) * self.derivative_relu(
            self.predictions), self.weights_hidden.T) * self.derivative_relu(self.hl)
        d_in = np.dot(self.inputs.T, del_in)
        self.weights_in+=self.alpha* d_in
        self.bias_in+=self.alpha_bias* np.sum(del_in,axis=0)/self.inputs.shape[0]
        self.weights_hidden+=self.alpha* d_hidden
        self.bias_hidden+=self.alpha_bias* np.sum(del_hidden,axis=0)/self.inputs.shape[0]

    def get_predictions(self):
        return self.predictions>0.5

def main():
    path='C:/Users/GANESH/Downloads/circles500.csv'
    data=pd.read_csv('C:/Users/GANESH/Downloads/circles500.csv')
    labels= data[['Class']].to_numpy()
    input_=data[['X0','X1']].to_numpy()
    X_train, X_test, y_train, y_test=train_test_split(input_,labels,test_size=0.2)
    nn=NeuralNet(X_train,y_train,3,3)
    # for start in range(0,len(X_train)-50,50):
    #     nn.inputs=X_train[start:start+50]
    #     nn.labels=y_train[start:start+50]
    for i in range(10000):
        nn.forward()
        nn.backprop()
        #nn.steps+=1
    predictions=nn.get_predictions()
    print(accuracy_score(y_train,predictions))
    nn.inputs=X_test
    nn.forward()
    predictions=nn.get_predictions()
    print(accuracy_score(y_test,predictions))
    #print(accuracy_score(y_train,predictions))
    dic={True:'green',False:'red'}
    dic1={True:'x',False:'o'}

    y_true=np.empty((0,3), int)
    y_false=np.empty((0,3), int)
    for x0,x1,y in zip(X_test[:,0],X_test[:,1],list(predictions[:,0])):
        if y:
            y_true=np.vstack((y_true,np.array([x0,x1,int(y)])))
        else:
            y_false=np.vstack((y_false,np.array([x0,x1,int(y)])))

    plt.scatter(y_true[:,0],y_true[:,1],
    c=[dic[x] for x in y_true[:,2]],
    marker='x'
    )
    plt.scatter(y_false[:,0],y_false[:,1],
    c=[dic[x] for x in y_false[:,2]],
    marker='o'
    )
    plt.show()
    # plt.scatter(X_test[:,0],X_test[:,1], 
    # c= [dic[x] for x in predictions[:,0] if x==True], 
    # marker= [dic1[x] for x in predictions[:,0] if x==True])
    # plt.scatter(X_test[:,0],X_test[:,1], 
    # c= [dic[x] for x in predictions[:,0] if x==False], 
    # marker= [dic1[x] for x in predictions[:,0] if x==False])
    # plt.show()

if __name__ == "__main__":
    main()
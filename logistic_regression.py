import numpy as np 
from numpy import log,dot,e,shape
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y = True)
from sklearn.model_selection import train_test_split  
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)

def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])

def accuracy1(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)

    return f1_score
         
class LogidticRegression:

    def sigmoid(self,z):
        sig = 1/(1+e**(-z))
        return sig

    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    

    def fit(self,X,y,alpha=0.001,iter=400):
        
        weights,X = self.initialize(X)
        
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y) 
            return cost
  
        
        cost_list = np.zeros(iter,)
        
        for i in range(iter):
    
            weights = weights - alpha * dot(X.T, self.sigmoid(dot(X,weights)) - np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis


standardize(X_tr)
standardize(X_te)
obj1 = LogidticRegression()
a= obj1.fit(X_tr,y_tr)
b = obj1.predict(X_te)
c = obj1.predict(X_tr)

print(accuracy1(y_tr,b))

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylabel('J(cost)')
ax.set_xlabel('iterations')
x = ax.plot(range(400),a,'b.')

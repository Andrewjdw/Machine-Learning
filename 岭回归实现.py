# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:54:53 2020

@author: jidw
"""


import numpy as np
import matplotlib.pyplot as plt 

#sklearn实现

from sklearn.linear_model import Ridge,RidgeCV#Ridge岭回归,RidgeCV带有广义交叉验证的岭回归

x = np.linspace([1.0, 1.0], [10.0,10.0], num=10)
w = np.array([2, 5]).reshape(2,-1)
y = np.dot(x,w)

ridge_alphas = np.logspace(-10, 1, 200) #生成200个e-10到e-2之间的数值

# Ridge

def clf():
    coefs_list0 = []
    coefs_list1 = []
    for i in ridge_alphas:
        ridge = Ridge(alpha=i, fit_intercept=False)  #fit_intercept是否计算截距
        ridge.fit(x, y)
        coefs_list0.append(ridge.coef_[0][0])
        coefs_list1.append(ridge.coef_[0][1])
    ax = plt.gca()
    ax.plot(ridge_alphas, coefs_list0)
    ax.plot(ridge_alphas, coefs_list1)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])# 反转数轴，越靠左边 alpha 越大，正则化也越厉害
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.axis('tight')
    plt.show()

clf()


# RidgeCV
model = RidgeCV(alphas=ridge_alphas)
model.fit(x, y)
print(model.alpha_)




#原理实现

def ridge(X, y, alpha=0.01, intercept=False):
    #获取weights的维度
    if intercept:
       X = np.c_[X,np.ones(X.shape[0])]
    weights = np.dot(np.dot((np.dot(X.T, X) + np.dot(alpha , np.identity(X.shape[1]))),X.T),y)
    loss = (np.dot((np.dot(X, weights) - y).T ,(np.dot(X, weights) - y)) + np.dot(alpha , np.dot(weights.T, weights)))[0,0]
    return weights,loss
    
def clf():
    coefs_list0 = []
    coefs_list1 = []
    for i in ridge_alphas:
        weights,_ = ridge(x, y, alpha=i)
        coefs_list0.append(weights[0][0])
        coefs_list1.append(weights[1][0])
    ax = plt.gca()
    ax.plot(ridge_alphas, coefs_list0)
    ax.plot(ridge_alphas, coefs_list1)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])# 反转数轴，越靠左边 alpha 越大，正则化也越厉害
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.axis('tight')
    plt.show()

clf()


#梯度下降法求weights
def ridge1(X, y, alpha=0.001, intercept=False,max_step=10, study_rate=0.001):
    #获取weights的维度
    if intercept:
       X = np.c_[X,np.ones(X.shape[0])]
    weights = np.zeros(X.shape[1])
    loss = (np.dot((np.dot(X, weights) - y).T ,(np.dot(X, weights) - y)) + np.dot(alpha , np.dot(weights.T, weights)))[0,0]
    step = 0
    while step < max_step:
        grandient = np.dot(np.dot(X.T, X),weights) - np.dot(X.T, y) + alpha * weights
        weights = weights - study_rate * grandient
        loss = (np.dot((np.dot(X, weights) - y).T ,(np.dot(X, weights) - y)))[0,0]
        print('第%d次,损失为%f' % (step, loss), '权重为：',weights)
        step += 1
    return weights,loss
    
ridge1(x,y)







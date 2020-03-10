# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:27:34 2020

@author: jidw
"""


# 求解极值点

import numpy as np
from sympy import *


#获取变量
def get_vars(*args):
    arg_str = ''
    i=0
    for arg in args:
        i += 1
        if i < len(args):
            arg_str += arg +  ','
        else:
            arg_str += arg
    return arg_str

#获取hessian矩阵
def get_hessian(a,b):
    hessian = zeros(2, 2)  
    for i,fi in enumerate(f):
        for j,r in enumerate(vars):
            for k, s in enumerate(vars):
                hessian[j,k] = diff(diff(fi, r),s).subs({vars[0]:a, vars[1]:b})
    return hessian

#牛顿法迭代
def newton(max_step, x_init):
    i = 1
    data = []
    while i < max_step:
        if i == 1: 
            #第一次迭代
            grandient = np.array([diff(f1,vars[0]).subs({vars[0]:x_init[0], vars[1]:x_init[1]}), 
                                 diff(f1,vars[1]).subs({vars[0]:x_init[0], vars[1]:x_init[1]})]) 
            hessian = get_hessian(x_init[0], x_init[1])
            #此处需要对hessian进行求逆，需要先将其转为类型为float的矩阵再进行求解，否则可能报错
            new_ab = x_init - np.matmul(np.linalg.inv(np.mat(hessian,dtype='float')), grandient)
        else: 
            grandient = np.array([diff(f1,vars[0]).subs({vars[0]:new_ab[0,0], vars[1]:new_ab[0,1]}), 
                             diff(f1,vars[1]).subs({vars[0]:new_ab[0,0], vars[1]:new_ab[0,1]})])
            hessian = get_hessian(new_ab[0,0], new_ab[0,1])
            new_ab = np.array(new_ab - np.matmul(np.linalg.inv(np.mat(hessian,dtype='float')), grandient))
        print('迭代第%d次：%.5f %.5f' %(i, new_ab[0,0], new_ab[0,1]))
        data.append([new_ab[0,0], new_ab[0,1]])
        i = i + 1
    return new_ab,data

var = get_vars('a', 'b')
vars = symbols(var)
f1 = vars[0]**3 + vars[1]**3
f = sympify([str(f1)])
x_init = np.array([-50,-50])
max_step = 30
ab, data = newton(max_step, x_init)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3Dd
a = np.arange(-50, 50, 0.1)
b = np.arange(-50, 50, 0.1)
a,b = np.meshgrid(a,b)
y = a**4 + b**4 + a * b
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')

x_value = [-25, -12.5, -6.25, -3.125, -1.5625, -0.78124, -0.39062, -0.19531, -0.4883]


for i in x_value:
    x
x = [-25, -12, ]
y1 = [-25, -12]
z = [(-25)**4 + (-25)**4 + (-25)*(-25), (-12)**4 + (-12)**4 + (-12)*(-12)]
figure1 = ax.plot(x,y1,z,c='r')
figure2 = ax.plot_surface(a,b,y,cmap='jet')
plt.show()


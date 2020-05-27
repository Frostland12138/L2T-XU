#implementation of L2T pipline
import numpy as np
import scipy.linalg as la
from math import *
import matplotlib.pyplot as plt
def solve_W(Z_t,X_t):
    '''
    :param Z_t: np array(nt * n_feature), transformed target feature
    :param X_t: np array(nt * n_feature), original target feature
    :return: LDL docomposition solution of W
    '''
    Z=np.array(Z_t)
    X=np.array(X_t)


def matrix_factorization(R,P,Q,K,steps=5000,alpha=0.002,beta=0.02): #矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q=Q.T                 # .T操作表示矩阵的转置
    result=[]
    for step in range(steps): #梯度下降
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j])       # .DOT表示矩阵相乘
                    for k in range(K):
                      if R[i][j]>0:        #限制评分大于零
                        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   #增加正则化，并对损失函数求导，然后更新变量P
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])   #增加正则化，并对损失函数求导，然后更新变量Q
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
              if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)      #损失函数求和
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2)) #加入正则化后的损失函数求和
        result.append(e)
        if e<0.001:           #判断是否收敛，0.001为阈值
            break
    return P,Q.T,result
def my_matrix_factorization(R,P,Q,K,steps=100,alpha=0.5,beta=0.02): #矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q=Q.T                 # .T操作表示矩阵的转置
    result=[]
    last_e=0
    ess=[]
    delta=0
    for step in range(steps): #梯度下降
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j])       # .DOT表示矩阵相乘
                    for k in range(K):
                        #P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   #增加正则化，并对损失函数求导，然后更新变量P
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])   #增加正则化，并对损失函数求导，然后更新变量Q
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
              if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)      #损失函数求和
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2)) #加入正则化后的损失函数求和
        result.append(e)
        delta=abs(last_e-e)
        last_e=e
        ess.append(e)
        print(e)

        if delta<0.0001:           #判断是否收敛，0.001为阈值
            break
    return P,Q.T,result,ess
def my_matrix_factorization_using_Q_Learning(R,P,Q,K,steps=100,alpha=0.5,beta=0.02): #矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q=Q.T                 # .T操作表示矩阵的转置
    result=[]
    last_e=0
    ess=[]
    delta=0
    mean_delta=0
    alpha=alpha
    for step in range(steps): #梯度下降
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j])       # .DOT表示矩阵相乘
                    for k in range(K):
                        #P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   #增加正则化，并对损失函数求导，然后更新变量P
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])   #增加正则化，并对损失函数求导，然后更新变量Q
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
              if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)      #损失函数求和
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2)) #加入正则化后的损失函数求和
        result.append(e)
        delta=abs(last_e-e)
        last_e=e
        ess.append(e)
        print(e)

        if delta<mean_delta:
            alpha=alpha*1.1
        else:
            alpha=alpha*0.9
        if alpha>0.65:
            alpha=0.65
        print(alpha)
        if mean_delta==0:
            mean_delta=delta
        else:
            mean_delta=(mean_delta+delta)/2
        if delta<0.001:           #判断是否收敛，0.001为阈值
            break
    return P,Q.T,result,ess
def decent_one_step(R,P,Q,K,steps=100,alpha=0.5,beta=0.02):
    result = []
    last_e = 0
    #ess = []
    delta = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            eij = R[i][j] - np.dot(P[i, :], Q[:, j])  # .DOT表示矩阵相乘
            for k in range(K):
                # P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   #增加正则化，并对损失函数求导，然后更新变量P
                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])  # 增加正则化，并对损失函数求导，然后更新变量Q
    eR = np.dot(P, Q)
    e = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)  # 损失函数求和
                for k in range(K):
                    e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))  # 加入正则化后的损失函数求和
    result.append(e)
    delta = abs(last_e - e)
    last_e = e
    #ess.append(e)
    return Q,e
def my_matrix_factorization_mmd(R,P,Q,K,steps=100,alpha=0.5,beta=0.02): #矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q=Q.T                 # .T操作表示矩阵的转置
    result=[]
    last_e=0
    ess=[]
    delta=0

    for step in range(steps): #梯度下降
        Q,e=decent_one_step(R,P,Q,K,steps=100,alpha=0.5,beta=0.02)
        result.append(e)
        delta = abs(last_e - e)
        last_e = e
        if delta<0.001:           #判断是否收敛，0.001为阈值
            break
    return P,Q.T,result,ess
def L2T_matrix_factorization(R,P,Q,K,steps=100,alpha=0.5,beta=0.02): #矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q=Q.T                 # .T操作表示矩阵的转置
    result=[]
    last_e=0
    ess=[]
    delta=0
    for step in range(steps): #梯度下降
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j])       # .DOT表示矩阵相乘
                    for k in range(K):
                        #P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   #增加正则化，并对损失函数求导，然后更新变量P
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])   #增加正则化，并对损失函数求导，然后更新变量Q
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
              if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)      #损失函数求和
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2)) #加入正则化后的损失函数求和
        result.append(e)
        delta=abs(last_e-e)
        last_e=e
        ess.append(e)
        print(e)

        if delta<0.001:           #判断是否收敛，0.001为阈值
            break
    return P,Q.T,result,ess
def decomposition_example(R_real):
    R = [  # 原始矩阵
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ]
    R = np.array(R)
    N = len(R)  # 原矩阵R的行数
    M = len(R[0])  # 原矩阵R的列数
    K = 3  # K值可根据需求改变
    P = np.random.rand(N, K)  # 随机生成一个 N行 K列的矩阵
    Q = np.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵
    nP, nQ, result = matrix_factorization(R, P, Q, K)
    print(R)  # 输出原矩阵
    R_MF = np.dot(nP, nQ.T)
    print(R_MF)  # 输出新矩阵
    # 画图
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
    zzz=1
def my_decomposition():
    R=[[39,54,69,84],[39,54,69,84]]
    R = np.array(R)
    N = len(R)  # 原矩阵R的行数
    M = len(R[0])  # 原矩阵R的列数
    K = 5  # K值可根据需求改变
    P = np.array([[1,2,3,4,5],[1,2,3,4,5]])
    Q = np.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵
    nP, nQ, result,ess = my_matrix_factorization(R, P, Q, K)
    print(R)  # 输出原矩阵
    #print([[1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4]])
    #print(nQ)
    R_MF = np.dot(nP, nQ.T)
    print(R_MF)  # 输出新矩阵
    # 画图
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
    zzz=1
def toy_example_MMDloss(X1,X2,Z1,Z2,W1,W2):
    P1 = np.array(X1)
    R1 = np.array(Z1)
    N1 = len(R1)  # 原矩阵R的行数
    M1 = len(R1[0])  # 原矩阵R的列数
    K1 = Z1.shape[1]  # K值可根据需求改变
    if W1.shape[0] == 1:
        Q1 = np.random.rand(M1, K1)  # 随机生成一个 M行 K列的矩阵
    else:
        Q1 = np.array(W1)
    P2 = np.array(X2)
    R2 = np.array(Z2)
    N2 = len(R2)  # 原矩阵R的行数
    M2 = len(R2[0])  # 原矩阵R的列数
    K2 = Z2.shape[1]  # K值可根据需求改变
    if W2.shape[0] == 1:
        Q2 = np.random.rand(M2, K2)  # 随机生成一个 M行 K列的矩阵
    else:
        Q2 = np.array(W2)
    nP1, nQ1, result1, ess1 = my_matrix_factorization(R1, P1, Q1, K1)
    nP2, nQ2, result2, ess2 = my_matrix_factorization(R2, P2, Q2, K2)
    #print(R1)  # 输出原矩阵
    # print([[1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4]])
    # print(nQ)
    R_MF = np.dot(nP1, nQ1.T)
    #plt.plot(range(len(ess)), ess)
    #plt.xlabel("iteration")
    #plt.ylabel("loss")
    #plt.show()
    # plt.scatter(R_MF[:, 0], R_MF[:, 1], c='g', s=0.1)
    # plt.scatter(R[:, 0], R[:, 1], c='r', s=0.1)
    # plt.show()
    '''
    D=1
    #print(R_MF)  # 输出新矩阵
    # 画图
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
    '''
    return nQ1.T, ess1,nQ2.T, ess2
def toy_example(X,Z,W):
    P=np.array(X)
    R=np.array(Z)
    N = len(R)  # 原矩阵R的行数
    M = len(R[0])  # 原矩阵R的列数
    K = Z.shape[1]  # K值可根据需求改变
    if W.shape[0] == 1:
        Q = np.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵
    else:
        Q = np.array(W)*200
    nP, nQ, result ,ess= my_matrix_factorization(R, P, Q, K)
    print(R)  # 输出原矩阵
    # print([[1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4]])
    # print(nQ)
    R_MF = np.dot(nP, nQ.T)
    #plt.plot(range(len(ess)), ess)
    #plt.xlabel("iteration")
    #plt.ylabel("loss")
    #plt.show()
    #plt.scatter(R_MF[:, 0], R_MF[:, 1], c='g', s=0.1)
    #plt.scatter(R[:, 0], R[:, 1], c='r', s=0.1)
    #plt.show()
    '''
    D=1
    #print(R_MF)  # 输出新矩阵
    # 画图
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
    '''
    return nQ.T,ess
    zzz = 1
def toy_example_Q_Learning(X,Z,W):
    P=np.array(X)
    R=np.array(Z)
    N = len(R)  # 原矩阵R的行数
    M = len(R[0])  # 原矩阵R的列数
    K = Z.shape[1]  # K值可根据需求改变
    if W.shape[0] == 1:
        Q = np.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵
    else:
        Q = np.array(W)*50
    nP, nQ, result ,ess= my_matrix_factorization_using_Q_Learning(R, P, Q, K)
    print(R)  # 输出原矩阵
    # print([[1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4]])
    # print(nQ)
    R_MF = np.dot(nP, nQ.T)
    #plt.plot(range(len(ess)), ess)
    #plt.xlabel("iteration")
    #plt.ylabel("loss")
    #plt.show()
    #plt.scatter(R_MF[:, 0], R_MF[:, 1], c='g', s=0.1)
    #plt.scatter(R[:, 0], R[:, 1], c='r', s=0.1)
    #plt.show()
    '''
    D=1
    #print(R_MF)  # 输出新矩阵
    # 画图
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
    '''
    return nQ.T,ess
    zzz = 1

class Experience:
    def __init__(self,X_s,Y_s,X_t,Y_t,W,L):
        '''
        :param X_s:
        :param Y_s:
        :param X_t:
        :param Y_t:
        :param W:
        :param L:
        '''
        self.X_s=X_s
        self.Y_s=Y_s
        self.X_t=X_t
        self.Y_t=Y_t
        self.W=W
        self.L=L

import numpy as np
import random
import math
from sklearn import metrics
from hyperopt import  hp,fmin, rand, tpe, space_eval
from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
def detailed_mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean(), XX, XY, YY

def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()
def Beta_Kernel_MMD(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("calculating MMD")
    it=(mmd_rbf(X,Y,g) for i,g in enumerate(Gamma))
    sum=0.0
    for i,B in enumerate(Beta):
        sum+=B*next(it)
    return sum
def Beta_Kernel_MMD_return_detail(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("calculating MMD")
    it=[mmd_rbf(X,Y,g) for i,g in enumerate(Gamma)]
    sum=0.0
    for i,B in enumerate(Beta):
        sum+=B*it[i]
    return sum,it
def d_Beta_Kernel_MMD(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100],W=[]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("calculating MMD")

    it=(d_mmd_rbf(X,Y,g,W) for i,g in enumerate(Gamma))
    sum=np.zeros(W.shape)
    for i,B in enumerate(Beta):
        sum+=B*next(it)
    return sum
def d_mmd_rbf(X,Y,gamma=1.0,W=[]):
    X=np.array(X)
    Y=np.array(Y)
    W=np.array(W)
    d_W_1=np.zeros(W.shape)
    for i,X_i in enumerate(X):
        for j,Y_j in enumerate(Y):
            d_W_1=d_W_1+d_mmd_rbf_cal_K_k_ii_(X_i,Y_j,gamma,W)
    d_W_1=d_W_1/X.shape[0]
    d_W_1=d_W_1/Y.shape[0]

    d_W_2 = np.zeros(W.shape)
    for i, X_i in enumerate(X):
        for j, Y_j in enumerate(X):
            d_W_2 = d_W_2 + d_mmd_rbf_cal_K_k_ii_(X_i, Y_j, gamma, W)
    d_W_2 = d_W_2 / X.shape[0]
    d_W_2 = d_W_2 / X.shape[0]

    d_W_3 = np.zeros(W.shape)
    for i, X_i in enumerate(Y):
        for j, Y_j in enumerate(Y):
            d_W_3 = d_W_3 + d_mmd_rbf_cal_K_k_ii_(X_i, Y_j, gamma, W)
    d_W_3 = d_W_3 / Y.shape[0]
    d_W_3 = d_W_3 / Y.shape[0]

    d_W=-2*d_W_1+d_W_2+d_W_3
    return d_W
def d_mmd_rbf_cal_K_k_ii_(X, Y, gamma=1.0,W=[]):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    X_i=[i for i in  X]
    Y_i=[i for i in  Y]
    X_i=np.array(X_i).reshape(1, -1)
    Y_i=np.array(Y_i).reshape(1, -1)
    K_k=metrics.pairwise.rbf_kernel(X_i, Y_i, gamma)
    res=(X_i.T.dot(Y_i)*W*(-2)*K_k[0][0])/gamma
    return res
def Estimate_Q(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("estimate Q")
    X = np.array(X)
    Y = np.array(Y)
    Beta=np.array(Beta)
    n=min(X.shape[0],Y.shape[0])
    it=(detailed_mmd_rbf(X,Y,g) for i,g in enumerate(Gamma))
    '''
    next(it)[0]:mmd distance of rbf_kernel_i
    next(it)[1]:XX matrix of rbf_kernel_i
    next(it)[2]:XY matrix of rbf_kernel_i
    next(it)[3]:YY matrix of rbf_kernel_i
    '''
    mmds=[]
    XXs=[]
    XYs=[]
    YYs=[]
    for i,B in enumerate(Beta):
        temp=next(it)
        mmds.append(temp[0])
        XXs.append(temp[1])
        XYs.append(temp[2])
        YYs.append(temp[3])
    Q_sum=0.0
    for i,B in enumerate(Beta):
        tmp1 = (XXs[i] - mmds[i]) * (XXs[i] - mmds[i])
        tmp2 = (XYs[i] - mmds[i]) * (XYs[i] - mmds[i])
        tmp3 = (YYs[i] - mmds[i]) * (YYs[i] - mmds[i])
        tmp4 = tmp1.mean() + tmp3.mean() - 2 * tmp2.mean()
        Q_sum=Q_sum+tmp4*B*tmp4*B
    Q_sum=Q_sum/(X.shape[0]*Y.shape[0]-1)
    return Q_sum
def Estimate_Q_detail(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("estimate Q")
    X = np.array(X)
    Y = np.array(Y)
    Beta=np.array(Beta)
    n=min(X.shape[0],Y.shape[0])
    it=(detailed_mmd_rbf(X,Y,g) for i,g in enumerate(Gamma))
    '''
    next(it)[0]:mmd distance of rbf_kernel_i
    next(it)[1]:XX matrix of rbf_kernel_i
    next(it)[2]:XY matrix of rbf_kernel_i
    next(it)[3]:YY matrix of rbf_kernel_i
    '''
    mmds=[]
    XXs=[]
    XYs=[]
    YYs=[]
    for i,B in enumerate(Beta):
        temp=next(it)
        mmds.append(temp[0])
        XXs.append(temp[1])
        XYs.append(temp[2])
        YYs.append(temp[3])
    Q_sum=0.0
    ttt=[]
    for i,B in enumerate(Beta):
        tmp1 = (XXs[i] - mmds[i]) * (XXs[i] - mmds[i])
        tmp2 = (XYs[i] - mmds[i]) * (XYs[i] - mmds[i])
        tmp3 = (YYs[i] - mmds[i]) * (YYs[i] - mmds[i])
        tmp4 = tmp1.mean() + tmp3.mean() - 2 * tmp2.mean()
        ttt.append(tmp4)
        Q_sum=Q_sum+tmp4*B*tmp4*B
    Q_sum=Q_sum/(X.shape[0]*Y.shape[0]-1)
    return Q_sum,ttt
def d_Estimate_Q(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("estimate Q")
    X = np.array(X)
    Y = np.array(Y)
    Beta=np.array(Beta)
    n=min(X.shape[0],Y.shape[0])
    it=(detailed_mmd_rbf(X,Y,g) for i,g in enumerate(Gamma))
    '''
    next(it)[0]:mmd distance of rbf_kernel_i
    next(it)[1]:XX matrix of rbf_kernel_i
    next(it)[2]:XY matrix of rbf_kernel_i
    next(it)[3]:YY matrix of rbf_kernel_i
    '''
    mmds=[]
    XXs=[]
    XYs=[]
    YYs=[]
    for i,B in enumerate(Beta):
        temp=next(it)
        mmds.append(temp[0])
        XXs.append(temp[1])
        XYs.append(temp[2])
        YYs.append(temp[3])
    Q_sum=0.0
    for i,B in enumerate(Beta):
        tmp1 = (XXs[i] - mmds[i]) * (XXs[i] - mmds[i])
        tmp2 = (XYs[i] - mmds[i]) * (XYs[i] - mmds[i])
        tmp3 = (YYs[i] - mmds[i]) * (YYs[i] - mmds[i])
        tmp4 = tmp1.mean() + tmp3.mean() - 2 * tmp2.mean()
        Q_sum=Q_sum+tmp4*B*tmp4*B
    Q_sum=Q_sum/(X.shape[0]*Y.shape[0]-1)
    return Q_sum
def d_Estimate_BQ(X,Y,Beta=[1,1,1,1,1],Gamma=[0.01,0.1,1,10,100]):
    """MMD using multiple rbf (gaussian) kernel (i.e., k_i(x,y) = exp(-gamma_i * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
        Beta [B_1,B_2,,_B_n] -- conbination vector
        Gamma [g_1,g_2,,_g_n] -- set of gamma argument
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    print("estimate Q")
    X = np.array(X)
    Y = np.array(Y)
    Beta=np.array(Beta)
    n=min(X.shape[0],Y.shape[0])
    it=(detailed_mmd_rbf(X,Y,g) for i,g in enumerate(Gamma))
    '''
    next(it)[0]:mmd distance of rbf_kernel_i
    next(it)[1]:XX matrix of rbf_kernel_i
    next(it)[2]:XY matrix of rbf_kernel_i
    next(it)[3]:YY matrix of rbf_kernel_i
    '''
    mmds=[]
    XXs=[]
    XYs=[]
    YYs=[]
    for i,B in enumerate(Beta):
        temp=next(it)
        mmds.append(temp[0])
        XXs.append(temp[1])
        XYs.append(temp[2])
        YYs.append(temp[3])
    Q_sum=0.0
    for i,B in enumerate(Beta):
        tmp1 = (XXs[i] - mmds[i]) * (XXs[i] - mmds[i])
        tmp2 = (XYs[i] - mmds[i]) * (XYs[i] - mmds[i])
        tmp3 = (YYs[i] - mmds[i]) * (YYs[i] - mmds[i])
        tmp4 = tmp1.mean() + tmp3.mean() - 2 * tmp2.mean()
        Q_sum=Q_sum+tmp4*B*2
    Q_sum=Q_sum/(X.shape[0]*Y.shape[0]-1)
    return Q_sum
def nearest_neighbor(D):
    '''
    :param D: Distance matrix shape(n_sample,n_sample)
    :return: nearest neighbor matrix , sorted
    '''
    res=[]
    for i,D_i in enumerate(D):
        tmp1=D[i]
        tmp2=sorted(enumerate(tmp1),key=lambda x:x[1])
        res.append(tmp2)
    return res
def is_n_neighbor(n_neighbor_matrix,a,b,n=3):
    '''
    :param n_neighbor_matrix:
    :param a: index of a
    :param b: index of b
    :param n: n_neighbor
    :return: true:is n_neighbor
    '''
    index_b=n_neighbor_matrix[a].index(b)
    return index_b < (n-1)
def is_eachother_neoghbor(n_neighbor_matrix,a,b,n=3):
    '''
    :param n_neighbor_matrix:
    :param a: index of a
    :param b: index of b
    :param n: n_neighbor
    :return: true:is n_neighbor
    '''
    index_b = n_neighbor_matrix[a].index(b)
    index_a = n_neighbor_matrix[b].index(a)
    return index_b+index_a < 2*(n-1)
def discriminative_ablility_detail(X=[],W=[],Beta=[1],Gamma=[1]):
    print("dis_ability")
    X=np.array(X)
    W=np.array(W)
    n=X.shape[0]
    sum=0.0
    ts=[]
    for i,g in enumerate(Gamma):
        XX=metrics.pairwise.rbf_kernel(X, X, Gamma[i])#for sorting ,cal distance
        n_neighbor=nearest_neighbor(XX)
        for j in range(n):
            n_neighbor[j]=n_neighbor[j][::-1]
            for k in range(n):
                n_neighbor[j][k]=n_neighbor[j][k][0]
        H=np.zeros((n,n))
        for j in range(n):
            for k in range(n):
                if(is_eachother_neoghbor(n_neighbor,j,k,3)):
                    H[j][k]=XX[j][k]
                else:
                    H[j][k]=0
        D=np.diag([x.sum() for x in H.T])
        L=D-H
        S_L=(X.T.dot(L).dot(X))/n*n
        res1=np.trace(W.T.dot(S_L).dot(W))
        HH=XX-H
        DD=np.diag([x.sum() for x in HH.T])
        LL=DD-HH
        S_N=(X.T.dot(LL).dot(X))/n*n
        res2 = np.trace(W.T.dot(S_N).dot(W))
        t=res2/res1
        ts.append(t)
        sum=sum+Beta[i]*t
    return sum,ts
def discriminative_ablility(X=[],W=[],Beta=[1],Gamma=[1]):
    print("dis_ability")
    X=np.array(X)
    W=np.array(W)
    n=X.shape[0]
    sum=0.0
    ts=[]
    for i,g in enumerate(Gamma):
        XX=metrics.pairwise.rbf_kernel(X, X, Gamma[i])#for sorting ,cal distance
        n_neighbor=nearest_neighbor(XX)
        for j in range(n):
            n_neighbor[j]=n_neighbor[j][::-1]
            for k in range(n):
                n_neighbor[j][k]=n_neighbor[j][k][0]
        H=np.zeros((n,n))
        for j in range(n):
            for k in range(n):
                if(is_eachother_neoghbor(n_neighbor,j,k,3)):
                    H[j][k]=XX[j][k]
                else:
                    H[j][k]=0
        D=np.diag([x.sum() for x in H.T])
        L=D-H
        S_L=(X.T.dot(L).dot(X))/n*n
        res1=np.trace(W.T.dot(S_L).dot(W))
        HH=XX-H
        DD=np.diag([x.sum() for x in HH.T])
        LL=DD-HH
        S_N=(X.T.dot(LL).dot(X))/n*n
        res2 = np.trace(W.T.dot(S_N).dot(W))
        t=res2/res1
        ts.append(t)
        sum=sum+Beta[i]*t
    return sum
def d_discriminative_ablility(X=[],W=[],Beta=[1],Gamma=[1]):
    print("dis_ability")
    X=np.array(X)
    W=np.array(W)
    n=X.shape[0]
    sum=0.0
    ts=[]
    for i,g in enumerate(Gamma):
        XX=metrics.pairwise.rbf_kernel(X, X, Gamma[i])#for sorting ,cal distance
        n_neighbor=nearest_neighbor(XX)
        for j in range(n):
            n_neighbor[j]=n_neighbor[j][::-1]
            for k in range(n):
                n_neighbor[j][k]=n_neighbor[j][k][0]
        H=np.zeros((n,n))
        for j in range(n):
            for k in range(n):
                if(is_eachother_neoghbor(n_neighbor,j,k,3)):
                    H[j][k]=XX[j][k]
                else:
                    H[j][k]=0
        D=np.diag([x.sum() for x in H.T])
        L=D-H
        S_L=(X.T.dot(L).dot(X))/n*n
        res1=np.trace(W.T.dot(S_L).dot(W))
        HH=XX-H
        DD=np.diag([x.sum() for x in HH.T])
        LL=DD-HH
        S_N=(X.T.dot(LL).dot(X))/n*n
        res2 = np.trace(W.T.dot(S_N).dot(W))
        t=res2/res1
        ts.append(t)
        sum=sum+Beta[i]*t
    return sum
def random_sample(start,end,n):
    resultlist=random.sample(range(start,end),n)
    return resultlist
def Huber_regression(X,Y,W):

    Beta=[0.03 for i in range(33)]
    gamma=[math.exp(x/10) for x in random_sample(-70,70,33)]
    gamma.sort()

    d=1
def f(args):
    Beta=args["Beta"]
    gamma=args["gamma"]
    #gamma = [math.exp(x / 10) for x in random_sample(-70, 70, 33)]
    #gamma.sort()
    lmbda=args["lmbda"]
    mu=args["mu"]
    b=args["b"]
    X1=np.array(args["X1"])
    X2=np.array(args["X2"])
    Z1=np.array(args["Y1"])
    Z2=np.array(args["Y2"])
    W1=np.array(args["W1"])
    W2=np.array(args["W2"])
    a=Beta_Kernel_MMD(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    b=Estimate_Q(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    c=discriminative_ablility(X2,W2,Beta=Beta,Gamma=gamma)
    d=np.array([abs(float(x)-0.03) for x in Beta]).sum()
    return a+lmbda*b+mu/c+d
def gradient_of_Bk(args,k):
    Beta=args["Beta"]
    gamma=args["gamma"]
    lmbda=args["lmbda"]
    mu=args["mu"]
    b=args["b"]
    X1=np.array(args["X1"])
    X2=np.array(args["X2"])
    Z1=np.array(args["Y1"])
    Z2=np.array(args["Y2"])
    W1=np.array(args["W1"])
    W2=np.array(args["W2"])
    le=args["le"]
    a,de=Beta_Kernel_MMD_return_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    b,Qe=Estimate_Q_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    c,Te=discriminative_ablility_detail(X2,W2,Beta=Beta,Gamma=gamma)
    d=np.array([abs(float(x)-0.03) for x in Beta]).sum()
    f=a+lmbda*b+mu/c+d
    if abs(f-1/le)<0.1:
        return (f-1/le)*(de[k]+2*lmbda*Qe[k]-(mu/(c*c))*Te[k])
    else:
        return ((f-1/le)/abs((f-1/le)))*0.1*(de[k]+2*lmbda*Qe[k]-(mu/(c*c))*Te[k])
def gradient_of_b(args):
    Beta=args["Beta"]
    gamma=args["gamma"]
    lmbda=args["lmbda"]
    mu=args["mu"]
    b=args["b"]
    X1=np.array(args["X1"])
    X2=np.array(args["X2"])
    Z1=np.array(args["Y1"])
    Z2=np.array(args["Y2"])
    W1=np.array(args["W1"])
    W2=np.array(args["W2"])
    le=args["le"]
    a,de=Beta_Kernel_MMD_return_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    b,Qe=Estimate_Q_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    c,Te=discriminative_ablility_detail(X2,W2,Beta=Beta,Gamma=gamma)
    d=np.array([abs(float(x)-0.03) for x in Beta]).sum()
    f=a+lmbda*b+mu/c+d
    if abs(f-1/le)<0.1:
        return (f-1/le)
    else:
        return ((f-1/le)/abs((f-1/le)))*0.1
def gradient_of_lmbda(args):
    Beta=args["Beta"]
    gamma=args["gamma"]
    lmbda=args["lmbda"]
    mu=args["mu"]
    b=args["b"]
    X1=np.array(args["X1"])
    X2=np.array(args["X2"])
    Z1=np.array(args["Y1"])
    Z2=np.array(args["Y2"])
    W1=np.array(args["W1"])
    W2=np.array(args["W2"])
    le=args["le"]
    a,de=Beta_Kernel_MMD_return_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    b,Qe=Estimate_Q_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    c,Te=discriminative_ablility_detail(X2,W2,Beta=Beta,Gamma=gamma)
    d=np.array([abs(float(x)-0.03) for x in Beta]).sum()
    f=a+lmbda*b+mu/c+d
    if abs(f-1/le)<0.1:
        return (f-1/le)*b
    else:
        return ((f-1/le)/abs((f-1/le)))*0.1*b
def gradient_of_mu(args):
    Beta=args["Beta"]
    gamma=args["gamma"]
    lmbda=args["lmbda"]
    mu=args["mu"]
    b=args["b"]
    X1=np.array(args["X1"])
    X2=np.array(args["X2"])
    Z1=np.array(args["Y1"])
    Z2=np.array(args["Y2"])
    W1=np.array(args["W1"])
    W2=np.array(args["W2"])
    le=args["le"]
    a,de=Beta_Kernel_MMD_return_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    b,Qe=Estimate_Q_detail(X1.dot(W1),X2.dot(W2),Beta=Beta,Gamma=gamma)
    c,Te=discriminative_ablility_detail(X2,W2,Beta=Beta,Gamma=gamma)
    d=np.array([abs(float(x)-0.03) for x in Beta]).sum()
    f=a+lmbda*b+mu/c+d
    if abs(f-1/le)<0.1:
        return (f-1/le)*b*(1/c)
    else:
        return ((f-1/le)/abs((f-1/le)))*0.1*(1/c)
def qq(args):
    x, y = args
    return x ** 2 - 2 * x + 1 + y ** 2
def return_of_f(X1,X2,Z1,Z2,W1,W2,le):
    space = {"Beta":[random.random() for x in range(33)],
             "lmbda":random.random(),
             "mu":random.random(),
             "b":random.random(),
             "X1":X1,
             "Y1":Z1,
             "W1":W1,
             "X2":X2,
             "Y2":Z2,
             "W2":W2,
             "le":le
    }
    return f(space)
    #a=q(space)
    #best = fmin(q, space, algo=rand.suggest, max_evals=10)

    #print(best)
def estimate_Beta_return_of_f(X1,X2,Z1,Z2,W1,W2,le):
    gamma = [math.exp(x / 10) for x in random_sample(-70, 70, 33)]
    gamma.sort()
    space = {"Beta": [random.random() for x in range(33)],
             "gamma":gamma,
             "lmbda": random.random(),
             "mu": random.random(),
             "b": random.random(),
             "X1": X1,
             "Y1": Z1,
             "W1": W1,
             "X2": X2,
             "Y2": Z2,
             "W2": W2,
             "le":le
             }
    k = f(space)
    ess=[k]
    ggggg=0.1
    for i in range(100):
        print("round",i)
        dBk=[]
        bs=0
        for k,B in enumerate(space["Beta"]):
            dBk.append(gradient_of_Bk(space,k))
            bs=bs+B
        lmbda=space["lmbda"]
        b=space["b"]
        mu=space["mu"]
        d_lmbda=gradient_of_lmbda(space)
        db=gradient_of_b(space)
        dmu=gradient_of_mu(space)
        for k, B in enumerate(space["Beta"]):
            space["Beta"][k]=space["Beta"][k]-(dBk[k]+ggggg*(bs-1)/abs(bs-1))
            if space["Beta"][k]<0:
                space["Beta"][k]=0
        #space["lmbda"]=space["lmbda"]-(d_lmbda+ggggg*lmbda/abs(lmbda))
        space["b"]=space["b"]-(db+ggggg*b/abs(b))
        space["mu"]=space["mu"]-(dmu+ggggg*mu/abs(mu))
        k1 = f(space)
        ess.append(k1)
        if abs(k-k1)<0.01:
            break
    return space,ess,gamma


def Do_decent(e):
    X1 = []
    X2 = []
    for i, d in enumerate(e[1]):
        X1.append([float(x) for x in d[2:]])
    for i, d in enumerate(e[2]):
        X2.append([float(x) for x in d[2:]])
    X1 = np.array(X1)
    X2 = np.array(X2)
    label1 = np.array([1, 1, 1])
    label2 = np.array([1, 1, 1])
    label1 = np.array(label1)
    label2 = np.array(label2)
    d = 1
    W_init = np.random.rand(512,512)
    Beta = [0.03 for i in range(33)]
    gamma = [math.exp(x / 10) for x in random_sample(-70, 70, 33)]
    gamma.sort()
    d_W_1=d_Beta_Kernel_MMD(X1,X2,Beta,gamma,W_init)
    d_W2=d_Estimate_Q(X1,X2,Beta,gamma,W_init)


    '''
    bda = BDA(kernel_type='primal', dim=512, lamb=1, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)
    Z1, Z2, acc, ypre, list_acc = bda.fit_predict(X1, label1, X2, label2)
    '''
    return 1

def test():
    ''''''
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    print(a)
    print(b)
    print(mmd_linear(a, b))  # 6.0
    print(mmd_rbf(a, b))  # 0.5822
    print(mmd_poly(a, b))  # 2436.5
    print(Beta_Kernel_MMD(a,b,[1,1],[1,1]))
    xx=[[1,1],[1,-1],[-1,-1],[-1,1],[0,0]]
    qqq=metrics.pairwise.rbf_kernel(xx, xx, 0.01)
    print(qqq)
    print(nearest_neighbor(qqq))
    yy=[[2,2],[2,-2],[-2,-2],[-2,2],[0,0]]
    #discriminative_ablility(xx)
    zz=[[3,3],[3,-1],[-1,-1],[-1,3],[1,1]]
    yyy=[[2,2],[2,-2],[-2,-2],[-2,2],[0,0],[0,2.8],[2.8,0],[0,-2.8],[-2.8,0],[0,0]]
    yyyy=[[0,2.8],[2.8,0],[0,-2.8],[-2.8,0],[0,0]]
    yyyyy = [[100, 2.8], [102.8, 0], [100, -2.8], [97.2, 0], [100, 0]]
    discriminative_ablility(yyy+yyyyy,[[1,1],[0,0]])
    discriminative_ablility(yyy, [[1, 1], [0, 0]])

    print("-------*-----")
    print(mmd_rbf(xx,yy))
    print(mmd_rbf(yy,zz))
    print(mmd_rbf(xx, zz))
    print(mmd_rbf(yy, yyy))
    print(mmd_rbf(yyyy,yyyyy))
    print("------------")
    print(Beta_Kernel_MMD(xx, yy))
    print(Beta_Kernel_MMD(xx, zz))
    print(Beta_Kernel_MMD(yy, yy))
    print(Beta_Kernel_MMD(yy, zz))
    print(Beta_Kernel_MMD(yyyy, yyyyy))
    print("-----------")
    print(Estimate_Q(xx,yyy))
    print(Estimate_Q(yy,yyy))
    print(Estimate_Q(yy, yy))
    print(Estimate_Q(yy, zz))
    print(Estimate_Q(yyyy,yyyyy))

    d=1

'''
if __name__ == '__main__':
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    print(a)
    print(b)
    print(mmd_linear(a, b))  # 6.0
    print(mmd_rbf(a, b))  # 0.5822
    print(mmd_poly(a, b))  # 2436.5
'''
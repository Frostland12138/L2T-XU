# encoding=utf-8
"""
    Created on 9:52 2018/11/14
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm
import random
import matplotlib.pyplot as plt


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int),
                         np.ones(nb_target, dtype=int)))

    clf = svm.LinearSVC(random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


def estimate_mu(_X1, _Y1, _X2, _Y2):
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(1, C + 1):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c) / C
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0
    return mu


class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=20, mode='BDA', estimate_mu=False):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode
        self.estimate_mu = estimate_mu

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        #X=X.reshape(-1,1)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    Ns = len(Ys[np.where(Ys == c)])
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])

                    if self.mode == 'WBDA':
                        Ps = Ns / len(Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                        alpha = Pt / Ps
                        mu = 1
                    else:
                        alpha = 1

                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / Ns
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -alpha / Nt
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            # In BDA, mu can be set or automatically estimated using A-distance
            # In WBDA, we find that setting mu=1 is enough
            if self.estimate_mu and self.mode == 'BDA':
                if Xs_new is not None:
                    mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                else:
                    mu = 0
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot(
                [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('{} iteration [{}/{}]: Acc: {:.4f}'.format(self.mode, t + 1, self.T, acc))
        return Xs_new,Xt_new,acc, Y_tar_pseudo, list_acc
def draw_3D(x,y,z,x1,y1,z1):
    #########################绘图###############################
    ax = plt.figure().add_subplot(111, projection='3d')
    # 基于ax变量绘制三维图
    # xs表示x方向的变量
    # ys表示y方向的变量
    # zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
    # m表示点的形式，o是圆形的点，^是三角形（marker)
    # c表示颜色（color for short）
    xs = [1, 2, 3, 4, 5]
    ys = [2, 2, 2, 2, 2]
    zs = [3, 3, 3, 3, 3]
    ax.scatter(x,y,z, c='r', marker='.')  # 点为红色三角形
    ax.scatter(x1, y1, z1, c='b', marker='.')

    # 设置坐标轴
    ax.set_xlabel('environment')
    ax.set_ylabel('society')
    ax.set_zlabel('corporate governance')

    # 显示图像
    plt.show()

    ax = plt.figure().add_subplot(111, projection='3d')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
def Do_BDA_final(e):
    X1 = []
    X2 = []
    for i, d in enumerate(e[1]):
        X1.append([float(x) for x in d])
    for i, d in enumerate(e[2]):
        X2.append([float(x) for x in d])
    X1 = np.array(X1)
    X2 = np.array(X2)
    label1 = np.array([1, 1,1,1,1,1])
    label2 = np.array([1, 1,1,1,1,1])
    label1 = np.array(label1)
    label2 = np.array(label2)
    d = 1
    bda = BDA(kernel_type='primal', dim=512, lamb=1, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)
    Z1, Z2, acc, ypre, list_acc = bda.fit_predict(X1, label1, X2, label2)
    return X1, Z1, X2, Z2
def Do_BDA_e(e):
    X1=[]
    X2=[]
    for i,d in enumerate(e[1]):
        X1.append([float(x) for x in d[2:]])
    for i, d in enumerate(e[2]):
        X2.append([float(x) for x in d[2:]])
    X1=np.array(X1)
    X2=np.array(X2)
    label1=np.array([1,1,1])
    label2=np.array([1,1,1])
    label1 = np.array(label1)
    label2 = np.array(label2)
    d=1
    bda = BDA(kernel_type='primal', dim=512, lamb=1, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)
    Z1, Z2, acc, ypre, list_acc = bda.fit_predict(X1, label1, X2, label2)
    return X1,Z1,X2,Z2
def Do_BDA(split_data):
    x1=0
    y1=0
    z1=0
    x2=1
    y2=1
    z2=1
    X1=[]
    Y1=[]
    label1=[]
    X2=[]
    Y2=[]
    label2=[]
    for i in range(1000):
        a=random.random()-0.5
        b=random.random()-0.5
        c = random.random() - 0.5
        L=[]
        L.append(x1+a)
        L.append(y1+b)
        #L.append(z1 + c)
        X1.append(L)
        label1.append(1)
    for i in range(1000):
        a = random.random()-0.5
        b = random.random()-0.5
        c = random.random() - 0.5
        L=[]
        L.append(x2 + a*2)
        L.append(y2 + b*4)
        #L.append(z2 + c*6)
        X2.append(L)
        label2.append(1)
    X1=np.array(X1)
    X2 = np.array(X2)
    label1 = np.array(label1)
    label2 = np.array(label2)
    bda = BDA(kernel_type='primal', dim=2, lamb=1, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)
    Z1,Z2,acc, ypre, list_acc = bda.fit_predict(X1, label1, X2, label2)
    #draw_3D(X1[:, 0], X1[:, 1],X1[:, 2],X2[:, 0], X2[:, 1],X2[:, 2])
    plt.scatter(X1[:, 0], X1[:, 1], c='b',s=0.1)
    plt.scatter(X2[:, 0], X2[:, 1], c='g',s=0.5)
    plt.show()
    plt.scatter(Z1[:,0],Z1[:,1],c='b',s=0.1)
    plt.scatter(Z2[:,0],Z2[:,1],c='r',s=0.1)
    plt.show()
    return X1,X2,Z1,Z2
    d=1

if __name__ == '__main__':
    d=1
    '''
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    i, j = 0, 1  # Caltech -> Amazon
    src, tar = '../data/' + domains[i], '../data/' + domains[j]
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    bda = BDA(kernel_type='primal', dim=30, lamb=1, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)
    acc, ypre, list_acc = bda.fit_predict(Xs, Ys, Xt, Yt)
    print(acc)
    '''












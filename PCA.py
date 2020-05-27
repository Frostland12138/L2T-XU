import numpy as np
from sklearn.decomposition import PCA
def Do_PCA(e,k):
    X1 = []
    X2 = []
    for i, d in enumerate(e[1]):
        X1.append([float(x) for x in d[2:]])
    for i, d in enumerate(e[2]):
        X2.append([float(x) for x in d[2:]])
    X1 = np.array(X1)
    X2 = np.array(X2)
    pca = PCA(n_components=k)
    pca.fit(X1)
    Z1=pca.transform(X1)
    Z2=pca.transform(X2)
    return X1, Z1, X2, Z2
    d=1
def Do_PCA_final(e,k):
    X1 = []
    X2 = []
    for i, d in enumerate(e[1]):
        X1.append([float(x) for x in d])
    for i, d in enumerate(e[2]):
        X2.append([float(x) for x in d])
    X1 = np.array(X1)
    X2 = np.array(X2)
    pca = PCA(n_components=k)
    pca.fit(X1)
    Z1=pca.transform(X1)
    Z2=pca.transform(X2)
    return X1, Z1, X2, Z2
    d=1
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold

def draw_tsne(data):
    #model = json.load(open(emb_filename, 'r'))
    X = np.array(data[0])#n_sample*n_feature
    y = np.array(data[1])#n_samples*label
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print(X.shape)
    print(X_tsne.shape)
    print(y.shape)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    nums=[]
    '''
    for x in data[2][:-1]:
        for k in range(3):
            nums.append(x+k)
    '''
    nums=[]
    for i in range(X_norm.shape[0]):
        ''''''
        if i in nums:
            plt.text(X_norm[i, 0], X_norm[i, 1], 'x', color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 18})
        else:
            plt.text(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 18})

    plt.xticks([])
    plt.yticks([])
    plt.show()

def draw(data):

    #emb_filename = ("Algorithms-Test/show/show_DRML_AU12.json")
    draw_tsne(data)
    d=1

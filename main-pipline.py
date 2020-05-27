import os
import re
import math
import pandas
import numpy as np
import L2T
import Huber
import MMD
import BDA
import PCA
import random
import T_SNE
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
def write_csv(sql_data,filename):#TODO DONE√ 把列表写到CSV中
    #file_name = 'test1.csv'
    file_name=filename
    save = pandas.DataFrame(sql_data)
    save.to_csv(file_name,encoding='utf_8_sig')
    return 0
def read_in_csv(filename):#TODO DONE√ 读CSV中数据
    sql_data=pandas.read_csv(filename)
    sql_data=sql_data.values.tolist()
    return sql_data
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim
def load_faces():
    data = read_in_csv("./face_csv/total.csv")
    data = np.array(data)
    data = data[:-2]
    names = data[:, 1]
    attributes = [x.split('_') for x in names]
    names = [x[0] for x in attributes]
    names, ind = np.unique(names, return_index=True)
    names = names[np.argsort(ind)]
    split_data = []
    names_i = 0
    l = []
    for i, d in enumerate(data):
        if (attributes[i][0] == names[names_i]):
            l.append(d)
        else:
            names_i = names_i + 1
            split_data.append(l)
            l = []
            l.append(d)
    means = []
    split_data.append(l)
    print("spliting data")
    for i, split in enumerate(split_data):
        mean = []

        for j, d in enumerate(split):
            t = d[1].split("_")
            if str(t[1]).startswith("11"):
                if str(d[1]).endswith("00F"):
                    mean.append([float(x) for x in d[2:]])
                if str(d[1]).endswith("15L"):
                    mean.append([float(x) for x in d[2:]])
                if str(d[1]).endswith("15R"):
                    mean.append([float(x) for x in d[2:]])
        if (len(mean) == 0):
            for j, d in enumerate(split):
                if str(d[1]).endswith("00F"):
                    mean.append([float(x) for x in d[2:]])
                if str(d[1]).endswith("15L"):
                    mean.append([float(x) for x in d[2:]])
                if str(d[1]).endswith("15R"):
                    mean.append([float(x) for x in d[2:]])
        mean = np.array(mean)
        temp = mean.mean(axis=0)
        means.append(temp)
    return data,attributes,split_data
def test():
    print("--------start--------")
    X1, X2, Z1, Z2 = BDA.Do_BDA("1")
    Z = np.row_stack((Z1, Z2))
    X = np.row_stack((X1, X2))
    sample1 = random.sample(range(0, len(Z) - 1), len(X1))
    sample2 = random.sample(range(0, len(Z) - 1), len(X2))
    # Z1=np.array([Z[x] for x in sample1])
    # Z2 = np.array([Z[x] for x in sample2])
    # L2T.my_decomposition()
    W_1 = L2T.toy_example(X1, Z1)
    W_2 = L2T.toy_example(X2, Z2)
    W_3 = L2T.toy_example(X, Z)

    d=1
    T_SNE.draw(np.row_stack((X1,X2)))
    T_SNE.draw(np.row_stack((Z1,Z2)))
    # split_data=load_faces()
    #plt.scatter(X1[:, 0], X1[:, 1], c='b', s=0.1)
    # plt.scatter(Z1[:, 0], Z1[:, 1], c='r', s=0.1)
    #plt.scatter(X2[:, 0], X2[:, 1], c='g', s=0.1)
    # plt.scatter(Z2[:, 0], Z2[:, 1], c='y', s=0.1)
    plt.scatter(X1.dot(W_1)[:, 0], X1.dot(W_1)[:, 1], c='b', s=0.1)
    plt.scatter(X2.dot(W_2)[:, 0], X2.dot(W_2)[:, 1], c='g', s=0.1)
    plt.show()
    d = 1
    '''
    MMD.test()
    R=1
    #L2T.decomposition_example(R)
    L2T.my_decomposition()
    XX=np.array([[1,0,0,0,0],[0,2,3,0,0],[0,3,3,0,0],[0,0,0,4,0],[0,0,0,0,5]])
    print(XX)
    qr=la.qr(XX)
    print(qr[0].dot(qr[1]))
    lll=la.cholesky(XX)
    print(lll)
    print("***************")
    print(np.sqrt(XX).dot(np.sqrt(XX)))
    X=np.array([[1,2,3,4,5],[1,2,3,4,5]])
    print(X)
    print("-------------------")
    W=np.array([[1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4]])
    print(W)
    print("-------------------")
    Z=X.dot(W)
    print(Z)
    print("-------------------")
    Z_1 = np.linalg.pinv(X)
    Z_2=np.linalg.pinv(X.T)
    res=Z_1.dot(Z).dot(Z.T).dot(Z_2)
    print(res)
    print("-------------------")
    l, d, p = la.ldl(res)
    print(l)
    print("-------------------")
    print(d)
    print("-------------------")
    print(l.dot(d).dot(l.T))
    print("-------------------")
    e=np.exp(d)
    print(e)
    fin=l.dot(np.sqrt(d))
    print(fin)
    '''
def load_exp(dir):
    data=read_in_csv(dir+"experience_0_data.csv")
    dis = read_in_csv(dir + "experience_0_dis.csv")
    loss = read_in_csv(dir + "experience_0_loss.csv")
    names = read_in_csv(dir + "experience_0_names.csv")
    W = read_in_csv(dir + "experience_0_W.csv")
    return data,dis,loss,names,W
def toy_estimate():
    data, dis, loss, names, W = load_exp("./Exp_NEW/")
    X1 =[x[1:] for x in data[:3]]
    X2=[x[1:] for x in data[3:6]]
    Z1=[x[1:] for x in data[6:9]]
    Z2=[x[1:] for x in data[9:]]
    W1=[x[1:] for x in W[:512]]
    W2=[x[1:] for x in W[512:]]
    d=1
    le=np.array(dis[1][1:]).mean()/np.array(dis[0][1:]).mean()
    start1 = time.clock()
    space,ess,gamma = MMD.estimate_Beta_return_of_f(X1, X2, Z1, Z2, W1, W2,le)
    print(space)
    print(ess)
    write_csv(gamma,"gamma1.csv")
    write_csv(ess,"ess1.csv")
    write_csv(space["Beta"],"Beta1.csv")
    print(time.clock() - start1)
    #MMD.toy_extimater(X1,X2,Z1,Z2,W1,W2)
def load_f():
    Beta=read_in_csv("./Beta1.csv")
    Beta=[x[1] for x in Beta]
    for i,B in enumerate(Beta):
        if B<0:
            Beta[i]=0
    sum=np.array(Beta).sum()
    Beta=[x/sum for x in Beta]

    plt.plot(range(len(Beta)), Beta)
    plt.xlabel("K")
    plt.ylabel("Value of β_k")
    plt.show()
    d=1
def draw(a=[],b=[],c=[],d=[],namea="",nameb=""):
    aa=[min(a) for x in range(100)]
    bb=[min(b) for x in range(100)]
    cc = [min(c) for x in range(100)]
    dd = [min(d) for x in range(100)]
    for i,aaa in enumerate(a):
        aa[i]=aaa
    for j,bbb in enumerate(b):
        bb[j]=bbb
    for j,ccc in enumerate(c):
        cc[j]=ccc
    for j, ddd in enumerate(d):
        dd[j] = ddd
    fig, ax = plt.subplots()

    plt.xlabel('Iteration k')
    plt.ylabel('Loss Value')

    """set interval for y label"""
    #yticks = range(10, 110, 10)
    #ax.set_yticks(yticks)

    """set min and max value for axes"""
    #ax.set_ylim([10, 110])
    #ax.set_xlim([58, 42])


    x = [y for y in range(100)]
    plt.plot(x, aa, label="Normal")
    plt.plot(x, bb, label="With Q-Learning")
    plt.plot(x, cc, label="Linear combination initialization")
    plt.plot(x, dd, label="with Q-Learning and initialization")

    """open the grid"""
    plt.grid(True)

    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

    plt.show()
def draw_exp():
    names=read_in_csv("./result/names.csv")
    label=[]
    X1s=[x[1:] for x in read_in_csv("./result/X1s.csv")]
    for i,x in enumerate(X1s):
        label.append(0)

    X2s =[x[1:] for x in  read_in_csv("./result/X2s.csv")]
    for i,x in enumerate(X2s):
        label.append(1)
    Z1s =[x[1:] for x in  read_in_csv("./result/Z1s.csv")]
    Z2s =[x[1:] for x in  read_in_csv("./result/Z2s.csv")]
    U1s =[x[1:] for x in  read_in_csv("./result/U1s.csv")]
    U2s =[x[1:] for x in  read_in_csv("./result/U2s.csv")]
    UU1s =[x[1:] for x in  read_in_csv("./result/UU1s.csv")]
    UU2s =[x[1:] for x in  read_in_csv("./result/UU2s.csv")]
    Wrongs=[]
    for k in range(3,295):
        wrong=0
        _, PCA_1, _, PCA_2 = PCA.Do_PCA_final((names, X1s, X2s), k)
        for i in range(100):
            a=3*i
            b=a+1
            c=a+2
            vector_00f_1 = PCA_1[a]
            vector_00f_2 = PCA_1[b]
            vector_00f_3 = PCA_1[c]
            vector_dis_1 = PCA_2[a]
            vector_dis_2 = PCA_2[b]
            vector_dis_3 = PCA_2[c]
            vector_00f = [vector_00f_1, vector_00f_2, vector_00f_3]
            vector_dis = [vector_dis_1, vector_dis_2, vector_dis_3]
            for i,diss in enumerate(vector_dis):
                dis=[]
                for j,_00f in enumerate(vector_00f):
                    dis.append(cos_sim(diss,_00f))
                dis_res=np.argsort(dis)[::-1]
                if i in dis_res[:1]:
                    d=1
                else:
                    wrong=wrong+1
        Wrongs.append((k,wrong))
    print(Wrongs)


    T_SNE.draw((X1s+X2s,label))
    T_SNE.draw((Z1s + Z2s, label))
    T_SNE.draw((U1s + U2s, label))
    T_SNE.draw((UU1s + UU2s, label))
    d=1
def main():
    #T_SNE.draw("1")
    draw_exp()
    d=1
    #test()
    #load_f()
    #toy_estimate()
    #MMD.Huber_regression("","","")
    #W_experience=read_in_csv("./Experiences")

    data, attributes, split_data=load_faces()
    '''
    tt1=[float(x) for x in data[2276][2:]]
    tt2=[float(x) for x in data[2595][2:]]
    t1=[float(x) for x in data[2286][2:]]
    t2=[float(x) for x in data[2288][2:]]
    t3=[float(x) for x in data[2287][2:]]
    t4=[float(x) for x in data[2612][2:]]
    t5=[float(x) for x in data[2614][2:]]
    t6=[float(x) for x in data[2613][2:]]
    X1, ttt1, X2, ttt2 = BDA.Do_BDA_final((1,[tt1,tt1,tt1,tt2,tt2,tt2],[t1,t2,t3,t4,t5,t6]))

    d=1

    print(-math.log2(1 - cos_sim(ttt1[0],ttt2[0])))
    print(-math.log2(1 - cos_sim(ttt1[0], ttt2[1])))
    print(-math.log2(1 - cos_sim(ttt1[0], ttt2[2])))
    print(-math.log2(1 - cos_sim(ttt1[0], ttt2[3])))
    print(-math.log2(1 - cos_sim(ttt1[0], ttt2[4])))
    print(-math.log2(1 - cos_sim(ttt1[0], ttt2[5])))
    print(-math.log2(1 - cos_sim(ttt1[3], ttt2[0])))
    print(-math.log2(1 - cos_sim(ttt1[3], ttt2[1])))
    print(-math.log2(1 - cos_sim(ttt1[3], ttt2[2])))
    print(-math.log2(1 - cos_sim(ttt1[3], ttt2[3])))
    print(-math.log2(1 - cos_sim(ttt1[3], ttt2[4])))
    print(-math.log2(1 - cos_sim(ttt1[3], ttt2[5])))
    '''
    experience=[]

    label = []
    num=[0]
    data_x=[]
    '''
    for i in range(10):
        s1=split_data[i+65]
        s1=[[float(y) for y in z[2:]] for z in s1]
        for j,sss in enumerate(s1):
            label.append(i)
        num.append(len(label))
        data_x=data_x+s1

    data=(np.array(data_x),np.array(label),num)
    T_SNE.draw(data)
    '''
    d=1
    while 1:
        name=[]
        source=[]
        target=[]
        a=random.randint(0,234)#a 类人
        b=random.randint(0,234)#b 类人
        c=random.randint(0,234)#c 类人
        name.append(split_data[a][0][1])
        name.append(split_data[b][0][1])
        name.append(split_data[c][0][1])
        target_attribute_i=random.randint(0,len(split_data[a]))-1
        #a 类中随机一个人
        target_attribute=attributes[int(split_data[a][target_attribute_i][0])]
        t_a=target_attribute[1]+target_attribute[2]
        target.append(split_data[a][target_attribute_i])
        for i,d in enumerate(split_data[b]):
            attri=attributes[int(d[0])]
            attr=attri[1]+attri[2]
            if attr==t_a :
                target.append(d)
        for i,d in enumerate(split_data[c]):
            attri=attributes[int(d[0])]
            attr=attri[1]+attri[2]
            if attr==t_a :
                target.append(d)
        if len(target)!=3:
            continue
        if len(target)==3:
            source.append(split_data[a][0])
            source.append(split_data[b][0])
            source.append(split_data[c][0])

        if str(target[0][1]).endswith("_SU"):

            experience.append((name,source,target))
            print("experience adding ",len(experience))

        if len(experience)==10:
            break
    d=1
    #BDA.Do_BDA("w")
    BDA_res=[]
    i=0
    W_exp1=np.array([0])
    W_exp2=np.array([0])
    W_total_exp=np.array([0])
    name_total=[]
    X_total=[]
    Y_total=[]
    '''
    for i,e in enumerate(experience):
        for j in range(3):
            name_total.append(e[0][j])
            X_total.append(e[1][j])
            Y_total.append(e[2][j])
    for k in range(3,30):
        X1, Z1, X2, Z2 = PCA.Do_PCA((name_total,X_total,Y_total),k)
        dis1 = []
        dos_after = []
        for j, d in enumerate(X1):
            dis1.append(-math.log2(1 - cos_sim(d, X2[j])))
        for j, d in enumerate(Z1):
            dos_after.append(-math.log2(1 - cos_sim(d, Z2[j])))
        dis1 = np.array(dis1)
        print(k+1)
        print(dis1.mean())
        dos_after = np.array(dos_after)
        print(dos_after.mean())
    '''
    d = 1
    for i,e in enumerate(experience):
        #MMD.Do_decent(e)
        print("processing-----------",i)
        start1=time.clock()

        X1, Z1, X2, Z2 = BDA.Do_BDA_e(e)
        labele=[1,2,3,1,2,3]
        labele=np.array(labele)
        #L2T_S,L2T_T=MMD.Do_decent(e)
        print(time.clock() - start1)
        dis1=[]
        dos_after=[]
        for j,d in enumerate(X1):
            dis1.append(-math.log2(1-cos_sim(d,X2[j])))
        for j,d in enumerate(Z1):
            dos_after.append(-math.log2(1-cos_sim(d,Z2[j])))
        d=1
        b=1
        c=1
        start1 = time.clock()
        W_1,ess1=L2T.toy_example(X1,Z1,W_exp1)
        W_2,ess2=L2T.toy_example(X2,Z2,W_exp2)
        W_total,ess=L2T.toy_example(np.row_stack((X1,X2)),np.row_stack((Z1,Z2)),W_total_exp)
        #W_total2, ess2 = L2T.toy_example_Q_Learning(np.row_stack((X1, X2)), np.row_stack((Z1, Z2)), W_total_exp)
        U1=X1.dot(W_1)
        U2=X2.dot(W_2)
        UU1=X1.dot(W_total)
        UU2=X2.dot(W_total)
        a=np.row_stack((X1, X2))
        b=np.row_stack((Z1, Z2))
        c=np.row_stack((U1,U2))
        d=np.row_stack((UU1, UU2))
        e=np.row_stack((a,b))
        f=np.row_stack((c,d))
        g=np.row_stack((e,f))
        labele=[1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6]
        T_SNE.draw((g,labele))
        T_SNE.draw((np.row_stack((X1, X2)),labele))
        T_SNE.draw((np.row_stack((Z1, Z2)),labele))
        T_SNE.draw((np.row_stack((U1,U2)),labele))
        T_SNE.draw((np.row_stack((UU1, UU2)),labele))
        BDA_res.append((ess,ess2))
        print(time.clock() - start1)
        #W_exp1=W_1
        #W_exp2=W_2
        W_total_exp=W_total
        s1=np.zeros((200,1))
        s2=np.zeros((200,1))
        #for j in range(len(ess1)):
        #    s1[j]=ess1[j]
        #for j in range(len(ess2)):
        #    s1[j]=ess2[j]
        i=i+37
        tmp=np.array([-math.log2(1 - cos_sim(X1[i].dot(W_total), x.dot(W_total))) for i, x in enumerate(X2)]).mean()
        #tmp = np.array([-math.log2(1 - cos_sim(X1[i].dot(W_total2), x.dot(W_total2))) for i, x in enumerate(X2)]).mean()
        print(tmp)
        #start1 = time.clock()
        #space=MMD.estimate_Beta_return_of_f(X1,X2,Z1,Z2,W_1,W_2)
        #print(time.clock() - start1)

        #write_csv(np.array(e[1]+e[2]), "./Experiences/experience_" + str(i) + "_names.csv")
        #write_csv(np.row_stack((X1,X2,Z1,Z2)),"./Experiences/experience_"+str(i)+"_data.csv")
        #write_csv(np.row_stack((dis1,dos_after)), "./Experiences/experience_" + str(i) + "_dis.csv")
        #write_csv(np.row_stack((W_1,W_2)), "./Experiences/experience_" + str(i) + "_W.csv")
        #write_csv(np.row_stack((np.array(s1), np.array(s2))), "./Experiences/experience_" + str(i) + "_loss.csv")
        #BDA_res.append((X1, X2, Z1, Z2, dis1, dos_after,W_1,W_2,ess1,ess2))
        '''
        np.array([-math.log2(1-cos_sim(X1[i].dot(W_1),x.dot(W_2))) for i,x in enumerate(X2)]).mean()
        
        '''

    draw(BDA_res[0][0],BDA_res[0][1],BDA_res[1][0],BDA_res[1][1],"","")
    a=1
    d=1






    d=1
if __name__ == '__main__':
    print("--------start--------")
    main()

    d=1

    

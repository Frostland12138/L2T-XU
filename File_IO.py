import os
import re
import math
import pandas
import jieba
import tabula
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
from io import StringIO
from io import open
from pdfminer.converter import TextConverter
from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from skimage import io
def write_csv(sql_data,filename):#TODO DONE√ 把列表写到CSV中
    #file_name = 'test1.csv'
    file_name=filename
    save = pandas.DataFrame(sql_data)
    save.to_csv(file_name,encoding='utf_8_sig')
    return 0
def read_in_csv(filename):#TODO DONE√ 读CSV中表
    sql_data=pandas.read_csv(filename)
    sql_data=sql_data.values.tolist()
    return sql_data
def csv_name(file_dir):#获取所有csv文件名，返回列表
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            xx=os.path.splitext(file)
            if xx[1]=='.csv':
                L.append(xx)
            if xx[1]=='.CSV':
                L.append(xx)
    return L
def jpg_name(file_dir):#获取所有csv文件名，返回列表
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            xx=os.path.splitext(file)
            if xx[1]=='.jpg':
                L.append(xx)
            if xx[1]=='.JPG':
                L.append(xx)
    return L
def word_cut(str,mode):#mode1 返回迭代器  mode2 返回列表
    cut_generater=jieba.cut_for_search(str)
    if mode==1:
        return cut_generater
    elif mode==2:
        return [x for x in cut_generater]
    return 0
def convert_jpg(filename):
    '''
    :param filename: image file name
    :return: 1*(height*width) vector
    '''
    img = io.imread(filename)
    io.imshow(img)


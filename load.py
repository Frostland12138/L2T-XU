import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise load_modelValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)  # 通过checkpoint文件找到模型文件名
    if ckpt and ckpt.model_checkpoint_path:
        # ckpt.model_checkpoint_path表示模型存储的位置，不需要提供模型的名字，它回去查看checkpoint文件
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def main1():
    model_exp = "20180408-102900"
    meta_file, ckpt_file = get_model_filenames(model_exp)
    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)
    reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(model_exp, ckpt_file))
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("---------------------------------------")
    list = []
    for key in var_to_shape_map:
        list.append(key)
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))
    print("***************************************")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(),
                      os.path.join(model_exp, ckpt_file))
        print(tf.get_default_graph().get_tensor_by_name("input:0"))
        '''
        for i,name in enumerate(list):
            print(tf.get_default_graph().get_tensor_by_name(name+":0"))
        '''
def file_name(file_dir):#获取所有jpg文件名，返回列表
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            xx=os.path.splitext(file)
            if xx[1]=='.jpg':
                L.append(xx)
            if xx[1]=='.jpg':
                L.append(xx)
    return L

if __name__ == '__main__':
    main1()
'''
    sess=tf.Session()
    #先加载图和参数变量
    saver = tf.train.import_meta_graph('./20180408-102900/model-20180408-102900.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./20180408-102900'))


    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("Logits/weights:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict ={w1:13.0,w2:17.0}

    #接下来，访问你想要执行的op
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    print(sess.run(op_to_restore,feed_dict))
'''

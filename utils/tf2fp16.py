# coding:utf-8
import os
os.environ.__setitem__('CUDA_VISIBLE_DEVICES','0')

import numpy as np
import tensorflow as tf
import time

image_size = 224

def test1():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model_fp16/test_model.meta')
        saver.restore(sess,'./model_fp16/test_model')
        all_vars = tf.trainable_variables()
        for var in all_vars:
            print(var)
        input = tf.get_default_graph().get_tensor_by_name('images:0')
        output = tf.get_default_graph().get_tensor_by_name('pool5:0')
        feed_dict={input:np.ones([1,image_size,image_size,3],dtype=np.float16)}
        start = time.time()
        sess.run(output,feed_dict=feed_dict)
        end = time.time()
        print(end - start)

def test2():
    # saver = tf.train.import_meta_graph('./model/test.meta')
    from tensorflow.python import pywrap_tensorflow
    import numpy as np
    checkpoint_path = 'model_fp32/test'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    model = {}
    for key in var_to_shape_map:
        # print ("tensor_name", key)
        # sStr_2 = key[:-2]
        sStr_2 = key
        print sStr_2
        data = reader.get_tensor(key)
        data = data.astype(np.float16)
        if not model.has_key(sStr_2):
            model[sStr_2] = data
        else:
            model[sStr_2].append(data)

    np.save('name.npy', model)


def test3(fp16=True):
    sess = tf.Session()
    if fp16:
        tf.train.import_meta_graph('./model_fp16/test.meta')
        vars = tf.global_variables()
        mat = np.load('./name.npy').item()
        for var in vars:
            print var.name.split(":")[0]
            value = mat.get(var.name.split(":")[0])
            sess.run(var.assign(value.astype(np.float16)))
    else:
        saver = tf.train.import_meta_graph('./model_fp32/test_model.meta')
        saver.restore(sess, './model_fp32/test_model')

    input = tf.get_default_graph().get_tensor_by_name('images:0')
    output = tf.get_default_graph().get_tensor_by_name('pool5:0')
    total_time = 0
    step = 100
    for i in range(step):
        if fp16:
            feed_dict = {input: np.ones([1, image_size, image_size, 3], dtype=np.float16)}
        else:
            feed_dict = {input: np.ones([1, image_size, image_size, 3], dtype=np.float32)}

        start = time.time()
        sess.run(output, feed_dict=feed_dict)
        end = time.time()
        total_time += (end - start)
        print('%d epoch cost time : %f'%(i,end - start))
    print('cost total time : ',total_time)
    print('cost average time : ',total_time / step)

test3()





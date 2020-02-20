from mtcnn import MTCNN
from mtcnn.network.factory import NetworkFactory
import cv2
import json
import time
import os.path
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import keras
from tvm.contrib import graph_runtime
import numpy as np


img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
img_x = np.expand_dims(img, 0)
print(img_x.shape)
img_y = np.transpose(img_x, (0, 3, 1, 2))
print(img_y.shape)
#print(type(img_y.shape))
#detector = MTCNN()
#detection_result = []
"""
detection_result.append(detector.detect_faces(img))
detection_result.append(detection_result[0])
print(detection_result)
with open('../data/result/detected_face_pic.json', 'w', encoding='utf-8') as f:
    json.dump(detection_result, f, ensure_ascii=False, indent=4)
"""
weight_file = 'mtcnn/data/mtcnn_weights.npy'

network_factory = NetworkFactory()
p_net, r_net, o_net = network_factory.build_P_R_O_nets_from_file(weights_file=weight_file)

print("type of p_net")
print(type(p_net))
shape_dict = { 'input_1' : img_y.shape }
mod, params =  relay.frontend.from_keras(p_net, shape_dict)

target = 'cuda'
ctx = tvm.gpu(0)

with relay.build_config(opt_level=3):
    graph , lib , params = tvm.relay.build_module.build( mod , target='cuda',params=params )

m = graph_runtime.create(graph, lib, ctx)
m.set_input("input_1",value = img_y )
m.set_input(**params )

timer = 0
time = m.module.time_evaluator("run",ctx,number=1, repeat = 10)
prof_res = np.array( (time().results) )*1000
timer = timer + np.mean(prof_res)
#print(timer)
#m.run()
result = m.get_output(0)
result2 = m.get_output(1)
print(result)
print(result2)

"""
shape_dict = { 'input_1' : img_y.shape }
mod, params =  relay.frontend.from_keras(p_net, shape_dict)

target = 'cuda'
ctx = tvm.gpu(0)

with relay.build_config(opt_level=3):
    graph , lib , params = tvm.relay.build_module.build( mod , target='cuda',params=params )

m = graph_runtime.create(graph, lib, ctx)
m.set_input("input_1",value = img_y )
m.set_input(**params )

print("type of m")
print(type(m))
#type of m
#<class 'tvm.contrib.graph_runtime.GraphModule'>
timer = 0
time = m.module.time_evaluator("run",ctx,number=1, repeat = 10)
prof_res = np.array( (time().results) )*1000
timer = timer + np.mean(prof_res)
print(timer)

print(m.get_num_outputs())
result = m.get_output(0)

print(result.shape)

"""

'''
for pl in p_net.layers:
    print(pl.name)
    print(pl.input_shape)
print("------------------------------")
shape_dict = {'input_1': img_y.shape} 
print(shape_dict)
'''

"""
shape_dict = {
        'input_1': img_y.shape,
        'conv2d_1': (1, 275, 275, 32),        
        'conv2d_2': (1, 277, 277, 16),        
        'conv2d_3': (1, 279, 279, 10),        
        'conv2d_4': (1, 561, 561, 3),        
        'conv2d_5': (1, 275, 275, 32)
        }

input_1
(None, None, None, 3)
conv2d_1
(None, None, None, 3)
p_re_lu_1
(None, None, None, 10)
max_pooling2d_1
(None, None, None, 10)
conv2d_2
(None, None, None, 10)
p_re_lu_2
(None, None, None, 16)
conv2d_3
(None, None, None, 16)
p_re_lu_3
(None, None, None, 32)
conv2d_4
(None, None, None, 32)
conv2d_5
(None, None, None, 32)
softmax_1
(None, None, None, 2)
------------------------------
_convert_input_layer
input_1
(1, 3, 561, 561)
expr_name
input_1
_convert_input_layer
input_1
(1, 3, 561, 561)
================================
conv2d_1 (None, None, None, 3)
================================
Conv2D
weightList[0].shape
(3, 3, 3, 10)
================================
p_re_lu_1 (None, None, None, 10)
================================
PReLU
================================
max_pooling2d_1 (None, None, None, 10)
================================
MaxPooling2D
_convert_pooling called
MaxPooling2D
(2, 2)
(2, 2)
2 2 2
================================
conv2d_2 (None, None, None, 10)
================================
Conv2D
weightList[0].shape
(3, 3, 10, 16)
================================
p_re_lu_2 (None, None, None, 16)
================================
PReLU
================================
conv2d_3 (None, None, None, 16)
================================
Conv2D
weightList[0].shape
(3, 3, 16, 32)
================================
p_re_lu_3 (None, None, None, 32)
================================
PReLU
================================
conv2d_4 (None, None, None, 32)
================================
Conv2D
weightList[0].shape
(1, 1, 32, 2)
================================
conv2d_5 (None, None, None, 32)
================================
Conv2D
weightList[0].shape
(1, 1, 32, 4)
================================
softmax_1 (None, None, None, 2)
================================
Softmax
"""


"""
Cannot find config for target=cuda, workload=('conv2d', (1, 32, 275, 275, 'float32'), (4, 32, 1, 1, 'float32'), (1, 1), (0, 0), (1, 1), 'NCHW', 'float32'). A fallback configuration is used, which may bring great performance regression.
Cannot find config for target=cuda, workload=('conv2d', (1, 16, 277, 277, 'float32'), (32, 16, 3, 3, 'float32'), (1, 1), (0, 0), (1, 1), 'NCHW', 'float32'). A fallback configuration is used, which may bring great performance regression.
Cannot find config for target=cuda, workload=('conv2d', (1, 10, 279, 279, 'float32'), (16, 10, 3, 3, 'float32'), (1, 1), (0, 0), (1, 1), 'NCHW', 'float32'). A fallback configuration is used, which may bring great performance regression.
Cannot find config for target=cuda, workload=('conv2d', (1, 3, 561, 561, 'float32'), (10, 3, 3, 3, 'float32'), (1, 1), (0, 0), (1, 1), 'NCHW', 'float32'). A fallback configuration is used, which may bring great performance regression.
Cannot find config for target=cuda, workload=('conv2d', (1, 32, 275, 275, 'float32'), (2, 32, 1, 1, 'float32'), (1, 1), (0, 0), (1, 1), 'NCHW', 'float32'). A fallback configuration is used, which may bring great performance regression.
"""
'''
#shape_dict = {'input_1': (1, 3, 12, 12)}
#shape_dict = {'input_1': (1, 12, 12, 3)}
p_mod, p_params = relay.frontend.from_keras(p_net, shape_dict)
# compile the model
target = 'cuda'
print(p_mod)
#print(p_params)
ctx = tvm.gpu(0)
with relay.build_config(opt_level=3):
    graph_runtime.create(
    #p_executor = relay.build_module.create_executor('graph', p_mod, ctx, target)


dtype = 'float32'
#print(tvm.nd.array(img_y.astype(dtype)))
#print(p_params)


tvm_out = p_executor.evaluate()(tvm.nd.array(img_y.astype(dtype)), **p_params)
top1_tvm = np.argmax(tvm_out.asnumpy()[0])
'''

'''
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
# compile the model
target = 'cuda'
ctx = tvm.gpu(0)
with relay.build_config(opt_level=3):
    executor = relay.build_module.create_executor('graph', mod, ctx, target)

mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
# compile the model
target = 'cuda'
ctx = tvm.gpu(0)
with relay.build_config(opt_level=3):
    executor = relay.build_module.create_executor('graph', mod, ctx, target)

src_vid = '../data/vid/sample_parasite_10s.mp4'
vid_cap = cv2.VideoCapture(src_vid)
detector = MTCNN()

while (vid_cap.isOpened()):
    ret, frame = vid_cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_result.append(detector.detect_faces(img))
vid_cap.release()

start = time.time()

print("execution time tvm : ", time.time() - start)

with open('../data/result/detected_face_vid.json', 'w', encoding='utf-8') as f:
    json.dump(detection_result, f, ensure_ascii=False, indent=4)
'''

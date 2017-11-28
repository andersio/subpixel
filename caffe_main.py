#!/usr/bin/python
import caffe
import numpy as np
from scipy import misc as spmisc
from subpixel_numpy import _phase_shift
import sys
import os
import shutil
from glob import glob
from caffe_ps import unshuffle

def run_caffe():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    solver = caffe.get_solver('caffe_solver.prototxt')

    if len(sys.argv) >= 3 and os.path.exists(sys.argv[2]):
        solver.restore(sys.argv[2])

    solver.solve()

    loss = 0
    batch_size = solver.test_nets[0].blobs['lr_input'].shape[0]
    iterations = int(19962 / batch_size)

    for i in xrange(0, iterations):
        solver.test_nets[0].forward()
        loss += np.mean(solver.test_nets[0].blobs['tanh1_loss'].data) / iterations

    print("Euclidean Loss: %f" % (loss))

def run_caffe_inference():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net('caffe_subpxconv.prototxt', 'caffe_subpxconv_snapshot_iter_10000.caffemodel', caffe.TEST)
    net.forward()
    export(net.blobs['tanh1'].data)
    print("Euclidean Loss: %s" % net.blobs['tanh1_loss'].data)

def run_caffe_mps_export():
    net = caffe.Net('caffe_subpxconv.prototxt', 'caffe_subpxconv_snapshot_iter_10000.caffemodel', caffe.TEST)

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    WEIGHT = 0
    BIAS = 1

    # Caffe order: Out * In * k_y * k_x
    # MPS order:   Out * k_y * k_x * In (MPSCNNConvolution)
    # MPS order:   In * k_y * k_x * Out (MPSCNNConvolutionTranspose)
    # BNNS order:  k_width * k_height * In * Out

    with open(os.path.join(sys.argv[2], 'b_conv1'), 'w') as f:
        f.write(net.params['conv1'][BIAS].data.tobytes())
    with open(os.path.join(sys.argv[2], 'b_conv2'), 'w') as f:
        f.write(net.params['conv2'][BIAS].data.tobytes())
    with open(os.path.join(sys.argv[2], 'b_conv3'), 'w') as f:
        f.write(net.params['conv3'][BIAS].data.tobytes())
    with open(os.path.join(sys.argv[2], 'w_conv1'), 'w') as f:
        w_h0_mps = np.transpose(net.params['conv1'][WEIGHT].data, [1, 2, 3, 0])
        w_h0_mps = np.flip(np.flip(w_h0_mps, axis=1), axis=2)
        f.write(w_h0_mps.tobytes())
    with open(os.path.join(sys.argv[2], 'w_conv2'), 'w') as f:
        w_h1_mps = np.transpose(net.params['conv2'][WEIGHT].data, [1, 2, 3, 0])
        w_h1_mps = np.flip(np.flip(w_h1_mps, axis=1), axis=2)
        f.write(w_h1_mps.tobytes())
    with open(os.path.join(sys.argv[2], 'w_conv3'), 'w') as f:
        w_h2_mps = np.transpose(net.params['conv3'][WEIGHT].data, [1, 2, 3, 0])
        w_h2_mps = np.flip(np.flip(w_h2_mps, axis=1), axis=2)
        f.write(w_h2_mps.tobytes())
    with open(os.path.join(sys.argv[2], 'w_conv1_bnns'), 'w') as f:
        w_h0_mps = np.transpose(net.params['conv1'][WEIGHT].data, [3, 2, 1, 0])
        f.write(w_h0_mps.tobytes())
    with open(os.path.join(sys.argv[2], 'w_conv2_bnns'), 'w') as f:
        w_h1_mps = np.transpose(net.params['conv2'][WEIGHT].data, [3, 2, 1, 0])
        f.write(w_h1_mps.tobytes())
    with open(os.path.join(sys.argv[2], 'w_conv3_bnns'), 'w') as f:
        w_h2_mps = np.transpose(net.params['conv3'][WEIGHT].data, [3, 2, 1, 0])
        f.write(w_h2_mps.tobytes())

# data: N x 48 x 32 x 32
def export(data):
    if data.shape[1] != 48:
        raise Exception("Expecting only 48 feature channels. Got %d channels." % data.shape[1])

    # tanh1_img: 48N x 32 x 32
    tanh1_img = np.reshape(data, (data.shape[0] * 48, data.shape[2], data.shape[3]))

    # images: 3 x N x 16 x 32 x 32
    images = np.split(data, 3, axis=1)
    images = np.array([_phase_shift(channel_group, 4) for channel_group in images])
    images = np.transpose(images, (1, 2, 3, 0))
    images = np.flip(images, axis=3)

    export_path = sys.argv[2]
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    os.makedirs(export_path)

    for idx, img in enumerate(images):
        img = (img * 255).astype(np.uint8)
        spmisc.imsave(os.path.join(export_path, "%d_2.jpg" % idx), img)

    #for idx, img in enumerate(tanh1_img):
        #spmisc.imsave(os.path.join(export_path, "%d_%d.jpg" % (idx / 48, idx % 48)), img)

def run_test_shuffle_ops():
    if os.path.exists(sys.argv[3]):
        shutil.rmtree(sys.argv[3])
    os.makedirs(sys.argv[3])

    files = glob(os.path.join(sys.argv[2], "*.jpg"))

    for idx, path in enumerate(files):
        img = spmisc.imread(path)
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        img = unshuffle(img)    # 48-channel output
        
        img = np.split(np.expand_dims(img, axis=0), 3, axis=1)
        img = np.array([_phase_shift(channel_group, 4) for channel_group in img])
        img = (np.transpose(np.squeeze(img, axis=1), (1, 2, 0)) + 1.0) * 127.5

        spmisc.imsave(os.path.join(sys.argv[3], "%d.jpg" % idx), img)

        if idx == 63:
            break

if __name__ == '__main__':
    if sys.argv[1] == "train":
        run_caffe()
    elif sys.argv[1] == "infer":
        run_caffe_inference()
    elif sys.argv[1] == "export-mps":
        run_caffe_mps_export()
    elif sys.argv[1] == "test-shuffle-ops":
        run_test_shuffle_ops()
    else:
        print("Unknown action.")

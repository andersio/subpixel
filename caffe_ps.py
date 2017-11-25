import caffe
import numpy as np
import scipy
import multiprocessing as mp

class PeriodicUnshufflingLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.scaleFactor = params["scale"]
        self.pool = mp.Pool(processes=16)
        
    def reshape(self, bottom, top):
        # bottom: hr_ref            N x  3 x 128 x 128
        # top:    hr_ref_unshuffled N x 48 x 32  x 32
        batch_size = bottom[0].shape[0]
        out_c = bottom[0].shape[1] * 16
        top[0].reshape(batch_size, out_c, 32, 32)

    def forward(self, bottom, top):
        y_start = (bottom[0].data.shape[2] - 128) / 2
        x_start = (bottom[0].data.shape[3] - 128) / 2

        cropped = bottom[0].data[:, :, y_start:y_start+128, x_start:x_start+128]
        top[0].data[:] = self.pool.map(unshuffle, cropped)

    def backward(self, bottom, top):
        raise Exception("This layer is meant for forward passes only.")

def unshuffle(instance):
    # in:  3  x 128 x 128
    # out: 48 x 32  x 32
    scaleFactor = 4
    out_c_per_c = scaleFactor * scaleFactor
    
    # 32 x 32 x 48
    out = np.empty((instance.shape[1] / scaleFactor, instance.shape[2] / scaleFactor, out_c_per_c * instance.shape[0]))
    
    for i in xrange(0, instance.shape[0]):
        for y in xrange(0, out.shape[0]):
            for x in xrange(0, out.shape[1]):
                for ry in xrange(0, scaleFactor):
                        y_idx = i * out_c_per_c + ry * 4
                        out[y, x, y_idx:y_idx+scaleFactor] = instance[i, y * scaleFactor + ry, x*scaleFactor:(x+1)*scaleFactor]

    return np.transpose(out, (2, 0, 1))

    
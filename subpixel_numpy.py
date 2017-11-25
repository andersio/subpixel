import numpy as np

def _phase_shift(I, r):
    # I: N x 16 x 32 x 32
    if len(I.shape) is not 4:
        raise Exception("Unexpected shape.")

    out = np.empty((I.shape[0], I.shape[2] * r, I.shape[3] * r))

    for y in xrange(0, I.shape[2]):
        for x in xrange(0, I.shape[3]):
            for ry in xrange(0, r):
                out[:, y*r+ry, x*r:(x+1)*r] = I[:, ry*r:(ry+1)*r, y, x]

    return out
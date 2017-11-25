#!/usr/bin/python
import os
from glob import glob
import sys
from scipy import misc
import multiprocessing as mp

print(sys.argv)
files = sorted(glob(os.path.join(sys.argv[1], "*.jpg")))
files = [os.path.relpath(path, sys.argv[1]) for path in files]

if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])
if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

with open(sys.argv[4], 'w') as f:
    for file in files:
        f.write("%s %d\n" % (file, 0))

def processFile(file):
    org = os.path.join(sys.argv[1], file)
    hr = os.path.join(sys.argv[2], file)
    lr = os.path.join(sys.argv[3], file)
    val = misc.imread(org)
    h_start = (val.shape[0] - 128) / 2
    w_start = (val.shape[1] - 128) / 2
    cropped = val[h_start:h_start+128, w_start:w_start+128, :]
    misc.imsave(hr, cropped)
    misc.imsave(lr, misc.imresize(cropped, (32, 32)))

pool = mp.Pool(processes=16)
pool.map(processFile, files)
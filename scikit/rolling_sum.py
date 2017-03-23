import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
from skcuda import misc

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

a = np.arange(100000,dtype=np.float32)
b = np.array(rolling_window(a,5))
misc.init()	
dest_gpu = gpuarray.to_gpu(b)
c = misc.sum(dest_gpu,axis=1)



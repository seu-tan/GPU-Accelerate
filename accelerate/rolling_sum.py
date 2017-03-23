from __future__ import print_function
import numpy as np
import sys
from numba import guvectorize, cuda
import pandas as pd
import time

@guvectorize(['void(float32[:], float32[:])'],'(n)->()', target='cuda')
def row_sum(inp,out):
    temp = 0.
    for i in range(inp.shape[0]):
        temp += inp[i]
    out[0] = temp
    
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

x = np.arange(30000000,dtype=np.float32)
Out = np.array(rolling_window(x,5))                               
out = np.empty(29999996, dtype=np.float32)
t1 = time.time()
da = cuda.to_device(Out)
dev_out = cuda.to_device(out, copy=False)
#t1 = time.time()
row_sum(da,out=dev_out)
dev_out.copy_to_host(out)
t2=time.time()
print(out)
print(t2-t1)


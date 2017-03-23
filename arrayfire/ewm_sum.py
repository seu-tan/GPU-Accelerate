import numpy as np
import arrayfire as af
import time
af.set_backend('cuda')
def test():
  x1 =np.arange(2000)
  x = af.Array(x1.ctypes.data, x1.shape, x1.dtype.char)
  alpha = 2/3
  s_old = x[0]

  a = time.time()
  for i in range(1, 2000):
    s = alpha * x[i] + (1- alpha) * s_old
    s_old = s
  #time.sleep(1)
  b = time.time()
  print(b - a)
  return s
print(test())
#test()



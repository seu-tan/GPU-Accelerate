import pandas as pd
import numpy as np

a = np.arange(100000,dtype=np.float32)
b = pd.Series(a,dtype=np.float32)
c = b.rolling(5).sum()



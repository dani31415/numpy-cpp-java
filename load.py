import os
import sys

result = os.system("cd environment && make")
if result!=0:
    raise "Compilation error"
sys.path.insert(0, './environment')

import numpy as np
import spam

A=np.array([[1.0,2,3]])
print(A)
n=spam.ndim(A)
print(n)
n=spam.ndim(9)
print(n)
import os
import sys

result = os.system("cd environment && make")
if result!=0:
    raise "Compilation error"
sys.path.insert(0, './environment')

import numpy as np
import model

A=np.array([[1.0,2,3]])
# print(A)
n=model.ndim(A)
# print(n)
D=model.imageDims()
print(D)
I=model.image()
print(I)
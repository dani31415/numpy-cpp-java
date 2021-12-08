import ctypes
import pathlib
import os
# import numpy;

result = os.system("make")
print(result)
if result!=0:
    raise "Compilation error"

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = "spam.dll"
    c_lib = ctypes.CDLL(libname)

print(c_lib)
print("hi")
print(c_lib.system(9,6))
import numpy as np
res = np.ones((5,5), dtype="int16")
internal = np.zeros((3,3), dtype="int16")
internal[1,1] = 9
res[1:-1,1:-1] = internal

print(res)
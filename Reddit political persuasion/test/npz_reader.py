from numpy import load
import numpy as np


data = load('/xxxx')
data1 = load("/xxxx")

lst = data["arr_0"]
lst1 = data1["arr_0"]
k = np.subtract(lst, lst1)
print(np.count_nonzero(lst))
print(np.count_nonzero(lst1))

print(not k.any())
a = np.nonzero(k)
# print(set(a[0]))    # print(a[i+1])
print(set(a[1]))

i = 0
for j in range(len(a[0])):
    if i < 10:
        if a[1][j] > 30:
            print(a[0][j], a[1][j])
            print(lst.item((a[0][j], a[1][j])))
            print(lst1.item((a[0][j], a[1][j])))
            i+=1
    else:
        exit()


import numpy as np

arr1 = np.array([1,2,3,4,5])

for x in arr1:
    print(x)

arr2 = np.array([[1,2,3],[4,5,6]])

for x in arr2:
    print(x)
# [1 2 3]
# [4 5 6]
for x in arr2:
    for y in x:
        print(y)

# 1
# 2
# 3
# 4
# 5
# 6

arr3 = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

for x in arr3:
    for y in x:
        for z in y:
            print(z)

# using nditer

for x in np.nditer(arr3):
    print(x)
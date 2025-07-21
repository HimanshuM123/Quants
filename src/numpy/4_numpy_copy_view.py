import numpy as np

arr1 = np.array([1,2,3,4,5])

X = arr1.copy() # copies in original array

arr1[0]=67
print(arr1) #[67  2  3  4  5]
print(X) #[1 2 3 4 5]

arr2= np.array([10,20,30,40,50])
Y = arr2.view()

arr2[0]=90
#Make a view, change the original array, and display both arrays:

print(arr2) #[90 20 30 40 50]
print(Y)  #[90 20 30 40 50]
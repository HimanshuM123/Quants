import  numpy as np

#NumPy arrays have an attribute called shape that returns a tuple with each index having
# the number of corresponding elements.


arr1 = np.array([1,2,3,4,5])
print(arr1.shape) #(5,)

arr2 = np.array([[1,2,3],[4,5,6]])
print(arr2.shape) #(2, 3)

arr3= np.array(([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])) #(2, 2, 3)
print(arr3.shape)

arr4 = np.array([1,2,3,4,5], ndmin=5)
print(arr4) #[[[[[1 2 3 4 5]]]]]
print(arr4.shape) #(1, 1, 1, 1, 5)

#Reshape
#Convert the following 1-D array with 12 elements into a 2-D array.

arr5 =  np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newArr = arr5.reshape(4,3)
print(newArr)
#[[ 1  2  3]
# [ 4  5  6]
# [ 7  8  9]
# [10 11 12]]

arr6= np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newArr2 = arr6.reshape(2,3,2)
print(newArr2)

# [[[ 1  2]
#   [ 3  4]
#   [ 5  6]]
#
#  [[ 7  8]
#   [ 9 10]
#   [11 12]]]
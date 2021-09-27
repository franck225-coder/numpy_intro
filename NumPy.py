# Set up
import numpy as np

# we can set seed in order to have the same random value
seed = np.random.seed(seed=1234)

# scalar
x = np.array(6)
print("x: ",x)
print("x ndim: ",x.ndim) # number of dimensions
print("x shape: ",x.shape) # dimension
print("x size: ",x.size) # number of size
print("x dtype: ",x.dtype) # data type

# vector
x = np.array([2.4, 1.1, 1.7])
print("x: ",x)
print("x ndim: ",x.ndim) # number of dimensions
print("x shape: ",x.shape) # dimension
print("x size: ",x.size) # number of size
print("x dtype: ",x.dtype) # data type

# matrix
x = np.array([[2,5], [3,9]])
print("x ", x)
print("x ndim: ", x.ndim) # number of dimensions
print("x shape: ", x.shape) # dimension
print("x size: ", x.size) # size of elements
print("x dtype: ", x.dtype) # data type

# 3-D tensor
x = np.array([[[2,4,1], [5,1,4], [4,4,1]], [[7,9,5], [3,0,3], [5,5,7]]])
print("x: ", x)
print("x ndim: ", x.ndim) # number of dimensions
print("x shape: ", x.shape) # dimensions
print("x size: ", x.size) # size of elements
print("x dtype: ", x.dtype) # data type

# Functions
print("np.zeros((2,2)):\n", np.zeros((2,2)))
print("np.ones((2,2)):\n", np.ones((2,2)))
print("np.eye((2)):\n", np.eye(5)) # identity matrix

# indexing
x = np.array([1, 2, 3, 4, 5])
print("x: ",x)
print("first element: ", x[0])
x[0] = 0 # change the value of the elements
print("x: ",x)

# slicing
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print (x)
print("x in column 1: ", x[:, 1])
print("x in row 1: ", x[1, :])
print("x in last row ", x[-1]) # return last row
print("x rows 0,1 & cols 1,2: ", x[0:2, 1:3])

# Integer array indexing
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(x)
row_to_get = np.array([0, 1, 2])
print("row_to_get: ", row_to_get)
col_to_get = np.array([0, 2, 1])
print("col_to_get: ", col_to_get)
value_to_get = x[row_to_get, col_to_get]
print("value_to_get: ", value_to_get)

# Boolean array indexing
x = np.array([[1, 2], [3, 4], [5, 6]])
print ("x: ", x)
print ("x > 2: ", x>2)
print ("x[x > 2]: ", x[x>2])
print ("x[x != 0]: ", x[x != 0])
print ("[x != 0]: ", [x != 0])

# Arithmetic
# Basic math
x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[1,2], [3,4]], dtype=np.float64)
print("x + y: ", np.add(x,y))
print("x - y: ", np.subtract(x,y))
print("x * y: ", np.multiply(x,y))

# Dot product
a = np.array([[1,2,3], [4,5,6]], dtype=np.float64)
b = np.array([[7, 8], [9, 8], [11, 12]], dtype=np.float64)
c = a.dot(b)
print(f"{a.shape} . {b.shape} = {c.shape}")
print(c)
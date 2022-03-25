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
x = np.array([[[2,4], [5,1], [4,4]], [[7,9], [3,0], [5,5]]])
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
print("x in last column ", x[:,-1]) # return last row
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

# Axis operations
# Sum across a dimension
x = np.array([[1, 2], [3, 4]])
print(x)
print("sum all: ", np.sum(x))
print("sum accross rows: ", np.sum(x, axis=0))
print("sum accross column: ", np.sum(x, axis=1))

# Min/max
x = np.array([[1,2,3], [4,5,6], [2,1,12]])
print("min: ", x.min())
print("max: ", x.max())
print("max across row ", x.max(axis=0))
print("min across row ", x.min(axis=0))
print("max across column ", x.max(axis=1))
print("min across column ", x.min(axis=1))

# Broadcast
x = np.array([1,2]) # vector
y = np.array(3) # scalar
z = x+y
print(z)

# transpose
x = np.array([[1,2,3], [4,5,6]]) # 2x3 matrix
print("x: ",x)
print("x shape: ",x.shape)
y = np.transpose(x, (1,0)) # flip dimensions at indexe 0 and 1
print("y: ",y)
print("y shape: ",y.shape)

# Reshape
x = np.array([[1,2,3,4,5,6]])
print (x)
print ("x.shape: ", x.shape)
y = np.reshape(x,(3,2))
print ("y: ", y)
print ("y.shape: ", y.shape)
z = np.reshape(x, (2,-1))
print ("z: ", z)
print ("z.shape: ", z.shape)

# for instance
x = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
            [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
t = np.transpose(x, ((1,0,2)))
t = np.reshape(t, (t.shape[0],-1))
print(t.shape)
print(t)

# joining
x = np.random.random((2, 3))
print(x)
print(x.shape)

# concatenation
y = np.concatenate([x, x], axis=1)
print(y)
print(y.shape)

# stacking
z = np.stack([x, x], axis=0) # stack on new axis
print(z)
print(z.shape)

# Expanding / reducing
# Adding dimensions
x = np.array([[1,2,3],[4,5,6]])
print ("x: ", x)
print ("x.shape: ", x.shape)
y = np.expand_dims(x, 1)
print ("y: ", y)
print ("y.shape: ", y.shape) # notice extra set of brackets are added

# Removing dimensions
x = np.array([[[1,2,3]],[[4,5,6]]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
y = np.squeeze(x, 1) # squeeze dim 1
print ("y: \n", y)
print ("y.shape: ", y.shape)  # notice extra set of brackets are gone

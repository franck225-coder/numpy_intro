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
print("x ndim ", x.ndim) # number of dimensions
print("x shape ", x.shape) # dimension
print("x size ", x.size) # size of elements
print("x dtype ", x.dtype) # data type
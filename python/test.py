import numpy as np
import cnn_lenet as cnn
# [xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = cnn.load_mnist(fullset=True)
# w = np.zeros((0, 2), dtype=np.float32)
# a = np.arange(4).reshape((2,2))
# b = np.amax(a, axis=1)
# c = np.amax(a, axis=0)
# w = np.vstack((w, b));
# w = np.vstack((w, c));
# # w[w > 2] = 0
# w1 = np.zeros((2,3))
# w1[:, 0] = b
# w = np.array([[1,2,9],[7,5,1],[4,6,4]])
# a = np.argmax(w, axis=0)
a=np.arange(30)
b = a.reshape((2,3,5))
print a
print b

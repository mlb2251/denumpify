import numpy as np

# add new axis
def f1(x):
    return x[np.newaxis]

if __name__ == '__main__':
    print (f1(np.array([[1,2],[3,4]])))

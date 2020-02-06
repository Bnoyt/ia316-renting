import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def similarity(x,y):
    X = np.array(x)
    Y = np.array(y)
    
    a = np.mean(X*X)
    b = np.mean(Y*Y)
    
    if a == 0 or b == 0:
        return 0
    else:
        return np.mean(X*Y)/np.sqrt(a*b)

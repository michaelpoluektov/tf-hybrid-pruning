import numpy as np
import pickle

with open("resnet_l1_sparsity_only.pickle", "rb") as f:
    o = pickle.load(f)

print(o[3], o[0], o[15], o)

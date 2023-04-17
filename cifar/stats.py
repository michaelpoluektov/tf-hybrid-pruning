import pickle
import numpy as np

names = ["resnet_bn_finetune", "resnet_l1", "resnet_l2", "resnet_noreg"]

for n in names:
    f = open(f"model/{n}_decomp_only.pickle", "rb")
    data = pickle.load(f)
    print(data)

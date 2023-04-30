import pickle
from utils import get_decomp_weight, get_weight
from model import get_resnet
import tensorflow as tf
import numpy as np


def read_pickle(name):
    with open(name, "rb") as f:
        o = pickle.load(f)
    return o


base_model, model = get_resnet()
biasl = [
    l
    for l in base_model.layers
    if isinstance(l, tf.keras.layers.Conv2D) and l.kernel.shape[0] == 3
]
for name in ["resnet_l1", "resnet_l2", "resnet_bn_finetune"]:
    print(f"{name}" + "\n" + "=" * 20)
    decomps_ft = read_pickle(f"model/{name}_compress_and_val.pickle")
    tucker_ft = read_pickle(f"model/{name}_decomp_only.pickle")
    imps = []
    for (d, s), l, t in zip(decomps_ft, biasl, tucker_ft):
        dw = get_decomp_weight(l, [d, d]) / get_weight(l)
        impr = (t - dw) * 100
        if impr != 100 and impr > 0:
            imps.append(impr)
        else:
            imps.append(0)
        print(
            f"DECOMP: {d}: ({dw * 100:.2f}%), SPAR: {s:.2f}%, IMPROVEMENT: {impr:.2f}%"
        )
    imps = np.array(imps)
    print(
        f"MEAN IMPROVEMENT: {np.mean(imps):.2f}%, MEAN SPARSITY: {np.mean([a[1] for a in decomps_ft if a[1] != 100]):.2f}%"
    )
# with open("results_tucker.pickle", "wb") as f:
#     pickle.dump(i, f)

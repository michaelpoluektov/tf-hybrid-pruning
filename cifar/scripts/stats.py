import pickle
import numpy as np
from model import get_resnet
from dataset import get_dataset
from utils import get_whatif, get_decomp_weight, get_weight
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

names = ["resnet_bn_finetune", "resnet_l1", "resnet_l2", "resnet_noreg"]

# for n in names:
#     f = open(f"model/{n}_decomp_only.pickle", "rb")
#     data = pickle.load(f)
#     print(data)

SIZE = 224
base_model, model = get_resnet()
test_ds, val_ds = get_dataset(False, SIZE, 1, 16, True)
model.compile(metrics=["accuracy"])
_, base_accuracy = model.evaluate(test_ds)
biasl = [
    i
    for i, l in enumerate(base_model.layers)
    if (isinstance(l, Conv2D) and l.bias is not None)
]
bias3 = [i for i in biasl if base_model.layers[i].kernel.shape[0] == 3]

results = []

for n in names:
    model.load_weights(f"model/{n}.h5")
    for num, b in enumerate([0, 3, 15]):
        l = base_model.layers[bias3[b]]
        k = l.kernel.numpy()
        # o = k.shape[-1] // 64
        mse_lst = []
        acc_lst = []
        per_lst = []
        ws_lst = []

        for i in range(1, 99, 2):
            newk = np.zeros(k.shape)
            t = np.percentile(abs(k), 100 - i)
            mask = abs(k) > t
            newk[mask] = k[mask]
            # newk = get_whatif(k, rank=rank, spar=100)
            mse_lst.append(np.mean((k - newk).flatten() ** 2))
            ws_lst.append(i / 100)
            l.set_weights([newk, l.bias.numpy()])
            _, acc = model.evaluate(test_ds)
            l.set_weights([k, l.bias.numpy()])
            acc_lst.append(acc)
            sq = (k - newk) ** 2
            t = np.percentile(sq, 99)
            per_lst.append(np.mean(sq[sq > t]))
            # Compute and store the R2 score

        results.append((mse_lst, acc_lst, ws_lst, per_lst))
print(results)
with open("results.pickle", "wb") as f:
    pickle.dump(results, f)

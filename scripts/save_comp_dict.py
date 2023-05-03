import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import sys
from tqdm import tqdm

sys.path.append("../src")
from utils import get_weight
from dataset import get_dataset
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

PREFIX = "../models/keras/resnet"
FINETUNES = ["bn_finetune", "l1"]
STRUCTS = ["block_8x8", "channel", "filter", "unstruct"]
NUMS = ["0.02", "0.04", "0.06", "0.08", "0.10"]
BLOCKS = [
    "conv2_block1_2_conv",
    "conv2_block2_2_conv",
    "conv2_block3_2_conv",
    "conv3_block1_2_conv",
    "conv3_block2_2_conv",
    "conv3_block3_2_conv",
    "conv3_block4_2_conv",
    "conv4_block1_2_conv",
    "conv4_block2_2_conv",
    "conv4_block3_2_conv",
    "conv4_block4_2_conv",
    "conv4_block5_2_conv",
    "conv4_block6_2_conv",
    "conv5_block1_2_conv",
    "conv5_block2_2_conv",
    "conv5_block3_2_conv",
]

pbar = tqdm(total=len(FINETUNES) * len(STRUCTS) * len(NUMS))
total_dict = {}
test_ds, val_ds = get_dataset(False, 224, 1, 16, True)
for ft in FINETUNES:
    for st in STRUCTS:
        for n in NUMS:
            model_dict = {}
            pth = f"{PREFIX}_{ft}_{st}_{n}_fp.h5"
            model = tf.keras.models.load_model(pth, compile=False)
            lnames = [l.name for l in model.layers]
            for b in BLOCKS:
                decomp, spar = 0, 0
                if f"{b}_sp" in lnames:
                    k = model.get_layer(f"{b}_sp").kernel.numpy()
                    spar = 1 - np.count_nonzero(k == 0) / k.size
                if f"{b}_first" in lnames:
                    l = model.get_layer(f"{b}_first")
                    l2 = model.get_layer(f"{b}_last")
                    k = l.kernel.numpy()
                    k2 = l2.kernel.numpy()
                    rank = k.shape[-1]
                    cin = k.shape[-2]
                    cout = k2.shape[-1]
                    decomp = (9 * rank * rank + cin * rank + cout * rank) / (
                        9 * cin * cout
                    )
                if b in lnames:
                    decomp = 1

                model_dict[b] = (decomp, spar)
            total_dict[pth] = model_dict
            model.compile(metrics=["accuracy"])
            _, val_acc = model.evaluate(val_ds, verbose=0)
            _, test_acc = model.evaluate(test_ds, verbose=0)
            model_dict["test_acc"] = test_acc
            model_dict["val_acc"] = val_acc
            pbar.update(1)

with open("stats_dict_fp.pickle", "wb") as f:
    pickle.dump(total_dict, f)
print(total_dict)

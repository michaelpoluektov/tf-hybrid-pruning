import pickle
import subprocess
import numpy as np

with open("data/stats_dict.pickle", "rb") as f:
    stats = pickle.load(f)

with open("data/stats_dict_fp.pickle", "rb") as f:
    stats_fp = pickle.load(f)

with open("data/stats_dict_tucker.pickle", "rb") as f:
    stats_tucker = pickle.load(f)


def get_ps(s):
    if "block" in s:
        return "block"
    elif "channel" in s:
        return "channel"
    elif "filter" in s:
        return "filter"
    else:
        return "unstructured"


def get_input_path(s):
    if "bn" in s:
        return "../models/resnet_bn_finetune.h5"
    else:
        return "../models/resnet_l1.h5"


def get_output_path(s):
    return f"{s[:-3]}_fp.h5"


cs = np.array(
    [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
)
cs = cs / cs.sum()

for model in stats:
    s = 0
    ws = 0
    val_acc_1 = stats[model]["val_acc"]
    val_acc_2 = stats_fp[model[:-3] + "_fp.h5"]["val_acc"]
    for num, l in enumerate(stats[model]):
        if "conv" in l:
            s += stats[model][l][0]
            s += min(1, (stats[model][l][1]) * 1)
            ws += stats[model][l][0] * cs[num] + min(1, (stats[model][l][1])) * cs[num]

    s = s / 16
    print(
        f"{model.split('/')[-1]}: {s:.2f} ({ws:.2f}), accuracy FL: {val_acc_1:.3f}, accuracy FP: {val_acc_2:.3f}"
    )
    input_path = get_input_path(model)
    output_path = get_output_path(model)
    run_cmd = [
        "python",
        "compress.py",
        "--method",
        "fixed_params",
        "--pruning_structure",
        get_ps(model),
        "--compression_factor",
        str(s),
        "--block_size",
        "8,8",
        "--output_path",
        output_path,
        "--input_path",
        input_path,
        "--dataset_size",
        "10",
    ]
    # subprocess.run(run_cmd)

for model in stats_tucker:
    s = 0
    ws = 0
    val_acc_1 = stats_tucker[model]["val_acc"]
    for num, l in enumerate(stats_tucker[model]):
        if "conv" in l:
            s += stats_tucker[model][l][0]
            s += min(1, (stats_tucker[model][l][1]) * 4)
            ws += (
                stats_tucker[model][l][0] * cs[num]
                + min(1, (stats_tucker[model][l][1])) * cs[num]
            )

    s = s / 16
    print(f"{model.split('/')[-1]}: {s:.2f} ({ws:.2f}), accuracy FL: {val_acc_1:.3f}")

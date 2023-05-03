import pickle
import subprocess

with open("stats_dict.pickle", "rb") as f:
    stats = pickle.load(f)


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


for model in stats:
    s = 0
    for l in stats[model]:
        if "conv" in l:
            s += stats[model][l][0]
            s += min(1, (stats[model][l][1]) * 4)
    s = s / 16
    # print(f"{model}: {s:.2f}")
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
    subprocess.run(run_cmd)

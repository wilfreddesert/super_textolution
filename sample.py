import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def sample(net, device, dataset, cfg):
    lr, name = dataset[-1]
    print("ENTERED SAMPLE")
    scale = cfg.scale
    print(scale)
    t1 = time.time()
    lr = lr.unsqueeze(0).to(device)
    sr = net(lr, cfg.scale).detach().squeeze(0)
    lr = lr.squeeze(0)
    t2 = time.time()
    print("TESTING")

    sr_im_path = os.path.join(cfg.sample_dir, "{}".format(name.replace("LR", "SR")))
    static_path = os.path.join("./static", "{}".format(name.replace("LR", "SR")))
    save_image(sr, sr_im_path)
    save_image(sr, static_path)
    print(
        "Saved {} ({}x{} -> {}x{}, {:.3f}s)".format(
            sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2 - t1
        )
    )


def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=False, group=cfg.group, scale=cfg.scale)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    # Added map_location = torch.device('cpu') to use CPU for inference

    state_dict = torch.load(cfg.ckpt_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    dataset = TestDataset(cfg.test_data_dir, cfg.scale)

    print(len(dataset))
    sample(net, device, dataset, cfg)

import random

import torch
import numpy as np
from PIL import Image



def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_image_hash(image, hash_size):
    image_hash = image.resize((hash_size, hash_size), Image.Resampling.BILINEAR).convert("L")
    image_hash = np.array(image_hash, dtype=float).flatten()
    return image_hash


def masked_argmin(x, mask):
    valid_idxs = mask.nonzero()[0]
    return valid_idxs[x[valid_idxs].argmin()]

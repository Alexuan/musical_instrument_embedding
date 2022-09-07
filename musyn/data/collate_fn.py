import random

import numpy as np
import torch


def _pad(seq, max_len):
    seq = np.pad(seq, (0, max_len - len(seq)),
            mode='constant', constant_values=0)
    return seq


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, 0), (0, max_len - x.shape[1])],
            mode="constant", constant_values=0)
    return torch.Tensor(x)


def _pad_3d(x, max_len):
    x = np.pad(x, [(0, 0), (0, max_len - x.shape[1]), (0, 0)],
            mode="constant", constant_values=0)
    return x

def collate_fn(batch):
    """Create batch"""
    # TODO: hard code here, i am wondering how to inject param
    # min_len = 3*16000
    # max_len = 8*16000
    # chosen_len = random.randint(min_len, max_len)
    # chosen_len = 4 * 16000
    
    max_len = max([x['feat'].shape[1] for x in batch])
    a = torch.cat([_pad_2d(x['feat'], max_len) for x in batch], dim=0)
    feat = torch.FloatTensor(a)

    b = torch.stack([torch.Tensor([x['label_A']]) for x in batch])
    instr = b.long().squeeze()

    c = torch.stack([torch.Tensor([x['label_B']]) for x in batch])
    instr_fml = c.long().squeeze()

    # d = torch.stack([torch.Tensor([x['pitch']]) for x in batch])
    # pitch = d.long().squeeze()

    # e = torch.stack([torch.Tensor([x['velocity']]) for x in batch])
    # velocity = e.long().squeeze()

    # f = torch.stack([torch.Tensor([x['instr_src']]) for x in batch])
    # instr_src = f.long().squeeze()

    return {'feat': feat, 
            'label_A': instr, 'label_B': instr_fml,}
            # 'pitch': pitch, 'velocity': velocity, 
            # 'instr_src': instr_src}

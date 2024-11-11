from typing import Dict

import numpy as np
import pickle
import torch

from FundusDataset import FundusDataset
from transforms import get_transforms

def create_dataset_loader(split_fn: str, split_idx: int, dataset_root:str, class_map:Dict,  model_name: str, pretrain: bool, n_classes: int, train_batch_size: int,  test_batch_size: int, **kwargs):
    with open(split_fn, 'rb') as f:
        split_fn_obj = pickle.load(f)['data_split']
        split_fn_obj = split_fn_obj[split_idx]
        train_fns, val_fns, test_fns = split_fn_obj['train'], split_fn_obj['val'], split_fn_obj['test']
    print("Val Fns: {}, Test Fns: {}".format(len(val_fns), len(test_fns)))

    train_transforms, transforms = get_transforms(model_name, pretrain, **kwargs)
    train_dataset = FundusDataset(dataset_root, fn_set=set(train_fns), transform=train_transforms, class_map=class_map)
    val_dataset = FundusDataset(dataset_root, fn_set=set(val_fns), transform=transforms, class_map=class_map)
    test_dataset = FundusDataset(dataset_root, fn_set=set(test_fns), transform=transforms, class_map=class_map)
    assert len(test_dataset.classes) == len(val_dataset.classes) == n_classes

    train_shuffle = not kwargs['no_train_shuffle'] if 'no_train_shuffle' in kwargs else True
    if not train_shuffle: print("Train loader will not be shuffled.")
    if 'resampling' in kwargs and kwargs['resampling']:
        assert train_shuffle
        total_samples = len(train_dataset)
        train_ys = [x[1] for x in train_dataset.samples]
        class_counts = np.bincount(train_ys)
        class_weights = total_samples / (len(class_counts) * class_counts)
        print("resampling", class_weights)
        sample_weights = class_weights[train_ys]

        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=train_batch_size,
                sampler=sampler,
                num_workers=8)

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            num_workers=8)
   
    val_loader = torch.utils.data.DataLoader(val_dataset,
          batch_size=test_batch_size,
          shuffle=False,
          num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_dataset,
          batch_size=test_batch_size,
          shuffle=False,
          num_workers=8)
    return train_loader, val_loader, test_loader


def create_transform_dataset_loader(split_fn: str, split_idx: int, dataset_root:str, class_map:Dict, model_name: str, pretrain: bool, n_classes: int, batch_size: int, tran_type: str, tran_arg: float):
    with open(split_fn, 'rb') as f:
        split_fn_obj = pickle.load(f)['data_split']
        split_fn_obj = split_fn_obj[split_idx]
        val_fns, test_fns = split_fn_obj['val'], split_fn_obj['test']
    print("Val Fns: {}, Test Fns: {}".format(len(val_fns), len(test_fns)))

    _, transforms = get_transforms(model_name, pretrain, **{'tran_type_on_test': tran_type, 'tran_arg_on_test':tran_arg})
    print(transforms)

    val_dataset = FundusDataset(dataset_root, fn_set=set(val_fns), transform=transforms, class_map=class_map)
    test_dataset = FundusDataset(dataset_root, fn_set=set(test_fns), transform=transforms, class_map=class_map)
    assert len(test_dataset.classes) == len(val_dataset.classes) == n_classes

    val_loader = torch.utils.data.DataLoader(val_dataset,
          batch_size=batch_size,
          shuffle=False,
          num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_dataset,
          batch_size=batch_size,
          shuffle=False,
          num_workers=8)
    return val_loader, test_loader

def create_class_map(args):
    class_map = {i:i for i in range(11)}
    return class_map

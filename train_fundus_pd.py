from typing import Optional, Tuple

import argparse
import datetime
import numpy as np
import os
import random
import torch
import tqdm

from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from models import load_model 
from utils import create_class_map, create_dataset_loader


def train_epoch(model:torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        ep: int,
        optimizer: torch.optim.Optimizer,
        criterion,
        device:Optional[torch.device]=None,
        bayesian=False,
        log_int:Optional[int] = 100) -> Tuple:
    model.train()

    total_loss = 0
    total, mean_err = 0, 0

    for i, (xs, ys) in enumerate(tqdm.tqdm(loader, desc=f"Epoch {ep+1:03d}")):
        xs, ys = xs.to(device), ys.to(device)
        ys = t(ys)
        ys = torch.clamp(ys, min=1e-5)

        # bnn
        if bayesian:
            output_ = []
            kl_ = []
            for mc_run in range(1): # num_mc = 1 (default)
                output = model(xs)
                kl = get_kl_loss(model)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            loss = criterion(output, ys) + (kl / ys.shape[0])
        else:
            output = model(xs)
            loss = criterion(output, ys)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if criterion == nll_loss:
          mean_err += abs(output.mode - ys).sum() 
        else:
          mean_err += abs(output - ys).sum()
        total += ys.shape[0]

        if i % log_int == 0:
            tqdm.tqdm.write(f"\tLoss: {total_loss / (i+1):.4f}. {mean_err}/{total} = {mean_err/total:.2f}.") 


def eval_model(model:torch.nn.Module,
        loader:torch.utils.data.DataLoader,
        criterion,
        device:torch.device) -> Tuple:
    model.eval()
    total_loss = 0.0
    total, mean_err = 0, 0

    with torch.no_grad():
        for i, (xs, ys) in enumerate(tqdm.tqdm(loader)):
            xs, ys = xs.to(device), ys.to(device)
            ys = t(ys)
            ys = torch.clamp(ys, min=1e-5)

            output = model(xs)
            loss = criterion(output, ys)

            total_loss += loss.item()

            if criterion == nll_loss:
                mean_err += abs(output.mode - ys).sum() 
            else:
                mean_err += abs(output - ys).sum()
            total += ys.shape[0]

        print("finished")
        print(f"\tLoss: {total_loss / (len(loader)):.4f}. {mean_err}/{total} = {mean_err/total:.2f}")
    return total_loss / len(loader), mean_err / total


def t(x):
    return x.float()[..., None]


def nll_loss(dists, observations):
    return -dists.log_prob(observations).sum()


def train_model(model:torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        momentum: float,
        weight_decay: float,
        epochs: int,
        criterion, 
        optimizer,
        val_loader: Optional[torch.utils.data.DataLoader]=None,
        device: Optional[torch.device]=None,
        bayesian=False,
        start_ep: Optional[int] = 0) -> Tuple:

    train_losses = []
    val_best_err = float('inf')
    for ep in range(start_ep, epochs):
        train_loss = train_epoch(model, loader, ep, optimizer, criterion, device, bayesian=bayesian, log_int=len(loader)//4)
        train_losses.append(train_loss)

        if val_loader is not None:
            
            print("VAL")
            val_loss, val_mean_err = eval_model(model, val_loader, criterion, device)
            if val_best_err > val_mean_err:
                print("Better Val Mean Err: {:.2f} -> {:.2f}". format(val_best_err, val_mean_err))
                output_fn = os.path.join(args.output_dir, get_ckpt_fn(args, is_best=True))
                os.makedirs(args.output_dir, exist_ok=True)
                data = get_ckpt_data(args, model, optimizer, ep)
                torch.save(data, output_fn)
                print("Stored. {}.".format(output_fn))
                val_best_err = val_mean_err



def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    class_map = create_class_map(args)
    n_classes = len(set([x for x in class_map.values() if x >= 0]))

    print("SplitFN:", args.split_fn)

    train_loader, val_loader, test_loader = create_dataset_loader(args.split_fn, 
            args.split_idx, 
            args.dataset_root, 
            class_map,
            args.model,
            args.pretrain, 
            n_classes, 
            args.train_batch_size, 
            args.test_batch_size,
            resampling=args.resampling)

    model = load_model(args.model, pretrain=args.pretrain, n_labels = 1 if args.model in ['resnet18-b', 'resnet50-b'] else  n_classes, model_fn=args.pretrain_model_fn, device=device, bayesian=args.model.endswith('-b'), mean_bounded=args.mean_bounded, pd_type=args.pd_type, train_last_layer_only=args.train_last_layer_only)
    model = model.to(device)

    print("Number of Prameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    # Train
    if args.model in ['resnet18-b', 'resnet50-b']:
        criterion = torch.nn.MSELoss()
    else:
        criterion = nll_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, train_loader, args.lr, args.momentum, args.weight_decay, args.epochs, criterion, optimizer, val_loader, device)
    
    eval_model(model, test_loader, criterion, device)

    os.makedirs(args.output_dir, exist_ok=True)
    output_fn = os.path.join(args.output_dir, get_ckpt_fn(args, is_best=False))
    data = get_ckpt_data(args, model, optimizer, args.epochs -1)
    torch.save(data, output_fn)
    print(f"Stored {output_fn}.")
    print(datetime.datetime.now())


def get_ckpt_fn(args, is_best: bool):
    best_prefix = 'best' if is_best else 'last'
    return f'fundus_model_pd_{best_prefix}_seed{args.seed}.pt'


def get_ckpt_data(args, model, optimizer, ep):
    data = vars(args)
    data['model_weights'] = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    data['optimizer_state_dict'] = optimizer.state_dict()
    data['ep'] = ep
    data['class_map'] = create_class_map(args)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-root', default='~/Fundus-VA', type=str)
    parser.add_argument('--model', default='simple-pd', type=str,
            choices=['simple-pd', 'resnet18-pd', 'resnet50-pd', 'efficientnetv2-s-pd', 'resnet18-b', 'resnet50-b'])
    parser.add_argument('--split-fn', default='./data_split_fns_11cls.pkl', type=str)
    parser.add_argument('--split-idx', default=0, type=int)

    parser.add_argument('--train-batch-size', default=50, type=int)
    parser.add_argument('--test-batch-size', default=100, type=int)

    parser.add_argument('--resampling', default=False, action='store_true')
    parser.add_argument('--pretrain', default=None, choices=[None, 'V1', 'V2'])
    parser.add_argument('--pretrain-model-fn', default=None)

    parser.add_argument('--mean-bounded', default=False, action='store_true')

    parser.add_argument('--train-last-layer-only', default=False, action='store_true')

    parser.add_argument('--pd-type', default='Gaussian', choices=['Gaussian'])

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-3, type=float)
    parser.add_argument('--step-size', default=50, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--output-dir', default='./outputs', type=str)

    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    args.dataset_root = os.path.expanduser(args.dataset_root)

    main(args)

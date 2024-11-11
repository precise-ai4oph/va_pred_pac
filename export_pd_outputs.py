import argparse
import os
import pickle
import torch
import numpy as np


from models import load_model
from utils import create_dataset_loader
from train_fundus_pd import t, nll_loss


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def eval_model(model:torch.nn.Module,
        loader:torch.utils.data.DataLoader,
        criterion,
        device:torch.device):
    model.eval()
    total_loss = 0.0
    total, mean_err = 0, 0
    errs = [] 
    means = []
    scales = []
    all_ys = []
    
    with torch.no_grad():
        for i, (xs, ys) in enumerate(loader):
            print("\r{}/{} evaluating ...".format(i+1, len(loader)), end='')
            xs, ys = xs.to(device), ys.to(device)
            all_ys.append(ys)

            ys = t(ys)
            ys = torch.clamp(ys, min=1e-5)

            output = model(xs)
            loss = criterion(output, ys)

            total_loss += loss.item()

            mean_err += abs(output.mode - ys).sum()
            total += ys.shape[0]
            errs.append(abs(output.mode - ys))
            means.append(output.mean)
            scale = 1 / output.rate if model.pd_type == 'Gamma' else output.scale
            scales.append(scale)

    all_ys = torch.cat(all_ys)
    errs = torch.cat(errs).squeeze()
    means = torch.cat(means).squeeze()
    scales = torch.cat(scales).squeeze()
    
    print("finished")
    print(f"\tLoss: {total_loss / (len(loader)):.4f}. {mean_err}/{total} = {mean_err/total:.2f}")
    
    return total_loss / len(loader), mean_err / total, errs, means, scales, all_ys


def analyze_fn(name, fn, output_dir, device, blurred, kernel_size, store_result):
    set_seeds(100)
    print("Name:", name, "File:", fn)
    d = torch.load(fn)
    print("Epoch:", d['ep'])
    n_classes = len(set([x for x in d['class_map'].values() if x >= 0]))
    age_filter = d['age_filter'] if 'age_filter' in d else float('inf')
    print("age filter", age_filter)
    train_loader, val_loader, test_loader = create_dataset_loader(d['split_fn'],
                                                              d['split_idx'],
                                                              d['dataset_root'],
                                                              d['class_map'],
                                                              d['model'],
                                                              d['pretrain'],
                                                              n_classes,
                                                              d['train_batch_size'],
                                                              d['test_batch_size'],
                                                              age_filter=age_filter)
    
    # model
    model = load_model(d['model'], d['pretrain'], n_classes, model_fn=d['pretrain_model_fn'], device=device, mean_bounded=d['mean_bounded'] if 'mean_bounded' in d else False, pd_type=d['pd_type'])
    model.load_state_dict(d['model_weights'])
    model = model.to(device)
   
    if blurred:
        import torchvision
        existing_transforms = test_loader.dataset.transform.transforms[:]
        if type(existing_transforms[-1]) != torchvision.transforms.ToTensor:
            updated_transforms = existing_transforms[:-2] + [torchvision.transforms.GaussianBlur(kernel_size, 5.0)] + existing_transforms[-1:]
        else:
            updated_transforms = existing_transforms + [torchvision.transforms.GaussianBlur(kernel_size, 5.0)]
        test_loader.dataset.transform = torchvision.transforms.Compose(updated_transforms)
        print("existing transforms", existing_transforms)
        print("updated transforms", updated_transforms)


    val_loss, val_mean_err, val_errs, val_means, val_scales, val_ys = eval_model(model, val_loader, nll_loss, device)
    test_loss, test_mean_err, test_errs, test_means, test_scales, test_ys = eval_model(model, test_loader, nll_loss, device)

    print(val_scales)
    
    print('est. mean', test_mean_err)
    print('est. scale', test_scales.mean().item(), test_scales.std().item())

    if not store_result:
        print("not stored result")
        return
    
    fn = os.path.join(output_dir, 'output_data_{}.pkl'.format(name if not blurred else f"{name}_blurred{kernel_size}"))
    data = {'name': name,
           'fn': fn,
           'val_loss': val_loss,
           'val_mean_err': val_mean_err,
           'val_errs': val_errs,
           'val_means': val_means,
           'val_scales': val_scales,
           'val_ys': val_ys,
           'test_loss': test_loss,
           'test_mean_err': test_mean_err,
           'test_errs': test_errs,
           'test_means': test_means,
           'test_scales': test_scales,
           'test_ys': test_ys,
           'blurred': blurred,
           'kernel_size': kernel_size}
    
    with open(fn, 'wb') as f:
        pickle.dump(data, f)
    print('stored', fn)
    del model
    del d


def main(args):
  print(args)
  device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

  os.makedirs(args.output_dir, exist_ok=True)

  analyze_fn(args.name, args.fn, args.output_dir, device, args.blurred, args.kernel_size, args.store_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--fn', type=str)
    parser.add_argument('--blurred', default=False, action='store_true')
    parser.add_argument('--kernel-size', default=0, type=int)
    parser.add_argument('--store-result', default=False, action='store_true')
    parser.add_argument('--output-dir', default='analyze_outputs', type=str)

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    main(args)

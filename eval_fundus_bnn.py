import glob
import numpy as np
import os
import pickle
import torch
import tqdm

from models import load_model
from utils import create_dataset_loader


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def eval_fn(model_fn):
    n_monte_carlo = 50
    set_seeds(10)

    d = torch.load(model_fn)

    n_classes = len(set([x for x in d['class_map'].values() if x >= 0]))
    train_loader, val_loader, test_loader = create_dataset_loader(d['split_fn'],
            d['split_idx'],
            d['dataset_root'],
            d['class_map'],
            d['model'],
            d['pretrain'],
            n_classes,
            d['train_batch_size'],
            10,
            no_normalize=False,
            no_train_shuffle=True)

    print("Create model", d['model'], d['pretrain'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = load_model(d['model'], d['pretrain'], 1, model_fn=d['pretrain_model_fn'], device=device)
    model.load_state_dict(d['model_weights'])
 
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_outputs = []
        all_ys = []

        for idx, (xs, ys) in enumerate(tqdm.tqdm(test_loader)):
            xs, ys = xs.to(device), ys.to(device)

            output_mc = []
            for mc_run in range(n_monte_carlo):
                output = model.forward(xs)
                output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_ = output_.reshape(n_monte_carlo, -1, 1)
            output_ = output_.permute(1, 0, 2)  # Batch X Monte X Class
            all_outputs.append(output_)
            all_ys.append(ys)
               
        all_outputs = torch.cat(all_outputs, dim=0)
        all_ys = torch.cat(all_ys)

        preds = all_outputs.mean(dim=1).squeeze()

        print(preds[:10], all_ys[:10])
        
        err = (preds - all_ys).abs().mean()
        print("err", err)
    return all_outputs, preds, all_ys, err, {'model_archi': d['model'], 'seed': d['seed']}


def main(root_dir):
    for fn in glob.glob(root_dir + "/*best*.pt"):
        print(fn)

        all_outputs, preds, all_ys, err, fn_info = eval_fn(fn)

        data = {'fn': fn,
                'all_outputs': all_outputs,
                'preds': preds,
                'all_ys': all_ys,
                'err': err}
        
        output_fn = os.path.join(root_dir, 'result_{}_seed{}.pkl'.format(fn_info['model_archi'], fn_info['seed']))
        with open(output_fn, 'wb') as f:
            pickle.dump(data, f)
        print("stored", output_fn)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])


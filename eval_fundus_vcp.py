import glob
import numpy as np
import os
import pickle
import torch
import random


def set_seeds(seed):
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def vanilla_cp(val_means, val_ys, test_means, test_ys, alpha):
  val_diffs = (val_means - val_ys).abs()
    
  n = val_diffs.shape[0]
  q_level = np.ceil((n+1) * (1 - alpha)) / n

  q_hat = np.quantile(val_diffs, q_level, method='higher')

  lb = test_means - q_hat
  ub = test_means + q_hat

  lb = np.clip(lb, 0, 10)
  ub = np.clip(ub, 0, 10)

  coverage = (lb <= test_ys) & (test_ys <= ub)
  return coverage, lb, ub


def main(output_dir):
  set_seeds(100)

  val_means = {}
  val_scales = {}
  val_ys = {}

  test_errs = {}
  test_means = {}
  test_scales = {}
  test_ys = {}
  for fn in glob.glob(output_dir+"/output_data_*.pkl"):
    name = '_'.join(os.path.basename(fn)[:-4].split("_")[2:])
    print(fn, name)

    with open(fn, 'rb') as f:
      d = pickle.load(f)
      assert d['name'].startswith(name), "{} vs {}".format(name, d['name'])

      val_means[name] = d['val_means'].cpu()
      val_scales[name] = d['val_scales'].cpu()
      val_ys[name] = d['val_ys'].cpu()

      test_errs[name] = d['test_errs'].cpu()
      test_means[name] = d['test_means'].cpu()
      test_scales[name] = d['test_scales'].cpu()
      test_ys[name] = d['test_ys'].cpu()

 
  vanilla_cp_coverages = {}
  vanilla_cp_widths = {}
  for name in sorted(test_means):
    rep_name = "_".join(name.split("_")[:2])
    if rep_name not in vanilla_cp_coverages:
      vanilla_cp_coverages[rep_name] = []
      vanilla_cp_widths[rep_name ] = []
        
    coverage, lb, ub = vanilla_cp(val_means[name], val_ys[name], test_means[name], test_ys[name], 0.3)

    vanilla_cp_coverages[rep_name].append(coverage.float().mean().item())
    vanilla_cp_widths[rep_name].append( (ub-lb).mean().item())
    

  for name in vanilla_cp_coverages:
    print(name, np.mean(vanilla_cp_coverages[name]), np.std(vanilla_cp_coverages[name]))
    print("\t", np.mean(vanilla_cp_widths[name]), np.std(vanilla_cp_widths[name]))

if __name__ == '__main__':
  import sys
  main(sys.argv[1])

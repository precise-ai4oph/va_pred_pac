import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import scipy
import seaborn as sns
import sys
import torch


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def prediction_interval(mean_pred, std_pred, n_classes, z_score=1.96):
    lb = mean_pred - z_score * std_pred
    ub = mean_pred + z_score * std_pred
    lb = torch.clamp(lb, min=0)
    ub = torch.clamp(ub, max=n_classes-1)
    return lb, ub


def analyze_interval(lb, ub, ys):
    cov_mask = (lb <= ys) & (ys <= ub)
    coverage = (cov_mask).float().mean()
    width = (ub - lb).mean()
    int_cnt = (torch.floor(ub) - torch.ceil(lb) + 1).mean()
    return coverage.item(), width.item(), int_cnt.item()

def compute_beta_interval(means, std, ys, beta):
    lb = means - std * beta
    ub = means + std * beta

    idx = (lb <= ys) & (ys <= ub)
    return idx

def clopper_pearson_interval(n, k, alpha=0.05):
    lo = scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def find_thr(val_means, val_scales, val_ys, s, delta, gamma):
    found = False
    for b in range(0, 1000, s):
        b = b / 100
        idx = compute_beta_interval(val_means, val_scales, val_ys, b).float()
        n, s = idx.shape[0], idx.sum().item()
        lo, hi = clopper_pearson_interval(n, s, delta)
        if lo >= 1 - gamma:
            found = True
            break
    return b if found else -1

def group_res(arrays, group_maps, bounded):
    if bounded:
        int_group = [x for x in group_maps if not x.startswith('koa') and x.endswith('bounded')]
    else:
        int_group = [x for x in group_maps]
            
    group_res = [[] for _ in group_maps]
    for i, grp_name in enumerate(sorted(int_group)):
        for key in group_maps[grp_name]:
            group_res[i].append(arrays[key])
    return group_res


def check_pac_ps(val_means, val_scales, val_ys, test_means, test_scales, test_ys, s, delta, gamma):
    pac_coverages = {}
    pac_widths = {}
    pac_alphas = {}

    for name in sorted(test_means):
#        print(name)
        b = find_thr(val_means[name], val_scales[name], val_ys[name], s, delta, gamma)
        pac_alphas[name] = b
        if b == -1:
            print(name, "skipped")
            pac_coverages[name], pac_widths[name] = float('NaN'), float('NaN')
            continue
        lb, ub = prediction_interval(test_means[name], test_scales[name], n_classes=11, z_score=b)

        cov_cor = (lb <= test_ys[name]) & (test_ys[name] <= ub)
        idx = ~cov_cor
        pac_coverages[name], pac_widths[name], _ =  analyze_interval(lb, ub, test_ys[name])
    return pac_coverages, pac_widths


def plot_grp_analysis(grp_keys, grp_covs, grp_widths, delta):
    from matplotlib.ticker import PercentFormatter
    ## grp_covs and grp_widths are dictionory of different gammas. Each gamma has same format with grp_maes in plot_grp_maes

    grp_covs_sorted = {}
    grp_widths_sorted = {}
    for key in grp_covs:
        _, grp_covs_sorted[key] = _re_order_grp_vals(grp_keys, grp_covs[key])
    for key in grp_widths:
        grp_keys_sorted, grp_widths_sorted[key] = _re_order_grp_vals(grp_keys, grp_widths[key])
    grp_keys = grp_keys_sorted

    grp_covs_all = []
    grp_widths_all = []
    
    grp_keys_all = []

    for gamma in sorted(grp_covs):
        grp_covs_all += grp_covs_sorted[gamma]
        grp_widths_all += grp_widths_sorted[gamma]

        grp_keys_all += [x+"_G{}".format(gamma) for x in grp_keys]


    plt.figure()
    sns.boxplot(grp_covs_all)
    plt.xticks(np.arange(len(grp_keys_all)), grp_keys_all, rotation=45)
    for gamma in grp_covs:
        plt.hlines(y=(1-gamma), xmin=-0.5, xmax=len(grp_keys_all), colors='r', linestyles='dashed')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.tight_layout()
    fig_dir = 'figs'
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'fig_d{}_cov.pdf'.format(delta)))


    plt.figure()
    sns.boxplot(grp_widths_all)
    plt.xticks(np.arange(len(grp_keys_all)), grp_keys_all, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_d{}_width.pdf'.format(delta)))

def plot_grp_analysis_separate(grp_keys, grp_covs, grp_widths, delta):
    from matplotlib.ticker import PercentFormatter

    grp_covs_sorted = {}
    grp_widths_sorted = {}

    for gamma in grp_covs:
        grp_keys_sorted, grp_covs_sorted[gamma] = _re_order_grp_vals(grp_keys, grp_covs[gamma])
        grp_keys_sorted, grp_widths_sorted[gamma] = _re_order_grp_vals(grp_keys, grp_widths[gamma])

        plt.figure()
        sns.boxplot(grp_covs_sorted[gamma])
        plt.xticks(np.arange(len(grp_keys_sorted)), grp_keys_sorted, rotation=45)
        plt.hlines(y=(1-gamma), xmin=-0.5, xmax=len(grp_keys_sorted), colors='r', linestyles='dashed')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.tight_layout()
        fig_dir = 'figs'
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, 'fig_d{}_g{}_cov.pdf'.format(delta, gamma)))

        plt.figure()
        sns.boxplot(grp_widths_sorted[gamma])
        plt.xticks(np.arange(len(grp_keys_sorted)), grp_keys_sorted, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_d{}_g{}_width.pdf'.format(delta, gamma)))



def _re_order_grp_vals(cur_grp_keys, cur_grp_vals):
#    correct_order = ['Simple-CNN', 'ResNet18', 'ResNet50', 'EfficientNetV2-S', 'RetFound']
    correct_order = ['Simple-CNN', 'ResNet18', 'ResNet50', 'EfficientNetV2-S']
    cur_mapped = {key: val for key, val in zip(cur_grp_keys, cur_grp_vals)}
    sorted_val = [cur_mapped[key] for key in correct_order]
    return correct_order, sorted_val


def plot_grp_maes(grp_keys, grp_maes):
    plt.figure()
    grp_keys, grp_maes = _re_order_grp_vals(grp_keys, grp_maes)
    print(grp_keys)
    print(len(grp_maes))
    for m in grp_maes:
        print('d', m)
    sns.boxplot(grp_maes)
    plt.xticks(np.arange(len(grp_keys)), grp_keys, rotation=45)
    plt.tight_layout()
    
    fig_dir = 'figs'
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'fig_grp_maes.pdf'))


def main(output_dir):
    print(output_dir)
    set_seeds(100)

    val_means = {}
    val_scales = {}
    val_ys = {}

    test_errs = {}
    test_means = {}
    test_scales = {}
    test_ys = {}
    for fn in glob.glob(output_dir+"/output_data_*.pkl"):
        if "UCVA" in fn: continue
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

    # check 
    for name in sorted(val_means):
        vm = val_means[name]
        tm = test_means[name]            

        vs = val_scales[name]
        ts = test_scales[name]
        print(name, (vm<0).sum().item(), (vm>10).sum().item(), (tm<0).sum().item(), (tm>10).sum().item(), (vs < 0.01).sum().item(), (ts<0.01).sum().item())

        val_means[name][vm<0] = 0
        val_means[name][vm>10] = 10
        test_means[name][tm<0] = 0
        test_means[name][tm>10] = 10

    name_map = {'efficientnetv2s-pd': 'EfficientNetV2-S', 'resnet18_pd': 'ResNet18', 'resnet50_pd': 'ResNet50', 'simple_pd': 'Simple-CNN', 'retfound_pd': 'RetFound', 'efficientnetv2s_pd': 'EfficientNetV2-S'}

    # group analysis
    group_maps = {'Simple-CNN':[], 'ResNet18': [], 'ResNet50': [], 'EfficientNetV2-S':[]}
    for key in test_means:
        if 'seed' not in key:
            continue                       

        grp_name = "_".join(key.split("_")[:-1])
        grp_name = name_map[grp_name]
        print(grp_name)
        if grp_name not in group_maps:
            group_maps[grp_name] = []

        group_maps[grp_name].append(key)

    print(group_maps.keys())
    for key in group_maps:
        group_maps[key] = sorted(group_maps[key])

    bounded = False
    if bounded:
        grp_analysis_keys = [x.replace("BCVA_", "") for x in sorted(group_maps.keys()) if not x.startswith('koa') and x.endswith('bounded')]
    else:
        grp_analysis_keys = [x.replace("BCVA_", "") for x in sorted(group_maps.keys())]

    # groupwise MAEs
    maes = {}
    for grp_name in group_maps:
        maes[grp_name] = []
        for mdl in sorted(group_maps[grp_name]):
            maes[grp_name].append(abs(test_means[mdl] - test_ys[mdl]).mean().item())

    for grp_name in maes:
        print("MAE {}: Mean {:.2f}, Std {:.2f}".format(grp_name, np.mean(maes[grp_name]), np.std(maes[grp_name])))
    grp_maes = [maes[x] for x in sorted(group_maps)]
    print(grp_maes)
    plot_grp_maes(grp_analysis_keys, grp_maes)


    # paci
    s       = 1
    delta   = 1e-5

    bounded = False
    if bounded:
        grp_analysis_keys = [x.replace("BCVA_", "") for x in sorted(group_maps.keys()) if not x.startswith('koa') and x.endswith('bounded')]
    else:
        grp_analysis_keys = [x.replace("BCVA_", "") for x in sorted(group_maps.keys())]


    grp_pac_cov_gammas = {}
    grp_pac_widths_gammas = {}
    for gamma in [0.10, 0.20, 0.30, 0.35, 0.40]:
        pac_coverages, pac_widths = check_pac_ps(val_means, val_scales, val_ys, test_means, test_scales, test_ys, s, delta, gamma)
        print("Step: {}, Gamma: {}, Delta: {}".format(s, gamma,delta), "="* 90)
        print("violation cases")
        vio_case = 0
        for name in pac_coverages:
            if pac_coverages[name] < 1 - gamma or np.isnan(pac_coverages[name]):
                print(name)
                print("\t", pac_coverages[name], pac_widths[name])
                vio_case += 1
        print("No Violation Case" if vio_case == 0 else "Violation case: {}".format(vio_case))
        print("="* 90)

        group_pac_coverages, group_pac_widths = group_res(pac_coverages, group_maps, bounded), group_res(pac_widths, group_maps, bounded)
        for name, covs, wids in zip(grp_analysis_keys, group_pac_coverages, group_pac_widths):
            print(name)
            print("\tCoverage. Mean: {:.2f} %, Std: {:.2f} %".format(np.mean(covs)*100, np.std(covs)*100))
            print("\tWidth. Mean: {:.2f}, Std: {:.2f}".format(np.mean(wids), np.std(wids)))
        print("="* 90)

        print(grp_analysis_keys)
        grp_pac_cov_gammas[gamma] = group_pac_coverages
        grp_pac_widths_gammas[gamma] = group_pac_widths

    plot_grp_analysis(grp_analysis_keys, grp_pac_cov_gammas, grp_pac_widths_gammas, delta)
    plot_grp_analysis_separate(grp_analysis_keys, grp_pac_cov_gammas, grp_pac_widths_gammas, delta)


    # based on only means - GNU 4 class 
    def compute_4cls_indices(ys):
        return {"A": ys == 0, 
                "B": (1 <= ys) & (ys <= 2), 
                "C": (3 <= ys) & (ys <= 7), 
                "D": (8 <= ys) & (ys <= 10)}
    def compare_mean_inc_4class(preds, ys):
        indices = compute_4cls_indices(ys)

        cor = torch.zeros_like(ys, dtype=bool)

        cor[indices["A"]] = preds[indices["A"]] <= 0    
        cor[indices["B"]] = (1 <= preds[indices["B"]]) & (preds[indices["B"]] <= 2)
        cor[indices["C"]] = (3 <= preds[indices["C"]]) & (preds[indices["C"]] <= 7)
        cor[indices["D"]] = (8 <= preds[indices["D"]]) & (preds[indices["D"]] <= 10)

        cw_accs = ([(cor[v].float().mean().item() * 100) for _, v in indices.items()])
        return cw_accs

    def analyze_mean_inc_4class(means, ys, roundup=False):
        mean_inc_4class_cw_accs = {}
        for name in means:
            mean_inc_4class_cw_accs[name] = compare_mean_inc_4class(means[name] if not roundup else torch.round(means[name]), ys[name])

        grp_mean_inc_4class_cw_accs = group_res(mean_inc_4class_cw_accs, group_maps, bounded)
        for grp_name, vals in zip(grp_analysis_keys, grp_mean_inc_4class_cw_accs):
            vals_arr = np.array(vals)
            ma_accs = vals_arr.mean(axis=1)
            print(grp_name)
            print("\tMA-ACC", "{:.2f} %".format(np.mean(ma_accs)), "{:.2f} %".format(np.std(ma_accs) ))
            print("\tCW-ACC", ["{:.2f} %".format(v) for v in vals_arr.mean(axis=0)], ["{:.2f} %".format(v) for v in vals_arr.std(axis=0)])

    print("Compare-4cls-mean only" + "=" * 50)

    analyze_mean_inc_4class(test_means, test_ys, roundup=False)
    analyze_mean_inc_4class(test_means, test_ys, roundup=True)


    # interval contains gt
    def compute_interval_inc_any(lb, ub, ys):
        indices = compute_4cls_indices(ys)

        def include_any(lb, ub, all_gt):
            cor = torch.zeros_like(lb, dtype=bool)
            for gt in all_gt:
                cor |= (lb <= gt) & (gt <= ub)
            return cor


        cor = torch.zeros_like(ys, dtype=bool)

        cor[indices["A"]] = include_any(lb[indices["A"]], ub[indices["A"]], [0])
        cor[indices["B"]] = include_any(lb[indices["B"]], ub[indices["B"]], [1,2])
        cor[indices["C"]] = include_any(lb[indices["C"]], ub[indices["C"]], [3,4,5,6,7])
        cor[indices["D"]] = include_any(lb[indices["D"]], ub[indices["D"]], [8,9,10])
        cw_accs = ([(cor[v].float().mean().item() * 100) for _, v in indices.items()])
        cw_widths = [(ub[v] - lb[v]).mean().item() for _, v in indices.items()]
        return cw_accs, cw_widths

    def analyze_interval_inc_any(lb, ub, ys):
        interval_inc_any_cw_accs = {}
        interval_inc_any_cw_widths = {}
        for name in ys:
            interval_inc_any_cw_accs[name], interval_inc_any_cw_widths[name] = compute_interval_inc_any(lb[name], ub[name], ys[name])

        grp_interval_inc_any_cw_accs = group_res(interval_inc_any_cw_accs, group_maps, bounded)
        grp_interval_inc_any_cw_widths = group_res(interval_inc_any_cw_widths, group_maps, bounded)
        for grp_name, vals, vals_w in zip(grp_analysis_keys, grp_interval_inc_any_cw_accs, grp_interval_inc_any_cw_widths):
            vals_arr = np.array(vals)
            vals_w_arr = np.array(vals_w)
            ma_accs = vals_arr.mean(axis=1)
            ma_widths = vals_w_arr.mean(axis=1)
            print(grp_name)
            print("\tMA-ACC", "{:.2f} %".format(np.mean(ma_accs)), "{:.2f} %".format(np.std(ma_accs) ))
            print("\tCW-ACC", ["{:.2f} % +- {:.2f} %".format(v1, v2) for v1, v2 in zip(vals_arr.mean(axis=0), vals_arr.std(axis=0))])

            print("\tMA-Width", "{:.2f}".format(np.mean(ma_widths)), "{:.2f}".format(np.std(ma_widths)))
            print("\tCW-Width", ["{:.2f} +- {:.2f}".format(v1, v2) for v1, v2 in zip(vals_w_arr.mean(axis=0), vals_w_arr.std(axis=0))]) 

    print("Compare-4cls-interval" + "=" * 50)

    for gamma in [0.1, 0.2, 0.3, 0.4]:
        lbs = {}
        ubs = {}
        print("GAMMA", gamma)
        for name in sorted(test_means):
            b = find_thr(val_means[name], val_scales[name], val_ys[name], s, delta, gamma)
            if b == -1:
                print(name, "skipped")
                continue
            lbs[name], ubs[name] = prediction_interval(test_means[name], test_scales[name], n_classes=11, z_score=b)


        analyze_interval_inc_any(lbs, ubs, test_ys)


    # compare with JHU
    def analyze_ratio(means, ys):
        mean_ratio_all = {}
        mean_ratio_best = {}
        mean_ratio_rest = {}
        mean_ratio_rest_nz = {}
        mean_ratio_z = {}
        mean_ratio_dist = {}

        for name in sorted(means):
            bigger = torch.maximum(means[name], ys[name])
            lesser = torch.minimum(torch.clamp(means[name], min=1e-1), torch.clamp(ys[name], min=1e-1))
            ratio = torch.div(bigger, lesser)
            mean_ratio_all[name] = ratio.mean().item()

            hh = torch.histogram(ratio, bins=torch.tensor([0, 1.26, 1.58, 1000]))
            mean_ratio_dist[name] = hh.hist / hh.hist.sum()

            idx = (8 <= ys[name]) & (ys[name] <= 10) # >= 0.8
            mean_ratio_best[name] = ratio[idx].mean().item()

            idx = ys[name] < 8
            mean_ratio_rest[name] = ratio[idx].mean().item()

            idx = (1 <= ys[name]) & (ys[name] < 8)
            mean_ratio_rest_nz[name] = ratio[idx].mean().item()

            idx = ys[name] == 0
            mean_ratio_z[name] = ratio[idx].mean().item()

        return mean_ratio_all, mean_ratio_best, mean_ratio_rest, mean_ratio_rest_nz, mean_ratio_z, mean_ratio_dist


    mean_ratio_all, mean_ratio_best, mean_ratio_rest, mean_ratio_rest_nz, mean_ratio_z, mean_ratio_dist = analyze_ratio(test_means, test_ys)
    grp_mean_ratio_all = group_res(mean_ratio_all, group_maps, bounded)
    grp_mean_ratio_best = group_res(mean_ratio_best, group_maps, bounded)
    grp_mean_ratio_rest = group_res(mean_ratio_rest, group_maps, bounded)
    grp_mean_ratio_rest_nz = group_res(mean_ratio_rest_nz, group_maps, bounded)
    grp_mean_ratio_z = group_res(mean_ratio_z, group_maps, bounded)
    grp_mean_ratio_dist = group_res(mean_ratio_dist, group_maps, bounded)

    for grp_name, a, b, r, rnz, z, dist in zip(grp_analysis_keys, grp_mean_ratio_all, grp_mean_ratio_best, grp_mean_ratio_rest, grp_mean_ratio_rest_nz, grp_mean_ratio_z, grp_mean_ratio_dist):
        print(grp_name, "All {:.2f} +- {:.2f}".format(np.mean(a), np.std(a)))
        print(grp_name, "Best {:.2f} +- {:.2f}".format(np.mean(b), np.std(b)))
        print(grp_name, "Rest {:.2f} +- {:.2f}".format(np.mean(r), np.std(r)))
        print(grp_name, "RestNz {:.2f} +- {:.2f}".format(np.mean(rnz), np.std(rnz)))
        print(grp_name, "Z {:.2f} +- {:.2f}".format(np.mean(z), np.std(z)))

        dist = torch.vstack(dist)
        print(grp_name, dist.mean(axis=0), dist.std(axis=0))

if __name__ == '__main__':
    root_dir = sys.argv[1]
    main(root_dir)

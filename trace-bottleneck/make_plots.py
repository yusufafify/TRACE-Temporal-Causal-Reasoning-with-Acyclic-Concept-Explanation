import pickle
import pandas as pd
import os
import time
import numpy as np
import plotly.io as pio
from src.plots import plot_intervention, plot_level_intervention

"""
task accuracy after invervention on each individual concept     
"""
title = {'celeba': 'CelebA',
         'colormnist': 'ColorMNIST',
         'colormnist_ood': 'ColorMNIST',
         'asia': 'Asia',
         'asia_true': 'Asia',
         'alarm': 'Alarm',
         'alarm_true': 'Alarm',
         'sachs': 'Sachs',
         'sachs_true': 'Sachs',
         'hailfinder': 'Hailfinder',
         'hailfinder_true': 'Hailfinder',
         'insurance': 'Insurance',
         'insurance_true': 'Insurance',
         'pneumothorax': 'Pneumothorax',
         'cub': 'CUB',
         'cub_original': 'CUB_original',
         'synthetic': 'Synthetic',
}

def cumulative_improvement(means, stds):
    """
    This function computes the cumulative improvement of the means and stds
    Args:
        means: the means of the metrics
        stds: the stds of the metrics
    Returns:
        cum_improvement: the cumulative improvement of the means and stds
    """
    for model in means.keys():
        values_means = np.array(list(means[model].values()))
        values_stds = np.array(list(stds[model].values()))
        for level in means[model].keys():
            means[model][level] = values_means[:level+1].sum()
            stds[model][level] = values_stds[:level+1].sum()
    return means, stds

def compute_accuracy(l, std_mean=False, std_95=False):
    std = np.array(l).std(ddof=1)
    if std_mean:
        std = std / np.sqrt(len(l))
    if std_95:
        std = 2 * std
    return f'{round(np.array(l).mean()*100,2)} ± {round(std*100,2)}'

def compute_accuracy_interv(average, std_mean=False, std_95=False):
    keys = average[0].keys()
    mean = {key: np.array([d[key] for d in average]).mean() for key in keys}
    std = {key: np.array([d[key] for d in average]).std(ddof=1) for key in keys}
    if std_mean:
        std = {key: v / np.sqrt(len(average)) for key, v in std.items()}
    if std_95:
        std = {key: 2 * v for key, v in std.items()}
    return mean, std

def write_plot(fig, path, mode):
    # write a random plot before the real one (to avoid weird box in the pdf)
    pio.write_image(fig, path)
    # wait 0.5 seconds
    time.sleep(1)
    # save a figure of 600dpi, with 2.0 inches, and  height 0.75inches
    if mode == 'bar':
        pio.write_image(fig, path, width=2.6*600, height=1.5*600, scale=1)
    elif mode == 'line':
        pio.write_image(fig, path, width=2.*600, height=2*600, scale=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
folder = 'plots'
os.makedirs(folder, exist_ok=True)
root_result_dir =   {   
                        # 'colormnist': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [0,10,20,30,40]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [1,11,21,31,41]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [2,12,22,32,42]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [3,13,23,33,43]],
                        #     'c2bm':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [4,14,24,34,44]]
                        #     },
                        # 'colormnist_ood': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [5,15,25,35,45]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [6,16,26,36,46]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [7,17,27,37,47]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [8,18,28,38,48]],
                        #     'scbm':       [f'outputs/multirun/2025-05-10/10-20-49_scbm_ood/{i}' for i in [0,1,2,3,4]],
                        #     'c2bm':       [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [9,19,29,39,49]]
                        # },
                        # true graphs
                        # 'asia_true': { 
                        #     'c2bm':        [f'outputs/multirun/2025-05-15/11-45-48_c2bm_bn/{i}' for i in [0,3,6,9,12]],
                        # },
                        # 'sachs_true': { 
                        #     'c2bm':        [f'outputs/multirun/2025-05-15/11-45-48_c2bm_bn/{i}' for i in [1,4,7,10,13]],
                        # },
                        # 'insurance_true': { 
                        #     'c2bm':        [f'outputs/multirun/2025-05-15/11-45-48_c2bm_bn/{i}' for i in [2,5,8,11,14]],
                        # },
                        # 'alarm_true': { 
                        #     'c2bm':        [f'outputs/multirun/2025-05-15/11-46-04_c2bm_bn/{i}' for i in [0,2,4,6,8]]
                        # },
                        # 'hailfinder_true': { 
                        #     'c2bm':        [f'outputs/multirun/2025-05-15/11-46-04_c2bm_bn/{i}' for i in [1,3,5,7,9]]
                        # },
    
                        # # learned graph
                        'asia': { 
                            'blackbox':   [f'outputs/multirun/2025-05-14/22-56-37_blackbox_bn/{i}'  for i in [0,5,10,15,20]],   # 5 seeds
                            # 'blackbox_m': [f'outputs/multirun/2025-05-15/14-45-32_blackbox_multi_bn/{i}'  for i in [0,5,10,15,20]],
                            'cbm_linear': [f'outputs/multirun/2025-05-14/22-56-44_cbm_linear_bn/{i}'for i in [0,5,10,15,20]],
                            'cbm_mlp':    [f'outputs/multirun/2025-05-14/22-56-50_cbm_mlp_bn/{i}'   for i in [0,5,10,15,20]],
                            'cem':        [f'outputs/multirun/2025-05-14/22-56-55_cem_bn/{i}'       for i in [0,5,10,15,20]],
                            'scbm':       [f'outputs/multirun/2025-05-14/22-57-06_scbm_global_bn/{i}' for i in [0,5,10,15,20]],
                            'c2bm':       [f'outputs/multirun/2025-05-14/22-57-00_c2bm_bn/{i}'      for i in [0,5,10,15,20]]
                        },
                        # 'sachs': { 
                        #     'blackbox':   [f'outputs/multirun/2025-05-14/22-56-37_blackbox_bn/{i}'  for i in [1,6,11,16,21]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/14-45-32_blackbox_multi_bn/{i}'  for i in [1,6,11,16,21]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-14/22-56-44_cbm_linear_bn/{i}'for i in [1,6,11,16,21]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-14/22-56-50_cbm_mlp_bn/{i}'   for i in [1,6,11,16,21]],
                        #     'cem':        [f'outputs/multirun/2025-05-14/22-56-55_cem_bn/{i}'       for i in [1,6,11,16,21]],
                        #     'scbm':       [f'outputs/multirun/2025-05-14/22-57-06_scbm_global_bn/{i}' for i in [1,6,11,16,21]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-14/22-57-00_c2bm_bn/{i}'      for i in [1,6,11,16,21]]
                        # },
                        # 'insurance': { 
                        #     'blackbox':   [f'outputs/multirun/2025-05-14/22-56-37_blackbox_bn/{i}'  for i in [2,7,12,17,22]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/14-45-32_blackbox_multi_bn/{i}'  for i in [2,7,12,17,22]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-14/22-56-44_cbm_linear_bn/{i}'for i in [2,7,12,17,22]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-14/22-56-50_cbm_mlp_bn/{i}'   for i in [2,7,12,17,22]],
                        #     'cem':        [f'outputs/multirun/2025-05-14/22-56-55_cem_bn/{i}'       for i in [2,7,12,17,22]],
                        #     'scbm':       [f'outputs/multirun/2025-05-14/22-57-06_scbm_global_bn/{i}' for i in [2,7,12,17,22]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-14/22-57-00_c2bm_bn/{i}'      for i in [2,7,12,17,22]]
                        # },
                        # 'alarm': { 
                        #     'blackbox':   [f'outputs/multirun/2025-05-14/22-56-37_blackbox_bn/{i}'  for i in [3,8,13,18,23]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/14-45-32_blackbox_multi_bn/{i}'  for i in [3,8,13,18,23]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-14/22-56-44_cbm_linear_bn/{i}'for i in [3,8,13,18,23]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-14/22-56-50_cbm_mlp_bn/{i}'   for i in [3,8,13,18,23]],
                        #     'cem':        [f'outputs/multirun/2025-05-15/03-17-10_cem_alarm/{i}'    for i in [0,1,2,3,4]],
                        #     'scbm':       [f'outputs/multirun/2025-05-14/22-57-06_scbm_global_bn/{i}' for i in [3,8,13,18,23]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-14/22-57-00_c2bm_bn/{i}'      for i in [3,8,13,18,23]]
                        # },
                        # 'hailfinder': { 
                        #     'blackbox':   [f'outputs/multirun/2025-05-14/22-56-37_blackbox_bn/{i}'  for i in [4,9,14,19,24]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/14-45-32_blackbox_multi_bn/{i}'  for i in [4,9,14,19,24]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-14/22-56-44_cbm_linear_bn/{i}'for i in [4,9,14,19,24]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-14/22-56-50_cbm_mlp_bn/{i}'   for i in [4,9,14,19,24]],
                        #     'cem':        [f'outputs/multirun/2025-05-14/22-56-55_cem_bn/{i}'       for i in [4,9,14,19,24]],
                        #     'scbm':       [f'outputs/multirun/2025-05-14/22-57-06_scbm_global_bn/{i}' for i in [4,9,14,19,24]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-14/22-57-00_c2bm_bn/{i}'      for i in [4,9,14,19,24]]
                        # },
                        # 'celeba': {       
                        #     'blackbox':   [f'outputs/multirun/2025-05-15/02-45-59_blackbox_real/{i}' for i in [0,3,6,9,12]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/15-10-07_blackbox_multi_real/{i}'  for i in [0,3,6,9,12]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-15/02-46-40_cbm_linear_real/{i}' for i in [0,3,6,9,12]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-15/02-48-29_cbm_mlp_real/{i}' for i in [0,3,6,9,12]],
                        #     'cem':        [f'outputs/multirun/2025-05-15/02-47-31_cem_real/{i}' for i in [0,3,6,9,12]],
                        #     'scbm':       [f'outputs/multirun/2025-05-11/00-45-36_scbm_celeba/{i}' for i in [0,1,2,3,4]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-15/02-47-45_c2bm_real/{i}' for i in [0,3,6,9,12]],
                        # },
                        # 'cub': {
                        #     'blackbox':   [f'outputs/multirun/2025-05-15/02-45-59_blackbox_real/{i}' for i in [1,4,7,10,13]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/15-10-07_blackbox_multi_real/{i}'  for i in [1,4,7,10,13]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-15/02-46-40_cbm_linear_real/{i}' for i in [1,4,7,10,13]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-15/02-48-29_cbm_mlp_real/{i}' for i in [1,4,7,10,13]],
                        #     'cem':        [f'outputs/multirun/2025-05-15/02-47-31_cem_real/{i}' for i in [1,4,7,10,13]],
                        #     'scbm':       [f'outputs/multirun/2025-05-11/11-22-42_scbm_cub/{i}' for i in [0,1,2,3,4]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-15/02-47-45_c2bm_real/{i}' for i in [1,4,7,10,13]],
                        # },
                        # 'pneumothorax': {
                        #     'blackbox':   [f'outputs/multirun/2025-05-15/02-45-59_blackbox_real/{i}' for i in [2,5,8,11,14]],
                        #     'blackbox_m': [f'outputs/multirun/2025-05-15/15-10-07_blackbox_multi_real/{i}'  for i in [2,5,8,11,14]],
                        #     'cbm_linear': [f'outputs/multirun/2025-05-15/02-46-40_cbm_linear_real/{i}' for i in [2,5,8,11,14]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-05-15/02-48-29_cbm_mlp_real/{i}' for i in [2,5,8,11,14]],
                        #     'cem':        [f'outputs/multirun/2025-05-15/02-47-31_cem_real/{i}' for i in [2,5,8,11,14]],
                        #     'scbm':       [f'outputs/multirun/2025-05-11/00-47-28_scbm_pneumo/{i}' for i in [0,1,2,3,4]],
                        #     'c2bm':       [f'outputs/multirun/2025-05-15/02-47-45_c2bm_real/{i}' for i in [2,5,8,11,14]],
                        # },
                        # 'synthetic': {
                        #     'scbm':        [f'outputs/multirun/2025-05-11/00-38-52_scbm_synth/{i}' for i in [0,1]],
                        #     'cbm_linear':  [f'outputs/multirun/2025-05-15/15-14-56_scbm_synth/{i}' for i in [0,1]],
                        #     'cem':         [f'outputs/multirun/2025-05-15/15-15-06_scbm_synth/{i}' for i in [0,1]],    
                        # }
                    }

std_mean = True
std_95 = True
cumulative = False
delta = False
add_diff = True

label_acc_results = pd.DataFrame(index=['blackbox', 'blackbox_m', 'cbm_linear', 'cbm_mlp', 'cem', 'scbm', 'c2bm'], 
                                 columns=root_result_dir.keys())
task_acc_results = pd.DataFrame(index=['blackbox', 'blackbox_m', 'cbm_linear', 'cbm_mlp', 'cem', 'scbm', 'c2bm'], 
                                columns=root_result_dir.keys())

for dataset in root_result_dir.keys():
    print(f'----{dataset}-----')
    root_result_dir_d = root_result_dir[dataset]

    # average accuracy
    for model in root_result_dir_d.keys():
        average_label = []
        average_task = []
        for i, run in enumerate(root_result_dir_d[model]):
            single = []
            y_accuracy = pickle.load(open(run + '/results/y_accuracy.pkl', 'rb'))
            c_accuracy = pickle.load(open(run + '/results/c_accuracy.pkl', 'rb'))
            if 'c2bm' in root_result_dir_d.keys():
                c_accuracy_c2bm = pickle.load(open(root_result_dir_d['c2bm'][i] + '/results/c_accuracy.pkl', 'rb'))
                valid_concepts = [k for k, v in c_accuracy_c2bm.items() if not np.isnan(v)]
            else:
                print('c2bm not found, valid concepts are taken from the current model')
                valid_concepts = [k for k, v in c_accuracy.items() if not np.isnan(v)]

            # task accuracy
            task_acc = y_accuracy['_baseline']
            average_task.append(task_acc)

            # label accuracy
            single.append(task_acc)
            # and all valid concepts
            for c in valid_concepts:
                single.append(c_accuracy[c])
            average_label.append(np.array(single).mean())

        # compute the average
        task_acc_results.loc[model, dataset] = compute_accuracy(average_task, std_mean, std_95)  
        label_acc_results.loc[model, dataset] = compute_accuracy(average_label, std_mean, std_95)


# ----------------------------------------------------------------------------------------------------------------------------------


    res_to_plot = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot = {model:{} for model in root_result_dir_d.keys()}
    # single concept interventions
    # print('Single concept interventions')
    for model in res_to_plot.keys():
        average = [] # to store the average over seed
        if model == 'blackbox' or model == 'blackbox_m':
            continue
        for i, run in enumerate(root_result_dir_d[model]):
            single_c_on_y = pickle.load(open(run + '/results/single_c_interventions_on_y.pkl', 'rb'))
            y_nosy_baseline = single_c_on_y['_baseline']
            y_metric = {}
            for c_name in single_c_on_y.keys():
                if c_name != '_baseline':
                    if delta:
                        y_metric[c_name] = ((single_c_on_y[c_name] - y_nosy_baseline)/y_nosy_baseline)*100.
                    else:
                        y_metric[c_name] = single_c_on_y[c_name]*100.
            average.append(y_metric)
        # compute the average
        res_to_plot[model], std_to_plot[model] = compute_accuracy_interv(average, std_mean, std_95)

    fig = plot_intervention(res_to_plot, # accuracy delta after noise wrt to true baseline
                            std_to_plot,
                            f'{title[dataset]}')
    write_plot(fig, f"{folder}/{dataset}_SI_on_y.pdf", mode='bar')


# ----------------------------------------------------------------------------------------------------------------------------------
    

    res_to_plot_y = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot_y = {model:{} for model in root_result_dir_d.keys()}
    res_to_plot_c = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot_c = {model:{} for model in root_result_dir_d.keys()}
    res_to_plot_l = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot_l = {model:{} for model in root_result_dir_d.keys()}
    # level interventions on y
    # print('Level interventions on y')
    for model in res_to_plot_y.keys():
        # print(model)
        average_y = []
        average_c = []
        average_l = []
        if model == 'blackbox' or model == 'blackbox_m':
            continue
        for i, run in enumerate(root_result_dir_d[model]):
            policy_dict = pickle.load(open(run + '/policy.pkl', 'rb'))
            policy = policy_dict['policy']
            
            level_on_y = pickle.load(open(run + '/results/level_interventions_on_y.pkl', 'rb'))
            level_on_c = pickle.load(open(run + '/results/level_interventions_on_c.pkl', 'rb'))
            y_nosy_baseline = level_on_y['level 0']
            c_noisy_baseline = {key.split('/')[1].split('node ')[-1]: level_on_c[key] 
                                for key in level_on_c.keys() if 'level 0' in key}
            if model != 'scbm': assert y_nosy_baseline == pickle.load(open(run + '/results/single_c_interventions_on_y.pkl', 'rb'))['_baseline']
            # valid concepts
            if 'c2bm' in root_result_dir_d.keys():
                c_accuracy_c2bm = pickle.load(open(root_result_dir_d['c2bm'][i] + '/results/c_accuracy.pkl', 'rb'))
                valid_concepts = [k for k, v in c_accuracy_c2bm.items() if not np.isnan(v)]
            else:
                print('c2bm not found, valid concepts are taken from the current model')
                valid_concepts = [k for k, v in c_accuracy.items() if 'level 0' not in k and not np.isnan(v)]

            y_metric = {}
            c_metric = {}
            l_metric = {}

            # for y
            for name in level_on_y.keys():
                level_number = int(name.split(' ')[-1])
                level = policy[level_number-1]
                if delta:
                    y_metric[level_number] = ((level_on_y[name] - y_nosy_baseline)/y_nosy_baseline)*100.
                else:
                    y_metric[level_number] = level_on_y[name]*100.

            # for c
            for level in range(len(policy)+1):
                nodes = [key.split('node ')[-1] for key in level_on_c.keys() if f'level {level}' in key]
                valid_nodes = [node for node in nodes if node in valid_concepts]
                if delta:
                    temp = [(level_on_c[f'level {level}/node {c_name}'] - c_noisy_baseline[c_name])/c_noisy_baseline[c_name]*100.
                            for c_name in valid_nodes]
                else:
                    temp = [level_on_c[f'level {level}/node {c_name}']*100. 
                            for c_name in valid_nodes]
                c_metric[level] = sum(temp)/len(temp)

            # y + c
            for level in range(len(policy)+1):
                # append task improvement
                if delta:
                    temp = [((level_on_y[f'level {level}'] - y_nosy_baseline)/y_nosy_baseline)*100]
                else:
                    temp = [level_on_y[f'level {level}']*100]
                # append concepts improvement
                nodes = [key.split('node ')[-1] for key in level_on_c.keys() if f'level {level}' in key]
                valid_nodes = [node for node in nodes if node in valid_concepts]
                if delta:
                    temp += [(level_on_c[f'level {level}/node {c_name}'] - c_noisy_baseline[c_name])/c_noisy_baseline[c_name]*100.
                            for c_name in valid_nodes]
                else:
                    temp += [level_on_c[f'level {level}/node {c_name}']*100 
                                for c_name in valid_nodes]
                l_metric[level] = sum(temp)/len(temp)

            average_y.append(y_metric)
            average_c.append(c_metric)
            average_l.append(l_metric)
        # compute the average
        res_to_plot_y[model], std_to_plot_y[model] = compute_accuracy_interv(average_y, std_mean, std_95)
        res_to_plot_c[model], std_to_plot_c[model] = compute_accuracy_interv(average_c, std_mean, std_95)
        res_to_plot_l[model], std_to_plot_l[model] = compute_accuracy_interv(average_l, std_mean, std_95)

    # res_to_plot_y['blackbox'] = {key: 0 for key in res_to_plot_y['c2bm'].keys()}
    # std_to_plot_y['blackbox'] = {key: 0 for key in std_to_plot_y['c2bm'].keys()}
    # res_to_plot_c['blackbox'] = {key: 0 for key in res_to_plot_c['c2bm'].keys()}
    # std_to_plot_c['blackbox'] = {key: 0 for key in std_to_plot_c['c2bm'].keys()}
    # res_to_plot_l['blackbox'] = {key: 0 for key in res_to_plot_l['c2bm'].keys()}
    # std_to_plot_l['blackbox'] = {key: 0 for key in std_to_plot_l['c2bm'].keys()}
    # compute cumulative improvement
    if cumulative:
        assert delta
        res_to_plot_y, std_to_plot_y = cumulative_improvement(res_to_plot_y, std_to_plot_y)
        res_to_plot_c, std_to_plot_c = cumulative_improvement(res_to_plot_c, std_to_plot_c)
        res_to_plot_l, std_to_plot_l = cumulative_improvement(res_to_plot_l, std_to_plot_l)
    fig = plot_level_intervention(res_to_plot_y, 
                                  std_to_plot_y,
                                  'Task accuracy (%)',
                                  f'{title[dataset]}',
                                  add_diff=add_diff)
    write_plot(fig, f"{folder}/{dataset}_LI_on_y.pdf", mode='line')

    fig = plot_level_intervention(res_to_plot_c, 
                                  std_to_plot_c,
                                  'Concept accuracy (%)',
                                  f'{title[dataset]}',
                                  add_diff=add_diff)
    write_plot(fig, f"{folder}/{dataset}_LI_on_c.pdf", mode='line')

    fig = plot_level_intervention(res_to_plot_l, 
                                  std_to_plot_l,
                                  'Label accuracy (%)',
                                  f'{title[dataset]}',
                                  add_diff=add_diff)
    write_plot(fig, f"{folder}/{dataset}_LI_on_both.pdf", mode='line')



print('-- Label accuracy (concepts + task) --')
print(label_acc_results)
print('')
print('-- Task accuracy --')
print(task_acc_results)

# print('-- after noise is injected at test time --')
# print(acc_results_noisy)


import copy
import os
import torch
import numpy as np
import random
import csv
import seaborn as sns
import matplotlib.pyplot as plt

def set_rand_seed(seed=42):
    random.seed(seed)
    # Numpy seed
    np.random.seed(seed)
    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def mkdir(path, is_dir=True):
    path = path.split('/')
    now = ""
    for i in range(len(path)-1):
        now += path[i] + '/'
        if not os.path.exists(now):
            os.mkdir(now)
    now += path[-1]
    if is_dir and not os.path.exists(now):
        os.mkdir(now)


def shell_table(result, save_path):
    header = []
    for _,r in result.items():
        for k in r.keys():
            if k not in header:
                header.append(k)
    with open(save_path,'w',encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'] + header)

        for k,r in result.items():
            row = [k]
            for h in header:
                row.append(r.get(h))
            writer.writerow(row)


def sea_born(result, metrics_fn=None,save_path=None):
    # result[x][metric]
    count = {}
    for k,v in result.items():
        for m, c in v.items():
            if metrics_fn is None or m.split('#')[0] in metrics_fn.keys():
                fn = metrics_fn.get(m.split('#')[0]) if metrics_fn else None
                if m not in count:
                    count[m] = []
                if fn is not None:
                    c = fn(c)
                count[m].append(c)

    for m in count:
        sns.kdeplot(count[m], bw=2, label=m)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    return count


def shuffle_train(raw_data, example_num=10, num=1, repeat=1, dummy=None, random_seed=None):
    if random_seed:
        set_rand_seed(random_seed)
    assert len(raw_data) >= example_num + (dummy is None)
    train_data = []
    if dummy is None:
        dummy_test_indices = random.sample(list(range(len(raw_data))), k=num % len(raw_data))
        dummy_test_indices += list(range(len(raw_data))) * (num // len(raw_data))
        dummy = [raw_data[i] for i in dummy_test_indices]
    else:
        dummy_test_indices = None
        dummy = [dummy] * num

    for n in range(num):
        if len(raw_data) == example_num:
            dev = raw_data
        elif dummy_test_indices:
            dev_indices = random.sample(list(range(len(raw_data) - 1)), k=example_num)
            dev = [raw_data[i] if i < dummy_test_indices[n] else raw_data[i + 1] for i in dev_indices]
        else:
            dev = [raw_data[i] for i in random.sample(list(range(len(raw_data))), k=example_num)]

        repeat_dev = []
        for r in range(repeat):
            repeat_dev += copy.deepcopy(dev)
            random.shuffle(dev)
        train_data.append(
            {
                'dev': repeat_dev,
                'dummy_test': dummy[n]
            }
        )
    return train_data


# ----------metric-------------- #

def metric_average(result, metric_name, config, mode='mean'):
    if metric_name == "top_k":
        scores = []
        for i in range(config['max_top']):
            if mode == 'mean':
                scores.append(sum([r["top_k"][i] for r in result]) / len(result))
            elif mode == 'max':
                scores.append(max([r["top_k"][i] for r in result]))
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    return scores

def normalize(table, metric_name, normal_column, avg_name):
    # table (data, metric)
    new_table = copy.deepcopy(table)
    if metric_name == 'top_k':
        count = {}
        for x, (d, tasks) in enumerate(table.items()):
            if x == 0:
                for t in tasks:
                    count[t] = [0] * len(tasks[t])
            for t, r in tasks.items():
                for k in range(len(r)):
                    new_table[d][t][k] = table[d][t][k] / table[d][normal_column][k]
                    count[t][k] += new_table[d][t][k]
            if x == len(table) - 1:
                for t in tasks:
                    for k in range(len(count[t])):
                        count[t][k] /= len(table)
        new_table[avg_name] = count
        return new_table
    else:
        raise NotImplementedError


def top_k_metric(config, logits, answer_ids):
    max_top = config['max_top']
    count = [0] * max_top
    result = []
    top_flag = []
    for logit, aid in zip(logits, answer_ids):
        target_ids = aid[0]
        target_k = torch.where(torch.argsort(logit, descending=True) == target_ids)[0].item()
        if target_k < max_top:
            count[target_k] += 1
        top_flag.append(target_k)
    cnt = 0
    for k in range(max_top):
        cnt += count[k]
        result.append(cnt / len(answer_ids))
    return result, top_flag
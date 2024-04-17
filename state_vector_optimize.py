import argparse
import copy
import json
import os
import sys
import time
import random
import torch
from tqdm import tqdm
from TVeval import ICLVectorEvaluator
from TVframework import Evaluator
from utils import set_rand_seed
from selected_layers import *

torch.set_grad_enabled(False)

metric = {
    'top_k': {
        'max_top': 1
    }
}

def str2list(chars):
    return chars.split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tasks", type=str2list, help="antonym/english-french/person-instrument/person-occupation/product-company/landmark-country/capitalize/country-capital/person-sport/singular-plural/present-past/ag_news")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./result")
    parser.add_argument("--model_path", type=str, default="./llama-2-7b")
    parser.add_argument("--device", type=str, default="5")
    parser.add_argument("--eos", type=str, default="\n\n")
    parser.add_argument("--proj_tokens", type=str, default="→")
    parser.add_argument("--question_prompt", type=str, default="{input}")
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--max_test_num", type=int, default=10)
    parser.add_argument("--rand_seed", type=int, default=233)
    args = parser.parse_args()

    set_rand_seed(args.rand_seed)
    task_layers = {'zs': get_data_layers(is_fewshot=False, is_llama='llama' in args.model_path),
                   'fs': get_data_layers(is_fewshot=True, is_llama='llama' in args.model_path)}
    args.format_dict = {'eos': args.eos, 'proj_tokens': args.proj_tokens}

    evaluator = ICLVectorEvaluator(metric, Evaluator(args.model_path, devices=args.device))
    data_root_tasks = os.listdir(args.data_root)
    for task in data_root_tasks:
        if task not in args.target_tasks: continue
        task_path = os.path.join(args.data_root, task)
        if not os.path.isdir(task_path): continue
        print(f'----------------{task}----------------')
        task_list = list(sorted(os.listdir(task_path), key=lambda x:eval(x.split('.')[0])))
        task_save_path = os.path.join(args.save_path, task)
        os.makedirs(task_save_path, exist_ok=True)

        for fid, file in enumerate(task_list[:args.max_files]):
            file_save_path = os.path.join(args.save_path, task, file)

            if os.path.exists(os.path.join(task_save_path, 'total.json')):
                task_results = json.load(open(os.path.join(task_save_path, 'total.json')))
            else:
                task_results = {}
            if file not in task_results:
                task_results[file] = {}

            data = json.load(open(os.path.join(task_path, file)))
            dev_data = data['demon']
            dummy_test = data['dummy_test']
            test_data = data['test']
            valid_data = data['valid']
            if args.max_test_num != 0:
                test_data = test_data[:args.max_test_num]
            iv_result = {}

            for i in range(len(test_data)):
                test_data[i]['demon'] = []
            acc, _ = evaluator.single_ICL_test(test_data, format_dict=args.format_dict)
            acc = acc[0]
            print('Regular', acc)
            iv_result['Regular'] = acc

            for i in range(len(test_data)):
                test_data[i]['demon'] = data['demon']
            acc, _ = evaluator.single_ICL_test(test_data, format_dict=args.format_dict)
            acc = acc[0]
            print('ICL baseline', acc)
            iv_result['ICL baseline'] = acc


            for nshot, fshot in [('zs', False),('fs', True)]:
                nlayer = task_layers[nshot][task]
                run_name = f"{nshot}_raw_l{nlayer}"
                optimizer_weight = [[1, 2, 4, 8, 16, 0]]
                optimizer_config = {
                    "none": {},
                    "fixed": {"beta": [1, 1, 1, 1, 1, 1, 1]},
                    "fix-one-step": {"lr": [1], "weight": optimizer_weight},
                }
                stime = time.time()
                acc, _ = evaluator.single_atv_test(dummy_queries=[valid_data[0]],
                                                       dev_data=[dev_data],
                                                       test_data=test_data,
                                                       layer_indices=list(range(nlayer + 1)),
                                                       optimizer_config = optimizer_config,
                                                       fs_eval=fshot,
                                                       shuffle_labels=False,
                                                       intervention_mode='add#0#1',
                                                       add_to='atten',
                                                       question_prompt=args.question_prompt,
                                                       format_dict=args.format_dict)
                for k, v in acc.items():
                    iv_result[run_name + '_' + k] = v[0]
                    print(run_name + '_' + k, v[0])



            task_results[file] = {**task_results[file], **iv_result}
            json.dump(task_results, open(os.path.join(task_save_path, 'total.json'), "w"), ensure_ascii=False, indent=2)


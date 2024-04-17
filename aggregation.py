import argparse
import copy
import json
import os
import random
import time

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
            iv_result = {}
            test_data = data['test']
            dev_pool = data['demon']
            dummy_data = data['valid']
            if args.max_test_num != 0:
                test_data = test_data[:args.max_test_num]

            for i in range(len(test_data)):
                test_data[i]['demon'] = data['demon'][:10]

            baseline, _ = evaluator.single_ICL_test(test_data, format_dict=args.format_dict)
            baseline = baseline[0]
            print('baseline', baseline)
            iv_result['baseline'] = baseline

            task_vector = {k: {} for k in range(task_layers['zs'][task] + 1)}
            dev_data = []
            for i in range(10):
                cur_dev = []
                for x in range(5):
                    dd = copy.deepcopy(dev_pool[i*10: (i+1)*10])
                    random.shuffle(dd)
                    cur_dev.append(dd)

                # average
                dev_data.append(dev_pool[i * 10: (i+1) * 10])
                run_name = f"zs_raw_l{task_layers['zs'][task]}_avg_{i * 10}shot"
                acc, _ = evaluator.single_atv_test(dummy_queries=dummy_data[:len(dev_data)],
                                                   dev_data=dev_data,
                                                   test_data=test_data,
                                                   layer_indices=list(range(task_layers['zs'][task] + 1)),
                                                   fs_eval=False,
                                                   shuffle_labels=False,
                                                   intervention_mode='add#0#1',
                                                   add_to='atten',
                                                   format_dict=args.format_dict)
                iv_result[run_name] = acc[0]


                sstime = time.time()
                single_tv = evaluator.get_task_vector(dummy_queries=[dummy_data[i]] * len(cur_dev),
                                                      dev_data=cur_dev,
                                                      layer_indices=list(range(task_layers['zs'][task] + 1)),
                                                      format_dict=args.format_dict
                                                      )
                for layer_name, layer_tv in single_tv.items():
                    task_vector[layer_name][f"dev_{i}"] = layer_tv[f"test"]

                cur_tv = evaluator.tv_write_then_read(dummy_queries=dummy_data[10:10+5],
                                                      dev_data=[dummy_data[0:i + 1]] * 5,
                                                      task_vector=task_vector,
                                                      intervention_mode=f'add#0#1',
                                                      add_to='atten',
                                                      format_dict=args.format_dict
                                                    )
                run_name = f"zs_raw_l{task_layers['zs'][task]}_{i*10}shot"
                acc, _ = evaluator.eval_task_vector(test_data=test_data,
                                                    task_vector=cur_tv,
                                                    fs_eval=False,
                                                    shuffle_labels=False,
                                                    intervention_mode=f'add#0#1',
                                                    add_to='atten',
                                                    format_dict=args.format_dict
                                                    )
                iv_result[run_name] = acc[0]
                print(run_name, acc[0])

            task_vector = {k: {} for k in range(task_layers['fs'][task] + 1)}
            dev_data = []
            for i in range(10):
                cur_dev = []
                for x in range(5):
                    dd = copy.deepcopy(dev_pool[i*10: (i+1)*10])
                    random.shuffle(dd)
                    cur_dev.append(dd)

                # average
                dev_data.append(dev_pool[i * 10: (i+1) * 10])
                run_name = f"fs_raw_l{task_layers['fs'][task]}_avg_{i * 10}shot"
                acc, _ = evaluator.single_atv_test(dummy_queries=dummy_data[:len(dev_data)],
                                                   dev_data=dev_data,
                                                   test_data=test_data,
                                                   layer_indices=list(range(task_layers['fs'][task] + 1)),
                                                   fs_eval=True,
                                                   shuffle_labels=False,
                                                   intervention_mode='add#0#1',
                                                   add_to='atten',
                                                   format_dict=args.format_dict)
                iv_result[run_name] = acc[0]

                # few shot
                single_tv = evaluator.get_task_vector(dummy_queries=[dummy_data[i]] * len(cur_dev),
                                                      dev_data=cur_dev,
                                                      layer_indices=list(range(task_layers['fs'][task] + 1)),
                                                      format_dict=args.format_dict
                                                      )
                for layer_name, layer_tv in single_tv.items():
                    task_vector[layer_name][f"dev_{i}"] = layer_tv[f"test"]

                run_name = f"fs_raw_l{task_layers['fs'][task]}_{i * 10}shot"
                acc, _ = evaluator.eval_dev_task_vector(test_data=test_data,
                                                        dev_data=dummy_data[0: i + 1],
                                                        task_vector=task_vector,
                                                        intervention_mode=f'add#0#1',
                                                        format_dict=args.format_dict
                                                        )
                iv_result[run_name] = acc[0]
                print(run_name, acc[0])


            task_results[file] = {**task_results[file], **iv_result}
            json.dump(task_results, open(os.path.join(task_save_path, 'total.json'), "w"), ensure_ascii=False, indent=2)




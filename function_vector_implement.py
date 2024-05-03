import json
import os
import random
import argparse
import torch
from tqdm import tqdm
from TVeval import ICLVectorEvaluator
from TVframework import Evaluator
from utils import set_rand_seed

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
    parser.add_argument("--max_valid_num", type=int, default=10)
    parser.add_argument("--rand_seed", type=int, default=233)
    args = parser.parse_args()

    set_rand_seed(args.rand_seed)
    evaluator = ICLVectorEvaluator(metric, Evaluator(args.model_path, devices=args.device))
    data_root_tasks = os.listdir(args.data_root)
    args.format_dict = {'eos': args.eos, 'proj_tokens': args.proj_tokens}

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
            for i in range(len(test_data)):
                test_data[i]['demon'] = data['demon']
            for i in range(len(valid_data)):
                valid_data[i]['demon'] = data['demon']


            # Find the top 10 attention heads with the greatest impact
            # function_vector: The average activation of the top 10 attention heads with the greatest impact.
            # top_head: top 10 attention heads with the greatest impact
            _, function_vector, top_head = evaluator.indirect_act_effect(dummy_queries=[valid_data[0]],
                                                                         dev_data=[dev_data],
                                                                         valid_data=valid_data[1:1+args.max_valid_num],
                                                                         n_top_heads=10,
                                                                         format_dict=args.format_dict)

            print("top_head", top_head)

            # Eval the function vector in few shot setting or zero shot setting across all layers
            fv_result = {}
            for fv_layer in range(evaluator.evaluator.model.config.num_hidden_layers):
                for fv_shot in ['fs', 'zs']:
                    fv_result[f"fv_{fv_layer}_{fv_shot}"] = \
                        evaluator.eval_act_effect(function_vector=function_vector, test_data=test_data,
                                                  layer_indices=[fv_layer], fs_eval=(fv_shot=='fs'),
                                                  shuffle_labels=False, intervention_mode='add#1#1',
                                                  add_to="hidden", format_dict=args.format_dict)
                    fv_result[f"fv_{fv_layer}_{fv_shot}"] = fv_result[f"fv_{fv_layer}_{fv_shot}"][0]
                    print(f'fv_{fv_layer}_{fv_shot}: {fv_result[f"fv_{fv_layer}_{fv_shot}"]}')

            print("fv_result done", fv_result)

            task_results[file] = {**task_results[file], **fv_result}
            json.dump(task_results, open(os.path.join(task_save_path, 'total.json'), "w"), ensure_ascii=False, indent=2)

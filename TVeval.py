import copy
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import yaml
import json
import torch
import random
import requests
import argparse
import time
from TVframework import Evaluator
from typing import List
from optimizer import get_optimizers, BaseOptimizer

from utils import *

start_time = time.strftime("%Y_%m-%d-%H-%M-%S", time.localtime()).split('_')[-1]
print(f"start time: {start_time}, PID: {os.getpid()}")


class ICLVectorEvaluator():
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.tokenizer = evaluator.tokenizer

    def avg_tv(self, raw_task_vector_list):
        task_vector_list = {}
        for layer_name in raw_task_vector_list[0]:
            task_vector_list[layer_name] = {}
            for tv_name in raw_task_vector_list[0][layer_name]:
                tv = [raw_task_vector_list[i][layer_name][tv_name] for i in range(len(raw_task_vector_list))]
                tv = torch.stack(tv, dim=0).mean(dim=0)
                task_vector_list[layer_name][tv_name] = tv
        return task_vector_list

    def get_task_vector(self,  dummy_queries, dev_data, layer_indices, question_prompt='{input}', add_to='atten', format_dict={}):
        task_vector_list = []
        for i, (dummy_query, dev) in enumerate(zip(dummy_queries, dev_data)):
            demon = []
            for example in dev:
                question = question_prompt.format_map(example)
                answer = example['output']
                demon.append((question, answer))
            dummy_query = question_prompt.format_map(dummy_query)
            if add_to == 'atten':
                task_vector, _ = self.evaluator.read_activation(dummy_query, demon, layer_indices, format_dict)
            else:
                task_vector, _ = self.evaluator.read_hidden(dummy_query, demon, layer_indices, format_dict)
            task_vector_list.append(task_vector)
        return self.avg_tv(task_vector_list)

    def single_atv_test(self, dummy_queries, dev_data, test_data, layer_indices, optimizer_config=None, fs_eval=False, shuffle_labels=False, intervention_mode='replace', add_to='atten', question_prompt='{input}', format_dict={}):
        task_vector_list = []
        for i, (dummy_query, dev) in enumerate(zip(dummy_queries, dev_data)):
            demon = []
            for example in dev:
                question = question_prompt.format_map(example)
                answer = example['output']
                demon.append((question, answer))
            dummy_query = question_prompt.format_map(dummy_query)
            task_vector, _ = self.evaluator.read_activation(dummy_query, demon, layer_indices, format_dict)
            task_vector_list.append(task_vector)

        ori_task_vector = self.avg_tv(task_vector_list)
        dev_task_vector = {k: [v[f'dev_{i}'] for i in range(len(v)-1)] for k,v in ori_task_vector.items()}
        test_task_vector = {k: v[f'test'] for k,v in ori_task_vector.items()}
        if optimizer_config is None:
            optimizers = [BaseOptimizer('none', None)]
            return_none = True
        else:
            optimizers = get_optimizers(optimizer_config)
            return_none = False

        topk = {}
        test_order = {}
        for opt in optimizers:
            task_vector = opt(dev_task_vector,test_task_vector)
            logit_list = []
            answer_ids_list = []
            for d in test_data:
                query = question_prompt.format_map(d)
                demon = []
                if fs_eval:
                    labels_map = list(range(len(d['demon'])))
                    if shuffle_labels:
                        random.shuffle(labels_map)
                    for x, example in enumerate(d['demon']):
                        question = question_prompt.format_map(example)
                        answer = d['demon'][labels_map[x]]['output']
                        demon.append((question, answer))

                logits = self.evaluator.write_activation(query, task_vector, demon, intervention_mode, add_to, format_dict)
                answer_ids = self.evaluator.get_answer_id(query=query, answer=d['output'], proj_tokens=format_dict.get("proj_tokens"))
                logit_list.append(logits)
                answer_ids_list.append(answer_ids)
            topk[opt.name], test_order[opt.name] = top_k_metric(self.config['top_k'], logit_list, answer_ids_list)

        if return_none:
            return topk['none'], test_order['none']
        return topk, test_order

    def single_hid_test(self, dummy_queries, dev_data, test_data, layer_indices, fs_eval=False, shuffle_labels=False, intervention_mode='replace', question_prompt='{input}', format_dict={}):
        task_vector_list = []
        for i, (dummy_query, dev) in enumerate(zip(dummy_queries, dev_data)):
            demon = []
            for example in dev:
                question = question_prompt.format_map(example)
                answer = example['output']
                demon.append((question, answer))
            dummy_query = question_prompt.format_map(dummy_query)
            task_vector, _ = self.evaluator.read_hidden(dummy_query, demon, layer_indices, format_dict)
            task_vector_list.append(task_vector)

        task_vector = self.avg_tv(task_vector_list)
        task_vector = {k: v['test'] for k,v in task_vector.items()}

        logit_list = []
        answer_ids_list = []
        for d in test_data:
            query = question_prompt.format_map(d)
            demon = []
            if fs_eval:
                labels_map = list(range(len(d['demon'])))
                if shuffle_labels:
                    random.shuffle(labels_map)
                for x, example in enumerate(d['demon']):
                    question = question_prompt.format_map(example)
                    answer = d['demon'][labels_map[x]]['output']
                    demon.append((question, answer))

            logits = self.evaluator.write_hidden(query, task_vector, demon, intervention_mode, format_dict)
            answer_ids = self.evaluator.get_answer_id(query=query, answer=d['output'], proj_tokens=format_dict.get("proj_tokens"))
            logit_list.append(logits)
            answer_ids_list.append(answer_ids)
        topk, test_order = top_k_metric(self.config['top_k'], logit_list, answer_ids_list)
        return topk, test_order

    def single_ICL_test(self, test_data, shuffle_labels=False, question_prompt='{input}', format_dict={}):
        logit_list = []
        answer_ids_list = []
        for d in test_data:
            query = question_prompt.format_map(d)
            demon = []
            labels_map = list(range(len(d['demon'])))
            if shuffle_labels:
                random.shuffle(labels_map)
            for x, example in enumerate(d['demon']):
                question = question_prompt.format_map(example)
                answer = d['demon'][labels_map[x]]['output']
                demon.append((question, answer))

            _, logits = self.evaluator.read_activation(query, demon, [], format_dict)
            answer_ids = self.evaluator.get_answer_id(query=query, answer=d['output'], proj_tokens=format_dict.get("proj_tokens"))
            logit_list.append(logits)
            answer_ids_list.append(answer_ids)
        topk, test_order = top_k_metric(self.config['top_k'], logit_list, answer_ids_list)
        return topk, test_order

    def tv_write_then_read(self, dummy_queries, dev_data, task_vector, intervention_mode='replace', add_to='atten', question_prompt='{input}', format_dict={}):
        write_position = None
        input_tv = {}
        for layer_name, layer_tv in task_vector.items():
            if write_position is None:
                write_position = list(sorted(layer_tv.keys()))
            input_tv[layer_name] = torch.cat([layer_tv[wp] for wp in write_position], dim=0) # debug
        write_position = [f"project_{wp[len('dev_'):]}" if wp.startswith('dev_') else wp for wp in write_position]
        task_vector_list = []
        for i, (dummy_query, dev) in enumerate(zip(dummy_queries, dev_data)):
            demon = []
            for example in dev:
                question = question_prompt.format_map(example)
                answer = example['output']
                demon.append((question, answer))
            dummy_query = question_prompt.format_map(dummy_query)

            input_ids, input_mask = self.evaluator.format_example_with_length(dummy_query, None, demon_list=demon, **format_dict)
            intervention_indices = torch.cat(
                [
                    torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == wp], dtype=torch.long)
                    for wp in write_position
                ], dim=0
            )
            logits, activations = self.evaluator.intervention_activation(input_ids, intervention_indices, input_tv, intervention_mode)

            task_vector = {l: {} for l in activations}
            for lay, tv in activations.items():
                for r in range(len(demon)):
                    proj_name = f"project_{r}"
                    indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name], dtype=torch.long)
                    task_vector[lay][f'dev_{r}'] = tv[indices]
                proj_name = f"project"
                indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name], dtype=torch.long)
                task_vector[lay]['test'] = tv[indices]

            task_vector_list.append(task_vector)
        return self.avg_tv(task_vector_list)

    def eval_task_vector(self, test_data, task_vector, optimizer_config=None, fs_eval=False, shuffle_labels=True, intervention_mode='add#0#1', add_to='atten', question_prompt='{input}', format_dict={}):
        """
        评估task vector的效果,
        """
        ori_task_vector = task_vector
        dev_task_vector = {k: [v[f'dev_{i}'] for i in range(len(v) - 1)] for k, v in ori_task_vector.items()}
        test_task_vector = {k: v[f'test'] for k, v in ori_task_vector.items()}
        if optimizer_config is None:
            optimizers = [BaseOptimizer('none', None)]
            return_none = True
        else:
            optimizers = get_optimizers(optimizer_config)
            return_none = False

        topk = {}
        test_order = {}
        for opt in optimizers:
            task_vector = opt(dev_task_vector, test_task_vector)
            logit_list = []
            answer_ids_list = []
            for d in test_data:
                query = question_prompt.format_map(d)
                demon = []
                if fs_eval:
                    labels_map = list(range(len(d['demon'])))
                    if shuffle_labels:
                        random.shuffle(labels_map)
                    for x, example in enumerate(d['demon']):
                        question = question_prompt.format_map(example)
                        answer = d['demon'][labels_map[x]]['output']
                        demon.append((question, answer))

                if add_to == 'atten':
                    logits = self.evaluator.write_activation(query, task_vector, demon, intervention_mode, add_to,format_dict)
                else:
                    logits = self.evaluator.write_hidden(query, task_vector, demon, intervention_mode, format_dict)
                answer_ids = self.evaluator.get_answer_id(query=query, answer=d['output'], proj_tokens=format_dict.get("proj_tokens"))
                answer_ids_list.append(answer_ids)
                if d.get("choices") is None:
                    logit_list.append(logits)
                else:
                    indices = [self.evaluator.get_answer_id(query=query, answer=c, proj_tokens=format_dict.get("proj_tokens"))[0] for c in d.get('choices')]
                    mask = torch.ones(logits.shape).to(logits)
                    mask[indices] = 0
                    logits[mask == 1] = -torch.inf
                    logit_list.append(logits)
            topk[opt.name], test_order[opt.name] = top_k_metric(self.config['top_k'], logit_list, answer_ids_list)

        if return_none:
            return topk['none'], test_order['none']
        return topk, test_order


    def eval_dev_task_vector(self, test_data, dev_data, task_vector, intervention_mode='replace', question_prompt='{input}', format_dict={}):
        write_position = None
        input_tv = {}
        for layer_name, layer_tv in task_vector.items():
            if write_position is None:
                write_position = list(sorted(layer_tv.keys()))
            input_tv[layer_name] = torch.cat([layer_tv[wp] for wp in write_position], dim=0) # debug
        write_position = [f"project_{wp[len('dev_'):]}" if wp.startswith('dev_') else wp for wp in write_position]

        logit_list = []
        answer_ids_list = []
        for d in test_data:
            query = question_prompt.format_map(d)
            demon = []
            for x, example in enumerate(dev_data):
                question = question_prompt.format_map(example)
                answer = example['output']
                demon.append((question, answer))

            input_ids, input_mask = self.evaluator.format_example_with_length(query, None, demon_list=demon, **format_dict)
            intervention_indices = torch.cat(
                [
                    torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == wp], dtype=torch.long)
                    for wp in write_position
                ], dim=0
            )
            logits, activations = self.evaluator.intervention_activation(input_ids, intervention_indices, input_tv, intervention_mode)
            answer_ids = self.evaluator.get_answer_id(query=query, answer=d['output'], proj_tokens=format_dict.get("proj_tokens"))
            logit_list.append(logits)
            answer_ids_list.append(answer_ids)

        topk, test_order = top_k_metric(self.config['top_k'], logit_list, answer_ids_list)

        return topk, test_order


    def single_generate(self, test_data, shuffle_labels=False, question_prompt='{input}', format_dict={}, return_topk=None):
        answers = []
        for d in test_data:
            query = question_prompt.format_map(d)
            demon = []
            labels_map = list(range(len(d['demon'])))
            if shuffle_labels:
                random.shuffle(labels_map)
            for x, example in enumerate(d['demon']):
                question = question_prompt.format_map(example)
                answer = d['demon'][labels_map[x]]['output']
                demon.append((question, answer))

            full_text, output_text = self.evaluator.generate(query, demon, format_dict, return_topk=return_topk)

            answers.append((full_text, output_text))
        return answers

    def single_atv_generate(self, dummy_queries, dev_data, test_data, layer_indices, optimizer_config=None, fs_eval=False, shuffle_labels=False, intervention_mode='replace', add_to='atten', question_prompt='{input}', format_dict={}, return_topk=None):
        task_vector_list = []
        for i, (dummy_query, dev) in enumerate(zip(dummy_queries, dev_data)):
            demon = []
            for example in dev:
                question = question_prompt.format_map(example)
                answer = example['output']
                demon.append((question, answer))
            dummy_query = question_prompt.format_map(dummy_query)
            task_vector, _ = self.evaluator.read_activation(dummy_query, demon, layer_indices, format_dict)
            task_vector_list.append(task_vector)

        ori_task_vector = self.avg_tv(task_vector_list)
        dev_task_vector = {k: [v[f'dev_{i}'] for i in range(len(v)-1)] for k,v in ori_task_vector.items()}
        test_task_vector = {k: v[f'test'] for k,v in ori_task_vector.items()}
        if optimizer_config is None:
            optimizers = [BaseOptimizer('none', None)]
            return_none = True
        else:
            optimizers = get_optimizers(optimizer_config)
            return_none = False

        topk = {}
        for opt in optimizers:
            task_vector = opt(dev_task_vector,test_task_vector)
            answers = []
            for d in test_data:
                query = question_prompt.format_map(d)
                demon = []
                if fs_eval:
                    labels_map = list(range(len(d['demon'])))
                    if shuffle_labels:
                        random.shuffle(labels_map)
                    for x, example in enumerate(d['demon']):
                        question = question_prompt.format_map(example)
                        answer = d['demon'][labels_map[x]]['output']
                        demon.append((question, answer))

                full_text, output_text = self.evaluator.generate_write(query, demon, format_dict=format_dict, task_vector=task_vector, intervention_mode=intervention_mode, add_to=add_to, return_topk=return_topk)
                answers.append((full_text, output_text))
            topk[opt.name] = answers

        if return_none:
            return topk['none']
        return topk


import logging
import re
import string
import os
import time

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Dict, Union
from baukit import TraceDict
import numpy as np

class Evaluator:
    def __init__(self, model_path, lora_weight=None, devices=None, half=False, load_float16=False, load_bfloat16=False, load_int8=False, model_max_length=None):
        torch.set_grad_enabled(False)
        self.model, self.tokenizer, self.num2attn, self.num2layer = self.load(
            model_path=model_path,
            lora_weight=lora_weight,
            devices=devices,
            half=half,
            load_float16=load_float16,
            load_bfloat16=load_bfloat16,
            load_int8=load_int8,
            model_max_length=model_max_length
        )

        self.forward_model_dict = {}
        for layer in range(self.model.config.num_hidden_layers):
            for name, module in self.model.named_modules():
                if name == self.num2attn(layer):
                    self.forward_model_dict[name] = module
                elif name == self.num2layer(layer):
                    self.forward_model_dict[name] = module

    @staticmethod
    def load(model_path, lora_weight=None, devices=None, half=False, load_float16=False, load_bfloat16=False, load_int8=False, model_max_length=None):

        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='left', padding_side='left', use_fast=False)
        if model_max_length is not None:
            tokenizer.model_max_length = model_max_length

        logging.info(f'loading model from: {model_path}')
        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [eval(i) for i in devices.split(',')]

        torch_dtype = torch.float32
        if load_float16:
            torch_dtype = torch.float16
        elif load_bfloat16:
            torch_dtype = torch.bfloat16

        if len(devices) == 1:
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         torch_dtype=torch_dtype,
                                                         load_in_8bit=load_int8).to(devices[0])
        else:
            map_list = {}
            for i in range(torch.cuda.device_count()):
                if i in devices:
                    map_list[i] = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3)}GB'
                else:
                    map_list[i] = '0GB'
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         device_map="auto",
                                                         max_memory=map_list,
                                                         torch_dtype=torch_dtype,
                                                         load_in_8bit=load_int8)
        if lora_weight:
            model = PeftModel.from_pretrained(
                model,
                lora_weight,
            )

        if half:
            model = model.half()
        model = model.eval()

        logging.info(f'loading {type(model)} model done')

        if 'llama' in model_path.lower():
            num2attn = lambda x: f'model.layers.{x}.self_attn.o_proj'
            num2layer = lambda x: f'model.layers.{x}'
        elif 'gptj' in model_path.lower():
            num2attn = lambda x: f'transformer.h.{x}.attn.out_proj'
            num2layer = lambda x: f'transformer.h.{x}'
        else:
            raise NotImplementedError

        return model, tokenizer, num2attn, num2layer


    def format_example(self, query:str, answer:str=None, demon_list:List=None, system:str='', proj_tokens:str='→', eos:str=None):
        if demon_list is None: demon_list = []
        if eos is None:  eos = ''
        if system is None: system = ''

        def tokenize(target_str):
            nonlocal sentence
            target_str = target_str.rstrip(' ')
            source = self.tokenizer(sentence, truncation=False, padding=False ,add_special_tokens=False).input_ids
            target = self.tokenizer(sentence + target_str, truncation=False, padding=False , add_special_tokens=False).input_ids
            assert len(source) < len(target)
            sentence += target_str
            return target[len(source): ]

        sentence = system
        input_tokens = self.tokenizer(sentence, truncation=False, padding=False).input_ids
        input_mask = ['bos'] + ['system'] * (len(input_tokens) - 1)

        for r, (q, a) in enumerate(demon_list):
            input_tokens += tokenize(q)
            input_mask += [f'query_{r}'] * (len(input_tokens) - len(input_mask))
            input_tokens += tokenize(proj_tokens)
            # assert (len(input_tokens) - len(input_mask)) == len(self.tokenizer.tokenize(proj_tokens))
            input_mask += [f'project_{r}'] * (len(input_tokens) - len(input_mask))
            input_tokens += tokenize(a)
            input_mask += [f'answer_{r}'] * (len(input_tokens) - len(input_mask))
            input_tokens += tokenize(eos)
            input_mask += [f'eos_{r}'] * (len(input_tokens) - len(input_mask))

        input_tokens += tokenize(query)
        input_mask += [f'query'] * (len(input_tokens) - len(input_mask))
        input_tokens += tokenize(proj_tokens)
        # assert (len(input_tokens) - len(input_mask)) == len(self.tokenizer.tokenize(proj_tokens))
        input_mask += [f'project'] * (len(input_tokens) - len(input_mask))

        if answer:
            input_tokens += tokenize(answer)
            input_mask += [f'answer'] * (len(input_tokens) - len(input_mask))

        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

        return input_ids, input_mask

    def format_example_with_length(self, query: str, answer: str = None, demon_list: List = None, system: str = '', proj_tokens: str = '→', eos: str = None, strict=True):
        max_len = self.model.config.max_position_embeddings
        if demon_list is None: demon_list = []
        for i in range(len(demon_list),-1,-1):
            input_ids, input_mask = self.format_example(query, answer, demon_list[:i], system, proj_tokens, eos)
            if input_ids.shape[1] >= max_len:
                if i == 0:
                    logging.info(f"[WARNING] zero-shot overflow!")
                assert not strict
            else:
                if i != len(demon_list):
                    logging.info(f"[WARNING] {i+1}-shot overflow!")
                break
        return input_ids, input_mask

    def get_answer_id(self,query, answer, proj_tokens=None):
        if proj_tokens is None: proj_tokens = '→'
        source = self.tokenizer(query + proj_tokens, truncation=False, padding=False).input_ids
        target = self.tokenizer(query + proj_tokens + answer, truncation=False, padding=False).input_ids
        assert len(source) < len(target) < self.tokenizer.model_max_length
        return target[len(source): ]
        # return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))

    def write_and_read_activation(self, input_ids, read_hook_names=None, write_hook_names=None, write_hook_fn=None):
        if read_hook_names is None:
            read_hook_names = []
        if write_hook_names is None:
            write_hook_names = []
            write_hook_fn = None
        hook_names = list(set(read_hook_names) | set(write_hook_names))
        with torch.no_grad():
            with TraceDict(self.model, layers=hook_names, clone=False, detach=False, retain_input=True, retain_output=False, edit_output=write_hook_fn) as activations_td:
                logits = self.model(input_ids.to(self.model.device)).logits.cpu()
        hook_input = {l: activations_td[l].input[0].cpu() for l in read_hook_names} #Layer: (Len, Hidden)
        return hook_input, logits[0,-1]

    def write_and_read_hidden(self, input_ids, read_hook_names=None, write_hook_names=None, write_hook_fn=None):
        if read_hook_names is None:
            read_hook_names = []
        if write_hook_names is None:
            write_hook_names = []
            write_hook_fn = None
        hook_names = list(set(read_hook_names) | set(write_hook_names))
        with torch.no_grad():
            with TraceDict(self.model, layers=hook_names, clone=False, detach=False, edit_output=write_hook_fn) as activations_td:
                logits = self.model(input_ids.to(self.model.device)).logits.cpu()
        hook_input = {l: activations_td[l].output[0][0].cpu() for l in read_hook_names} #Layer: (Len, Hidden)
        return hook_input, logits[0,-1]

    def select_task_vector(self, input_mask, vectors, example_num):
        task_vector = {}
        for r in range(example_num):
            proj_name = f"project_{r}"
            indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name], dtype=torch.long)
            task_vector[f'dev_{r}'] = vectors[indices]
        proj_name = f"project"
        indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name], dtype=torch.long)
        task_vector['test'] = vectors[indices]
        return task_vector

    def read_activation(self, query, demon_list, layer_indices=None, format_dict={}):
        if layer_indices is None:
            layer_indices = range(self.model.config.num_hidden_layers)
        layer_hook_names = [self.num2attn(x) for x in layer_indices]
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        activation, logits = self.write_and_read_activation(input_ids, layer_hook_names)
        tv = {l: self.select_task_vector(input_mask, activation[self.num2attn(l)], len(demon_list)) for l in layer_indices}
        return tv, logits

    def read_hidden(self, query, demon_list, layer_indices=None, format_dict={}):
        if layer_indices is None:
            layer_indices = range(self.model.config.num_hidden_layers)
        layer_hook_names = [self.num2layer(x) for x in layer_indices]
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        activation, logits = self.write_and_read_hidden(input_ids, layer_hook_names)
        tv = {l: self.select_task_vector(input_mask, activation[self.num2layer(l)], len(demon_list)) for l in layer_indices}
        return tv, logits

    def write_activation(self, query, task_vector, demon_list=None, intervention_mode='replace', add_to="atten", format_dict={}, write_pos=["project"] ,return_mode='logits'):
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.cat(
            [
                torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == wp], dtype=torch.long)
                for wp in write_pos
            ], dim=0
        )
        config = {"intervention_mode": intervention_mode}
        layer_indices = list(task_vector.keys())
        if add_to == 'atten':
            layer_hook_names = [self.num2attn(x) for x in layer_indices]
            task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
            intervention_fn = self.intervention_function(config, layer_hook_names, tv_indices, task_vector, self.model.device, self.forward_model_dict)
            activation, logits = self.write_and_read_activation(input_ids, layer_hook_names, layer_hook_names, intervention_fn)
            new_task_vector = {l: self.select_task_vector(input_mask, activation[self.num2attn(l)], len(demon_list)) for l in layer_indices}
        else:
            layer_hook_names = [self.num2layer(x) for x in layer_indices]
            task_vector = {self.num2layer(k): v for k, v in task_vector.items()}
            intervention_fn = self.intervention_function(config, layer_hook_names, tv_indices, task_vector, self.model.device, self.forward_model_dict)
            activation, logits = self.write_and_read_hidden(input_ids, layer_hook_names, layer_hook_names, intervention_fn)
            new_task_vector = {l: self.select_task_vector(input_mask, activation[self.num2layer(l)], len(demon_list)) for l in layer_indices}

        if return_mode == 'logits':
            return logits
        elif return_mode == 'tv':
            return new_task_vector
        else:
            raise NotImplementedError

    def write_hidden(self, query, task_vector, demon_list=None, intervention_mode='replace', format_dict={}):
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == "project"], dtype=torch.long)
        config = {"intervention_mode": intervention_mode}
        layer_indices = list(task_vector.keys())
        layer_hook_names = [self.num2layer(x) for x in layer_indices]
        task_vector = {self.num2layer(k): v for k, v in task_vector.items()}
        intervention_fn = self.intervention_function(config, layer_hook_names, tv_indices, task_vector, self.model.device)
        _, logits = self.write_and_read_hidden(input_ids, [], layer_hook_names, intervention_fn)
        return logits

    def write_wo_outputs(self, query, task_vector, demon_list=None, intervention_mode='replace', format_dict={}):
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == "project"], dtype=torch.long)
        config = {"intervention_mode": intervention_mode}
        layer_indices = list(task_vector.keys())
        layer_hook_names = [self.num2attn(x) for x in layer_indices]
        task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
        intervention_fn = self.intervention_function(config, layer_hook_names, tv_indices, task_vector, self.model.device)
        _, logits = self.write_and_read_activation(input_ids, [], layer_hook_names, intervention_fn)
        return logits

    def intervention_function(self, config, layer_hook_names, task_vector_indices, ICL_vector, device, forward_model=None):
        if config["intervention_mode"].startswith("merge") or config["intervention_mode"].startswith("add"):
            lam1 = eval(config["intervention_mode"].split('#')[1])
            lam2 = eval(config["intervention_mode"].split('#')[2])
            is_norm = config["intervention_mode"].startswith("merge")

        def replace(output, layer_name):
            if layer_name in layer_hook_names:
                if forward_model:
                    replace_tv = ICL_vector[layer_name].to(device)
                    out_proj = forward_model[layer_name].weight
                    replace_tv = torch.matmul(replace_tv, out_proj.T).unsqueeze(0)
                else:
                    replace_tv = ICL_vector[layer_name].to(device).unsqueeze(0)
                if isinstance(output, tuple):
                    output[0][:, task_vector_indices] = replace_tv.expand(output[0].shape[0], -1, -1)
                else:
                    output[:, task_vector_indices] = replace_tv.expand(output.shape[0], -1, -1)
            return output


        def merge(ori_output, layer_name):
            def norm(before, after):
                before_num = (before * before).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                after_num = (after * after).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                return after * before_num / (after_num + 1e-8)

            if layer_name in layer_hook_names:
                if isinstance(ori_output, tuple):
                    output = ori_output[0]
                else:
                    output = ori_output
                if forward_model:
                    out_proj = forward_model[layer_name].weight
                    merge_vector = lam1 * output[:, task_vector_indices] + \
                                   lam2 * torch.matmul(ICL_vector[layer_name].to(device), out_proj.T).unsqueeze(0).expand(output.shape[0], -1, -1)
                else:
                    merge_vector = lam1 * output[:, task_vector_indices] + \
                                   lam2 * ICL_vector[layer_name].to(device).unsqueeze(0).expand(output.shape[0], -1, -1)
                if is_norm:
                    output[:, task_vector_indices] = norm(output[:, task_vector_indices], merge_vector)
                else:
                    output[:, task_vector_indices] = merge_vector

            return ori_output

        if config["intervention_mode"] == "replace":
            return replace
        elif config["intervention_mode"].startswith("merge") or config["intervention_mode"].startswith("add"):
            return merge
        else:
            raise NotImplementedError

    def write_wo_inputs(self, query, task_vector, demon_list=None, intervention_mode='replace', format_dict={}):
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == "project"], dtype=torch.long)
        config = {"intervention_mode": intervention_mode}
        layer_indices = list(task_vector.keys())
        layer_hook_names = [self.num2attn(x) for x in layer_indices]
        task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
        intervention_fn = self.intervention_function_wo_input(config, layer_hook_names, tv_indices, task_vector, self.model.device, self.forward_model_dict)
        _, logits = self.write_and_read_activation(input_ids, [], layer_hook_names, intervention_fn)
        return logits

    def intervention_function_wo_input(self, config, layer_hook_names, task_vector_indices, ICL_vector, device, forward_model):
        if config["intervention_mode"].startswith("merge") or config["intervention_mode"].startswith("add"):
            lam1 = eval(config["intervention_mode"].split('#')[1])
            lam2 = eval(config["intervention_mode"].split('#')[2])
            is_norm = config["intervention_mode"].startswith("merge")

        def replace(output, layer_name, inputs):
            if layer_name in layer_hook_names:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                replace_tv = ICL_vector[layer_name].to(device).unsqueeze(0)
                inputs[:, task_vector_indices] = replace_tv.expand(inputs.shape[0], -1, -1)
                output = forward_model[layer_name].forward(inputs)
            return output


        def merge(ori_output, layer_name, inputs):
            def norm(before, after):
                before_num = (before * before).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                after_num = (after * after).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                return after * before_num / (after_num + 1e-8)

            if layer_name in layer_hook_names:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                merge_vector = lam1 * inputs[:, task_vector_indices] + \
                               lam2 * ICL_vector[layer_name].to(device).unsqueeze(0).expand(inputs.shape[0], -1, -1)
                if is_norm:
                    inputs[:, task_vector_indices] = norm(inputs[:, task_vector_indices], merge_vector)
                else:
                    inputs[:, task_vector_indices] = merge_vector
                ori_output = forward_model[layer_name].forward(inputs)
            return ori_output

        if config["intervention_mode"] == "replace":
            return replace
        elif config["intervention_mode"].startswith("merge") or config["intervention_mode"].startswith("add"):
            return merge
        else:
            raise NotImplementedError

    def indirect_effect(self, query, task_vector, demon_list=None, intervention_mode='replace', format_dict={}):
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == "project"], dtype=torch.long, device=self.model.device)
        config = {"intervention_mode": intervention_mode, "num_attention_heads": self.model.config.num_attention_heads}
        indirect_effect_storage = torch.zeros((self.model.config.num_hidden_layers, self.model.config.num_attention_heads, self.model.config.vocab_size), device=self.model.device)

        clean_logits = self.model(input_ids.to(self.model.device)).logits
        clean_prob = F.softmax(clean_logits[0,-1],dim=-1)

        task_vector_torch = torch.zeros(((len(task_vector.keys()), ) + task_vector[0].shape), device=self.model.device)
        for k, v in task_vector.items():
            task_vector_torch[k] = v.to(self.model.device)
        for layer in range(self.model.config.num_hidden_layers):
            layer_hook_names = [self.num2attn(layer)]
            for head_n in range(self.model.config.num_attention_heads):
                intervention_task = [(self.num2attn(layer), layer, head_n, tv_indices)]
                intervention_fn = self.indirect_intervention_function(config, layer_hook_names, intervention_task, task_vector_torch, self.forward_model_dict)
                with torch.no_grad():
                    with TraceDict(self.model, layers=layer_hook_names, retain_input=False, retain_output=False, edit_output=intervention_fn):
                        logits = self.model(input_ids.to(self.model.device)).logits
                probs = torch.softmax(logits[0,-1], dim=-1)  # convert to probability distribution
                indirect_effect_storage[layer, head_n] = (probs - clean_prob)

        return indirect_effect_storage.cpu()

    def indirect_intervention_function(self, config, layer_hook_names, intervention_task, ICL_vector, forward_model):
        num_attention_heads = config['num_attention_heads']

        def replace(output, layer_name, inputs):
            if layer_name in layer_hook_names:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]

                # Determine shapes for intervention
                original_shape = inputs.shape
                view_shape = (num_attention_heads, original_shape[-1] // num_attention_heads)
                new_shape = inputs.size()[:-1] + view_shape  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
                inputs = inputs.view(*new_shape)  # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

                for (layer, layer_n, head_n, tv_indices) in intervention_task:
                    if layer == layer_name:
                        replace_tv = ICL_vector.view(ICL_vector.size()[:-1] + view_shape)
                        inputs[0, tv_indices, head_n] = replace_tv[layer_n, :, head_n]

                inputs = inputs.view(*original_shape)
                out_proj = forward_model[layer_name].weight
                output = torch.matmul(inputs, out_proj.T)
            return output

        if config["intervention_mode"] == "replace":
            return replace
        else:
            raise NotImplementedError

    def compute_topk_effect(self, task_vector, indirect_effect, n_top_heads):
        model_head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        # Compute Top Influential Heads (L,H)
        h_shape = indirect_effect.shape
        topk_vals, topk_inds = torch.topk(indirect_effect.view(-1), k=n_top_heads, largest=True)
        top_heads = list(zip(*np.unravel_index(topk_inds, h_shape), [x.item() for x in topk_vals]))
        task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
        # Compute Function Vector as sum of influential heads
        function_vector = torch.zeros((task_vector[self.num2attn(0)].shape[0], self.model.config.hidden_size), dtype=self.model.dtype)

        for L, H, _ in top_heads[:n_top_heads]:
            out_proj = self.forward_model_dict[self.num2attn(L)]
            x = torch.zeros(task_vector[self.num2attn(L)].shape[0], self.model.config.hidden_size)
            x[:, H * model_head_dim:(H + 1) * model_head_dim] = task_vector[self.num2attn(L)][:, H * model_head_dim:(H + 1) * model_head_dim]
            d_out = out_proj.forward(x.to(self.model.device).to(self.model.dtype)).cpu()
            function_vector += d_out

        return function_vector, top_heads

    def write_head_layer_outputs(self, query, layer_head_task, task_vector, demon_list=None, intervention_mode='replace', format_dict={}):
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == "project"], dtype=torch.long)
        config = {"intervention_mode": intervention_mode}
        layer_indices = set()
        for layer, head in layer_head_task:
            layer_indices.add(layer)
        layer_indices = list(sorted(layer_indices))
        layer_hook_names = [self.num2attn(x) for x in layer_indices]

        # mask
        task_vector_mask = {}
        for layer in layer_indices:
            tv_mask = torch.zeros(self.model.config.hidden_size, dtype=torch.long)
            tv_mask = tv_mask.view(self.model.config.num_attention_heads, -1)
            for cur_layer, head in layer_head_task:
                if cur_layer == layer:
                    tv_mask[head] = 1
            task_vector_mask[self.num2attn(layer)] = tv_mask.view(-1)

        task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
        intervention_fn = self.mask_wo_intervention_function(config, layer_hook_names, tv_indices, task_vector_mask, task_vector, self.model.device, self.forward_model_dict)
        _, logits = self.write_and_read_activation(input_ids, [], layer_hook_names, intervention_fn)
        return logits

    def mask_wo_intervention_function(self, config, layer_hook_names, task_vector_indices, ICL_vector_mask, ICL_vector, device, forward_model):
        if config["intervention_mode"].startswith("merge") or config["intervention_mode"].startswith("add"):
            lam1 = eval(config["intervention_mode"].split('#')[1])
            lam2 = eval(config["intervention_mode"].split('#')[2])
            is_norm = config["intervention_mode"].startswith("merge")
        elif config["intervention_mode"] == "replace":
            lam1 = 0
            lam2 = 1
            is_norm = False

        def merge(ori_output, layer_name, inputs):
            def norm(before, after):
                before_num = (before * before).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                after_num = (after * after).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                return after * before_num / (after_num + 1e-8)

            if layer_name in layer_hook_names:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                ori_inputs = inputs[:, task_vector_indices]
                merge_vector = lam1 * ori_inputs + lam2 * ICL_vector[layer_name].to(device).unsqueeze(0).expand(inputs.shape[0], -1, -1)
                if is_norm:
                    merge_vector = norm(ori_inputs, merge_vector)
                mask = ICL_vector_mask[layer_name].to(device).unsqueeze(0).unsqueeze(0).expand(inputs.shape[0], len(task_vector_indices), -1)
                inputs[:, task_vector_indices] = merge_vector * mask + ori_inputs * (1-mask)
                ori_output = forward_model[layer_name].forward(inputs)
            return ori_output

        return merge

    def intervention_activation_fn(self, config, layer_hook_names, task_vector_indices, ICL_vector, device, forward_model):
        if config["intervention_mode"].startswith("merge") or config["intervention_mode"].startswith("add"):
            lam1 = eval(config["intervention_mode"].split('#')[1])
            lam2 = eval(config["intervention_mode"].split('#')[2])
            is_norm = config["intervention_mode"].startswith("merge")

        def merge(ori_output, layer_name, inputs):
            def norm(before, after):
                before_num = (before * before).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                after_num = (after * after).sum(dim=-1).unsqueeze(-1).expand(-1, -1, after.shape[-1])
                return after * before_num / (after_num + 1e-8)

            if layer_name in layer_hook_names:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                merge_vector = lam1 * inputs[:, task_vector_indices] + \
                               lam2 * ICL_vector[layer_name].to(device).unsqueeze(0).expand(inputs.shape[0], -1, -1)
                if is_norm:
                    inputs[:, task_vector_indices] = norm(inputs[:, task_vector_indices], merge_vector)
                else:
                    inputs[:, task_vector_indices] = merge_vector
                ori_output = forward_model[layer_name].forward(inputs)
            return ori_output

        return merge

    def intervention_activation(self, input_ids, intervention_indices, task_vector, intervention_mode='add#0#1'):
        config = {"intervention_mode": intervention_mode}
        layer_indices = list(task_vector.keys())
        task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
        intervention_fn = self.intervention_activation_fn(config, list(task_vector.keys()), intervention_indices, task_vector, self.model.device, self.forward_model_dict)

        with torch.no_grad():
            with TraceDict(self.model, layers=list(task_vector.keys()), clone=False, detach=False, retain_input=True, retain_output=False, edit_output=intervention_fn) as activations_td:
                logits = self.model(input_ids.to(self.model.device)).logits.cpu()
        activation = {l: activations_td[self.num2attn(l)].input[0].cpu() for l in layer_indices} #Layer: (Len, Hidden)

        return logits[0,-1], activation

    def obtain_activation(self, input_ids, layer_indices=None):
        if layer_indices is None:
            layer_indices = range(self.model.config.num_hidden_layers)
        layer_hook_names = [self.num2attn(l) for l in layer_indices]

        with torch.no_grad():
            with TraceDict(self.model, layers=layer_hook_names, clone=False, detach=False, retain_input=True, retain_output=False) as activations_td:
                logits = self.model(input_ids.to(self.model.device)).logits.cpu()
        activation = {l: activations_td[self.num2attn(l)].input[0].cpu() for l in layer_indices} #Layer: (Len, Hidden)

        return logits[0,-1], activation

    def generate(self, query, demon_list, format_dict={}, return_topk=None):
        if return_topk is None:
            return_topk = 1000000000
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        output_ids = self.model.generate(input_ids.to(self.model.device), top_p=0.9, temperature=0.1, max_new_tokens=64).cpu()
        full_str = self.tokenizer.decode(output_ids.squeeze()[:input_ids.shape[1]])
        gen_str = self.tokenizer.decode(output_ids.squeeze()[input_ids.shape[1]:input_ids.shape[1]+return_topk])
        return full_str, gen_str

    def generate_write(self, query, demon_list, task_vector=None, format_dict={}, intervention_mode='replace', add_to="atten", write_pos=["project"], return_topk=None):
        if return_topk is None:
            return_topk = 1000000000
        input_ids, input_mask = self.format_example_with_length(query, None, demon_list=demon_list, **format_dict)
        tv_indices = torch.cat(
            [
                torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == wp], dtype=torch.long)
                for wp in write_pos
            ], dim=0
        )
        config = {"intervention_mode": intervention_mode}
        layer_indices = list(task_vector.keys())

        if add_to == 'atten':
            layer_hook_names = [self.num2attn(x) for x in layer_indices]
            task_vector = {self.num2attn(k): v for k, v in task_vector.items()}
            intervention_fn = self.intervention_function(config, layer_hook_names, tv_indices, task_vector, self.model.device, self.forward_model_dict)

            with torch.no_grad():
                with TraceDict(self.model, layers=layer_hook_names, clone=False, detach=False, retain_input=True,
                               retain_output=False, edit_output=intervention_fn) as activations_td:
                    output_ids = self.model.generate(input_ids.to(self.model.device), top_p=0.9, temperature=0.1,
                                                     max_new_tokens=16,use_cache=False).cpu()
            full_str = self.tokenizer.decode(output_ids.squeeze()[:input_ids.shape[1]])
            gen_str = self.tokenizer.decode(output_ids.squeeze()[input_ids.shape[1]:input_ids.shape[1] + return_topk])
            return full_str, gen_str
        else:
            raise NotImplementedError


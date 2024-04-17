import os

import torch
from typing import List, Dict




def get_optimizers(config):
    optimizers = []
    for name, opt_config in config.items():
        if name.split('#')[0] == 'none':
            optimizers.append(BaseOptimizer(name))

        elif name.split('#')[0] == 'fixed':
            optimizers.append(FixOptimizer(name, opt_config))

        elif name.split('#')[0] == 'one-step':
            for w in opt_config['weight']:
                for l in opt_config['lr']:
                    optimizers.append(OneStepOptimizer(name + f'#weight{w}-lr{l}', {**opt_config, 'weight': w,'lr':l}))

        elif name.split('#')[0] == 'fix-one-step':
            for w in opt_config['weight']:
                for l in opt_config['lr']:
                    optimizers.append(FixOneStepOptimizer(name + f'#weight{w}-lr{l}', {**opt_config, 'weight': w,'lr':l}))


        elif name.split('#')[0] == 'adag':
            for w in opt_config['weight']:
                for l in opt_config['lr']:
                    optimizers.append(AdaGradOptimizer(name + f'#weight{w}-lr{l}', {**opt_config, 'weight': w, 'lr': l}))

        elif name.split('#')[0] == 'adam':
            for w in opt_config['weight']:
                for l in opt_config['lr']:
                    for ab1 in opt_config['adam_beta1']:
                        for ab2 in opt_config['adam_beta2']:
                            optimizers.append(
                                AdamOptimizer(name + f'#weight{w}-adam{ab1}-squa{ab2}-lr{l}',
                                              {**opt_config, 'weight':w,'lr':l, 'adam_beta1': ab1, 'adam_beta2': ab2})
                            )

        elif name.split('#')[0] == 'rms':
            for w in opt_config['weight']:
                for l in opt_config['lr']:
                    for dr in opt_config['decay_rate']:
                        optimizers.append(
                            RMSOptimizer(name + f'#weight{w}-decay{dr}-lr{l}',
                                         {**opt_config, 'weight':w,'lr':l, 'decay_rate': dr})
                        )
        else:
            raise NotImplementedError
    return optimizers

class BaseOptimizer:
    def __init__(self,name, config=None):
        self.config = config
        self.name = name

    def __call__(self, dev_ICL_vector, test_ICL_vector):
        return test_ICL_vector

class OneStepOptimizer(BaseOptimizer):
    def __call__(self, dev_ICL_vector: List[Dict], test_ICL_vector:Dict):
        # dev_ICL_vector: (example_num, lay_indices)
        # test_ICL_vector: (lay_indices)
        layer_indices = list(test_ICL_vector.keys())
        weight = self.config['weight']
        opt_task_vector = {}
        for layer in layer_indices:
            grad_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            for x,i in enumerate(range(len(grad_vector) - len(weight), len(grad_vector))):
                dx = (grad_vector[i] - grad_vector[i - 1])
                current_delta = dx
                delta_task_vector += current_delta * weight[x] / sum(weight)
            opt_task_vector[layer] = test_ICL_vector[layer] + delta_task_vector * self.config['lr']
        return opt_task_vector

class FixOneStepOptimizer(BaseOptimizer):
    def __call__(self, dev_ICL_vector: List[Dict], test_ICL_vector:Dict):
        # dev_ICL_vector: (example_num, lay_indices)
        # test_ICL_vector: (lay_indices)
        layer_indices = list(test_ICL_vector.keys())
        betas = [1] * (len(self.config['weight']) + 1)
        opt_task_vector = {}
        for layer in layer_indices:
            dev_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype, device=dev_vector[0].device)
            start = len(dev_vector) - len(betas)
            for x, i in enumerate(range(start, len(dev_vector))):
                delta_task_vector += dev_vector[i] * betas[x]
            opt_task_vector[layer] = delta_task_vector / (sum(betas))

        weight = self.config['weight']
        for layer in layer_indices:
            grad_vector = dev_ICL_vector[layer] + [opt_task_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            for x,i in enumerate(range(len(grad_vector) - len(weight), len(grad_vector))):
                dx = (grad_vector[i] - grad_vector[i - 1])
                current_delta = dx
                delta_task_vector += current_delta * weight[x] / sum(weight)
            opt_task_vector[layer] += delta_task_vector * self.config['lr']
        return opt_task_vector

class AdaGradOptimizer(BaseOptimizer):
    def __call__(self, dev_ICL_vector: List[Dict], test_ICL_vector:Dict):
        # dev_ICL_vector: (example_num, lay_indices)
        # test_ICL_vector: (lay_indices)
        layer_indices = list(test_ICL_vector.keys())
        betas = [1] * (len(self.config['weight']) + 1)
        opt_task_vector = {}
        for layer in layer_indices:
            dev_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype, device=dev_vector[0].device)
            start = len(dev_vector) - len(betas)
            for x, i in enumerate(range(start, len(dev_vector))):
                delta_task_vector += dev_vector[i] * betas[x]
            opt_task_vector[layer] = delta_task_vector / (sum(betas))

        weight = self.config['weight']
        for layer in layer_indices:
            grad_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            squared_sum = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            for x,i in enumerate(range(len(grad_vector) - len(weight), len(grad_vector))):
                dx = (grad_vector[i] - grad_vector[i-1])
                squared_sum += dx * dx
                current_delta = dx / (squared_sum.sqrt() + 1e-7)
                delta_task_vector += current_delta * weight[x] / sum(weight)
            opt_task_vector[layer] += delta_task_vector * self.config['lr']
        return opt_task_vector

class RMSOptimizer(BaseOptimizer):
    def __call__(self, dev_ICL_vector: List[Dict], test_ICL_vector:Dict):
        # dev_ICL_vector: (example_num, lay_indices)
        # test_ICL_vector: (lay_indices)
        layer_indices = list(test_ICL_vector.keys())
        decay_rate = self.config['decay_rate'] # when decay_rate is 0, it's AdaGrad
        weight = self.config['weight']
        betas = [1] * (len(self.config['weight']) + 1)
        opt_task_vector = {}
        for layer in layer_indices:
            dev_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype, device=dev_vector[0].device)
            start = len(dev_vector) - len(betas)
            for x, i in enumerate(range(start, len(dev_vector))):
                delta_task_vector += dev_vector[i] * betas[x]
            opt_task_vector[layer] = delta_task_vector / (sum(betas))

        for layer in layer_indices:
            grad_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            squared_sum = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            for x,i in enumerate(range(len(grad_vector) - len(weight), len(grad_vector))):
                dx = (grad_vector[i] - grad_vector[i-1])
                squared_sum = decay_rate * squared_sum + (1 - decay_rate) * dx * dx
                current_delta = dx / (squared_sum.sqrt() + 1e-7)
                delta_task_vector += current_delta * weight[x] / sum(weight)
            opt_task_vector[layer] += delta_task_vector * self.config['lr']
        return opt_task_vector


class AdamOptimizer(BaseOptimizer):
    def __call__(self, dev_ICL_vector: List[Dict], test_ICL_vector:Dict):
        # dev_ICL_vector: (example_num, lay_indices)
        # test_ICL_vector: (lay_indices)
        beta1 = self.config['adam_beta1']
        beta2 = self.config['adam_beta2']
        weight = self.config['weight']
        layer_indices = list(test_ICL_vector.keys())
        betas = [1] * (len(self.config['weight']) + 1)
        opt_task_vector = {}
        for layer in layer_indices:
            dev_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype,
                                            device=dev_vector[0].device)
            start = len(dev_vector) - len(betas)
            for x, i in enumerate(range(start, len(dev_vector))):
                delta_task_vector += dev_vector[i] * betas[x]
            opt_task_vector[layer] = delta_task_vector / (sum(betas))

        for layer in test_ICL_vector:
            grad_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            first_moment = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            second_moment = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype)
            cnt = 0
            for x,i in enumerate(range(len(grad_vector) - len(weight), len(grad_vector))):
                cnt += 1
                dx = (grad_vector[i] - grad_vector[i-1])
                first_moment = beta1 * first_moment + (1-beta1) * dx
                second_moment = beta2 * second_moment + (1-beta2) * dx * dx
                first_unbias = first_moment / (1 - beta1 ** cnt)
                second_unbias = second_moment / (1 - beta2 ** cnt)
                current_delta = first_unbias / (second_unbias.sqrt() + 1e-7)
                delta_task_vector += current_delta * weight[x] / sum(weight)
            opt_task_vector[layer] += delta_task_vector * self.config['lr']
        return opt_task_vector


class FixOptimizer(BaseOptimizer):
    def __call__(self, dev_ICL_vector: List[Dict], test_ICL_vector:Dict):
        # dev_ICL_vector: (example_num, lay_indices)
        # test_ICL_vector: (lay_indices)
        layer_indices = list(test_ICL_vector.keys())
        betas = self.config['beta']
        opt_task_vector = {}
        for layer in layer_indices:
            dev_vector = dev_ICL_vector[layer] + [test_ICL_vector[layer]]
            delta_task_vector = torch.zeros(test_ICL_vector[layer].shape, dtype=test_ICL_vector[layer].dtype, device=dev_vector[0].device)
            start = len(dev_vector) - len(betas)
            for x, i in enumerate(range(start, len(dev_vector))):
                delta_task_vector += dev_vector[i] * betas[x]
            opt_task_vector[layer] = delta_task_vector / (sum(betas))
        return opt_task_vector
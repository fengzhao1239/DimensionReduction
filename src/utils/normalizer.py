import torch
import numpy as np


class Normalizer:
    def __init__(self,params=[],method = '-11',dim=None):
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
        assert isinstance(data, torch.Tensor)

        if len(self.params) == 0:
            dims = self.dim
            if isinstance(dims, int):
                dims = (dims,)
            elif dims is None:
                dims = None
            else:
                dims = tuple(dims)

            if self.method in ['-11', '01']:
                if dims is None:
                    max_val = torch.max(data)
                    min_val = torch.min(data)
                else:
                    max_val = torch.amax(data, dim=dims, keepdim=True)
                    min_val = torch.amin(data, dim=dims, keepdim=True)
                self.params = (max_val, min_val)

            elif self.method == 'ms':
                if dims is None:
                    mean = torch.mean(data)
                    std = torch.std(data)
                else:
                    mean = torch.mean(data, dim=dims, keepdim=True)
                    std = torch.std(data, dim=dims, keepdim=True)
                self.params = (mean, std)

            elif self.method == 'none':
                self.params = None

        return self.fnormalize(data, self.params, self.method)

    def normalize(self, new_data):
        if not new_data.device == self.params[0].device:
            self.params = (
                self.params[0].to(new_data.device),
                self.params[1].to(new_data.device),
            )
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data_norm):
        if not new_data_norm.device == self.params[0].device:
            self.params = (
                self.params[0].to(new_data_norm.device),
                self.params[1].to(new_data_norm.device),
            )
        return self.fdenormalize(new_data_norm, self.params, self.method)

    def get_params(self):
        if self.method == 'ms':
            print('Returning mean and std')
        elif self.method in ['01', '-11']:
            print('Returning max and min')
        elif self.method == 'none':
            print('No normalization applied')
        return self.params

    @staticmethod
    def fnormalize(data, params, method):
        if method == '-11':
            return (data - params[1]) / (params[0] - params[1]) * 2 - 1
        elif method == '01':
            return (data - params[1]) / (params[0] - params[1])
        elif method == 'ms':
            return (data - params[0]) / params[1]
        elif method == 'none':
            return data

    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == '-11':
            return (data_norm + 1) / 2 * (params[0] - params[1]) + params[1]
        elif method == '01':
            return data_norm * (params[0] - params[1]) + params[1]
        elif method == 'ms':
            return data_norm * params[1] + params[0]
        elif method == 'none':
            return data_norm
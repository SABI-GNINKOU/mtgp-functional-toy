#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:14:25 2025

@author: sabi gninkou
"""
# gp_model.py

import torch
import gpytorch
from .composite_kernelmatern import CompositeKernel
from .MultitaskKernelFigp import MultitaskKernelFigpCompatible

class MultiTaskfunctional(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, num_func_inputs, par_f=None):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        self.covar_module = MultitaskKernelFigpCompatible(
            data_covar_module=CompositeKernel(num_func_inputs=num_func_inputs, par_f=par_f),
            num_tasks=num_tasks,
            rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


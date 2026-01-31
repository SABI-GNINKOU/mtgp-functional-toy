#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:28:37 2025

@author: Razak Christophe SABI GNINKOU
"""

# multitask_kernel_figp.py
import torch
from linear_operator import to_linear_operator
from linear_operator.operators import KroneckerProductLinearOperator
import gpytorch
from gpytorch.kernels import Kernel, IndexKernel, ScaleKernel
from typing import Optional

def unique(tensor):
    seen = set()
    unique_rows = []
    for row in tensor:
        
        row_tuple = tuple(row.tolist())
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)
    return torch.stack(unique_rows)

class MultitaskKernelFigpCompatible(Kernel):
    def __init__(
        self,
        data_covar_module: Kernel,
        num_tasks: int,
        rank: Optional[int] = 1,
        task_covar_prior: Optional[gpytorch.priors.Prior] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks,
            batch_shape=self.batch_shape,
            rank=rank,
            prior=task_covar_prior,
        )
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")

        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)

        x1_scalar, x1_functional = x1[..., :1], x1[..., 1:]
        x2_scalar, x2_functional = x2[..., :1], x2[..., 1:]

        x1_scalar = unique(x1_scalar)
        x2_scalar = unique(x2_scalar)
        x1_functional = unique(x1_functional)
        x2_functional = unique(x2_functional)

        
        scalar_kernel = self.data_covar_module.kernels[0]
        functional_kernel = self.data_covar_module.kernels[1]

        covar_x = to_linear_operator(ScaleKernel(scalar_kernel).forward(x1_scalar, x2_scalar, **params))
        covar_f = to_linear_operator(functional_kernel.forward(x1_functional, x2_functional, **params))

        res = KroneckerProductLinearOperator(covar_f, covar_x, covar_i)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def num_outputs_per_input(self, x1, x2):
        return self.num_tasks

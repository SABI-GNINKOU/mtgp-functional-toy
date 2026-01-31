#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:19:16 2025

@author: razak-christophe
"""




import torch
import gpytorch
from tqdm import tqdm  
from gpytorch.priors import GammaPrior



class FunctionalKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_func_inputs, par_f=None, **kwargs):
        super().__init__(**kwargs)

        if par_f is None:
            par_f = [20.0] * num_func_inputs 

        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(
                torch.tensor(par_f, dtype=torch.float).reshape(1, num_func_inputs)
            )
        )

    def forward(self, coeffs1, coeffs2, **params):
        """
        - coeffs1, coeffs2 : Coefficients of PCA/FPCA.
        """
        lengthscale = torch.nn.functional.softplus(self.raw_lengthscale)

       
        diff = coeffs1.unsqueeze(1) - coeffs2.unsqueeze(0)  
        squared_diff = diff ** 2
        squared_lengthscale = lengthscale ** 2
        squared_l2_matrix = torch.sum(
            squared_diff / torch.repeat_interleave(
                squared_lengthscale,
                repeats=int(coeffs1.shape[1] / self.raw_lengthscale.shape[-1])
            ).clamp(min=1e-4),
            dim=-1
        )

       
        r = torch.sqrt(squared_l2_matrix + 1e-12)

        
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=coeffs1.device))
        matern52 = (1.0 + sqrt5 * r + (5.0 / 3.0) * r**2) * torch.exp(-sqrt5 * r)

        return matern52


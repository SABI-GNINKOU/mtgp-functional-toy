#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 18:58:26 2025

@author: razak-christophe
"""

import gpytorch
import torch
import gpytorch

class CustomKernelMatern(gpytorch.kernels.Kernel):
    def __init__(self, sigma2_matern=1.0, theta_matern=0.1, **kwargs):
        super().__init__(**kwargs)

        # Matérn 5/2 Kernel
        self.matern = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        self.matern.outputscale = sigma2_matern
        self.matern.base_kernel.lengthscale = theta_matern

        # Kernel final = seulement le Matérn
        self.combined_kernel = self.matern

    def forward(self, x1, x2, diag=False, **params):
        return self.combined_kernel(x1, x2, diag=diag, **params)

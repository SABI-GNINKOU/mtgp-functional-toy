#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:17:12 2025

@author: razak-christophe
"""


import torch
import gpytorch
from .custom_kernelmatern import CustomKernelMatern
from .functional_kernel import FunctionalKernel


class CompositeKernel(gpytorch.kernels.ProductKernel):
    """
    Noyau combinant :
    - Un noyau scalaire basé sur `MaternKernel + PeriodicKernel`
    - Un noyau fonctionnel basé sur ACP pour les projections fonctionnelles
    """
    def __init__(self,num_func_inputs, par_f):
        super().__init__(
            CustomKernelMatern(),
            FunctionalKernel( num_func_inputs=num_func_inputs, par_f=par_f)
        )

    def forward(self, x1, x2, **params):
        """
        Sépare les features scalaires et fonctionnelles avant d'appliquer les noyaux.
        """
        x1_scalar, x1_functional = x1[..., :1], x1[..., 1:]
        x2_scalar, x2_functional = x2[..., :1], x2[..., 1:]

        k_scalar = self.kernels[0](x1_scalar, x2_scalar, **params)
        k_functional = self.kernels[1](x1_functional, x2_functional, **params)

        return k_scalar * k_functional


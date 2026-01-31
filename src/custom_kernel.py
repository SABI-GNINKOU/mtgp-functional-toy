

import gpytorch
import torch
import gpytorch


class CustomKernel(gpytorch.kernels.Kernel):
    def __init__(self, sigma2_matern=1, theta_matern=1.5,
                  sigma2_periodic=1, theta_periodic=0.5, period=1, **kwargs):
        super().__init__(**kwargs)

        self.matern = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        self.matern.outputscale = torch.tensor(sigma2_matern)
        self.matern.base_kernel.lengthscale = torch.tensor(theta_matern)

        self.periodic = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
        )
        self.periodic.outputscale = torch.tensor(sigma2_periodic)
        self.periodic.base_kernel.lengthscale = torch.tensor(theta_periodic)
        self.periodic.base_kernel.period_length = torch.tensor(period)

        self.combined_kernel = self.matern + self.periodic

    def forward(self, x1, x2, diag=False, **params):
        return self.combined_kernel(x1, x2, diag=diag, **params)




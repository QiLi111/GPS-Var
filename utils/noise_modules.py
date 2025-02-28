#!/usr/bin/env python3

import warnings
from typing import Any, Optional, Union

import torch
from linear_operator.operators import ConstantDiagLinearOperator, DiagLinearOperator, LinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.nn import Parameter

from gpytorch import settings
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.module import Module
from linear_operator.utils.warnings import NumericalWarning


class Noise_adapted(Module):
    def __call__(
        self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any
    ) -> Union[Tensor, LinearOperator]:
        # For corredct typing
        return super().__call__(*params, shape=shape, **kwargs)


class _HomoskedasticNoiseBase_adapted(Noise_adapted):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), num_tasks=1,**kwargs: Any):
        super().__init__()
        self.kwargs = kwargs
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        self.register_parameter(name=kwargs['para_name'], parameter=Parameter(torch.zeros(*batch_shape, num_tasks)))
        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, self._noise_param, self._noise_closure)

        self.register_constraint(kwargs['para_name'], noise_constraint)

    def _noise_param(self, m):
        return m.noise

    def _noise_closure(self, m, v):
        return m._set_noise(v)

    @property
    def noise(self):
        return getattr(self,self.kwargs['para_name']+'_constraint').transform(getattr(self,self.kwargs['para_name']))

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self._set_noise(value)

    def _set_noise(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(getattr(self,self.kwargs['para_name']))
        self.initialize(**{self.kwargs['para_name']:getattr(self,self.kwargs['para_name']+'_constraint').inverse_transform(value)})

    def forward(self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any) -> DiagLinearOperator:
        """In the homoskedastic case, the parameters are only used to infer the required shape.
        Here are the possible scenarios:
        - non-batched noise, non-batched input, non-MT -> noise_diag shape is `n`
        - non-batched noise, non-batched input, MT -> noise_diag shape is `nt`
        - non-batched noise, batched input, non-MT -> noise_diag shape is `b x n` with b' the broadcasted batch shape
        - non-batched noise, batched input, MT -> noise_diag shape is `b x nt`
        - batched noise, non-batched input, non-MT -> noise_diag shape is `b x n`
        - batched noise, non-batched input, MT -> noise_diag shape is `b x nt`
        - batched noise, batched input, non-MT -> noise_diag shape is `b' x n`
        - batched noise, batched input, MT -> noise_diag shape is `b' x nt`
        where `n` is the number of evaluation points and `t` is the number of tasks (i.e. `num_tasks` of self.noise).
        So bascially the shape is always `b' x nt`, with `b'` appropriately broadcast from the noise parameter and
        input batch shapes. `n` and the input batch shape are determined either from the shape arg or from the params
        input. For this it is sufficient to take in a single `shape` arg, with the convention that shape[:-1] is the
        batch shape of the input, and shape[-1] is `n`.

        If a "noise" kwarg (a Tensor) is provided, this noise is used directly.
        """
        if "noise" in kwargs:
            return DiagLinearOperator(kwargs.get("noise"))
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        noise = self.noise
        *batch_shape, n = shape
        noise_batch_shape = noise.shape[:-1] if noise.dim() > 1 else torch.Size()
        num_tasks = noise.shape[-1]
        batch_shape = torch.broadcast_shapes(noise_batch_shape, batch_shape)
        noise = noise.unsqueeze(-2)
        noise_diag = noise.expand(*batch_shape, 1, num_tasks).contiguous()
        if num_tasks == 1:
            noise_diag = noise_diag.view(*batch_shape, 1)
        if noise_diag.shape[-1] != 1:
            noise_diag = noise_diag.unsqueeze(-1)
        return ConstantDiagLinearOperator(noise_diag, diag_shape=n)


class HomoskedasticNoise_adapted(_HomoskedasticNoiseBase_adapted):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(),**kwargs: Any):
        super().__init__(noise_prior, noise_constraint, batch_shape, 1,**kwargs)


#!/usr/bin/env python3
# adapted version from gpytorch.likelihoods.bernoulli_likelihood.py, by adding noise

import warnings
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Bernoulli

from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.functions import log_normal_cdf
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood

from gpytorch.priors import Prior
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise
from linear_operator.operators import LinearOperator
from utils.noise_modules import HomoskedasticNoise_adapted



from typing import Any, Optional, Tuple, Union

class BernoulliLikelihood_Base(_OneDimensionalLikelihood):
    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF :math:`\Phi(x)`). Given the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood compactly as:

    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi((2y - 1)f)
        \end{equation*}

    .. note::
        BernoulliLikelihood has an analytic marginal distribution.

    .. note::
        The labels should take values in {0, 1}.
    """

    has_analytic_marginal: bool = True

    def __init__(self, noise_covar: Union[Noise, FixedGaussianNoise], **kwargs: Any) -> None:
        super().__init__()
        self.noise_covar = noise_covar
    
    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Bernoulli:
        output_probs = base_distributions.Normal(0, 1).cdf(function_samples)
        return base_distributions.Bernoulli(probs=output_probs)

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> Bernoulli:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """

        

        mean = function_dist.mean
        if args[0] == 'add3sigmas' or args[0] == 'add' or args[0] == 'noadd':
            full_mean = mean

        elif args[0] == 'add3sigmas3mus':
            num_annos = args[2] # number of annotators
            if len(self.noise)!=num_annos*2:
                raise ValueError("wrong noise & mu length")
            
            mu_valuse = self.noise[num_annos:]
            if len(mu_valuse)!=num_annos:
                raise ValueError("wrong mu length")
            mu_list = []
            for i in args[1]:
                if i == num_annos:
                    # do not add noise
                    mu_list.append(torch.zeros(int(mean.shape[0]/args[3])).to(mean.device))
                else:
                    mu_list.append(mu_valuse[i].expand(int(mean.shape[0]/args[3])))

            full_mean = mean + torch.cat(mu_list).to(mean.device)

        elif args[0] == 'add_1_bias_variance':
            if len(self.noise)!=2:
                raise ValueError("wrong noise & mu length")
            
            mu_valuse = self.noise[1:]
            if len(mu_valuse)!=1:
                raise ValueError("wrong mu length")
            mu_list = []
            for i in range(len(args[1])):
                if i ==3:
                    # do not add noise
                    mu_list.append(torch.zeros(403 * 361).to(mean.device))
                else:
                    mu_list.append(mu_valuse.expand(403 * 361))

            full_mean = mean + torch.cat(mu_list).to(mean.device)


        else:
            raise ValueError("Invalid noise type")

        
        
        var = function_dist.variance

        if args[0] == 'add':
            noise_covar = self._shaped_noise_covar(mean.shape, *args, **kwargs)
            full_covar = var + torch.diagonal(noise_covar)

        elif args[0] == 'add3sigmas' or args[0] == 'add3sigmas3mus':

            noise_values = self.noise[:num_annos]
            noise_list = []
            for i in args[1]:
                if i == num_annos:
                    # do not add noise - high quality data, so we do not want to add noise to it
                    noise_list.append(torch.zeros(int(mean.shape[0]/args[3])).to(var.device))
                else:
                    noise_list.append(noise_values[i].expand(int(mean.shape[0]/args[3])))

            noise_covar = torch.cat(noise_list).to(var.device)
            full_covar = var + noise_covar
            # print(self.noise)
        elif args[0] == 'add_1_bias_variance':
            noise_values = self.noise[:1]
            noise_list = []
            for i in range(len(args[1])):
                if i == 3:
                    # do not add noise
                    noise_list.append(torch.zeros(403 * 361).to(var.device))
                else:
                    noise_list.append(noise_values.expand(403 * 361))
        
            noise_covar = torch.cat(noise_list).to(var.device)
            full_covar = var + noise_covar
            # print(self.noise)

        elif args[0] == 'noadd':
            full_covar = var
        else:
            raise ValueError("Invalid noise type")

        link = full_mean.div(torch.sqrt(1 + full_covar))
        output_probs = base_distributions.Normal(0, 1).cdf(link)
        return base_distributions.Bernoulli(probs=output_probs)

    def expected_log_prob(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        if torch.any(observations.eq(-1)):
            # Remove after 1.0
            warnings.warn(
                "BernoulliLikelihood.expected_log_prob expects observations with labels in {0, 1}. "
                "Observations with labels in {-1, 1} are deprecated.",
                DeprecationWarning,
            )
        else:
            observations = observations.mul(2).sub(1)
        # Custom function here so we can use log_normal_cdf rather than Normal.cdf
        # This is going to be less prone to overflow errors
        log_prob_lambda = lambda function_samples: log_normal_cdf(function_samples.mul(observations))
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob


class BernoulliLikelihood_with_Noise(BernoulliLikelihood_Base):
    r"""
    The standard likelihood for classification.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. note::
        BernoulliLikelihood has an analytic marginal distribution.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param kwargs:

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(
        self,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs
        if self.kwargs['addnoise'] == 'add3sigmas':
            noise_covar0 = HomoskedasticNoise_adapted(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape,para_name='raw_noise0'
            )
            noise_covar1 = HomoskedasticNoise_adapted(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape,para_name='raw_noise1'
            )
            noise_covar2 = HomoskedasticNoise_adapted(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape,para_name='raw_noise2'
            )

            noise_covar = torch.nn.ModuleList([noise_covar0, noise_covar1, noise_covar2])
            
        elif self.kwargs['addnoise'] == 'add3sigmas3mus':

            # for anno in range(self.kwargs['num_annotators']):
            noise_covar = torch.nn.ModuleList()

            num_ann = self.kwargs['num_annotators']
            mu_constraint = Interval(-10000, 10000)

            for i in range(num_ann):
                # noise
                noise_i = HomoskedasticNoise_adapted(
                    noise_prior=noise_prior,
                    noise_constraint=noise_constraint,
                    batch_shape=batch_shape,
                    para_name=f'raw_noise{i}'
                )
                noise_covar.append(noise_i)

            for i in range(num_ann):
                # mu
                mu_i = HomoskedasticNoise_adapted(
                    noise_prior=noise_prior,
                    noise_constraint=mu_constraint,
                    batch_shape=batch_shape,
                    para_name=f'raw_mu{i}'
                )

                
                noise_covar.append(mu_i)


        elif self.kwargs['addnoise'] == 'add':
            noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
            )

        elif self.kwargs['addnoise'] == 'add_1_bias_variance':
            noise_covar0 = HomoskedasticNoise_adapted(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape,para_name='raw_noise0'
            )
            mu_constraint = Interval(-10000, 10000)
            mu_0 = HomoskedasticNoise_adapted(
                noise_prior=noise_prior, noise_constraint=mu_constraint, batch_shape=batch_shape,para_name='raw_mu0'
            )
            noise_covar = torch.nn.ModuleList([noise_covar0,mu_0])


        else:
            raise ValueError("Invalid noise type")
        
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        if self.kwargs['addnoise'] == 'add':
            return self.noise_covar.noise
        elif self.kwargs['addnoise'] == 'add3sigmas':
            return torch.cat((self.noise_covar[0].noise, self.noise_covar[1].noise, self.noise_covar[2].noise))
        elif self.kwargs['addnoise'] == 'add3sigmas3mus':
            return torch.cat([nc.noise for nc in self.noise_covar])

        elif self.kwargs['addnoise'] == 'add_1_bias_variance':
            return torch.cat((self.noise_covar[0].noise, self.noise_covar[1].noise))
        
        else:
            raise ValueError("Invalid noise type")

    @noise.setter
    def noise(self, value: Tensor) -> None:
        if self.kwargs['addnoise'] == 'add':
            self.noise_covar.initialize(noise=value)
        elif self.kwargs['addnoise'] == 'add3sigmas':
            self.noise_covar[0].initialize(noise=value[0])
            self.noise_covar[1].initialize(noise=value[1])
            self.noise_covar[2].initialize(noise=value[2])
        elif self.kwargs['addnoise'] == 'add3sigmas3mus':

            assert len(value) == len(self.noise_covar), \
                "Length of value must match number of noise parameters"

            for nc, v in zip(self.noise_covar, value):
                nc.initialize(noise=v)




        elif self.kwargs['addnoise'] == 'add_1_bias_variance':
            self.noise_covar[0].initialize(noise=value[0])
            self.noise_covar[1].initialize(noise=value[1])

        else:
            raise ValueError("Invalid noise type")

    @property
    def raw_noise(self) -> Tensor:
        if self.kwargs['addnoise'] == 'add':
            return self.noise_covar.raw_noise
        elif self.kwargs['addnoise'] == 'add3sigmas':
            return torch.cat((self.noise_covar[0].raw_noise0, self.noise_covar[1].raw_noise1, self.noise_covar[2].raw_noise2))
        elif self.kwargs['addnoise'] == 'add3sigmas3mus':
            
            num_ann = self.kwargs['num_annotators']

            return torch.cat(
                [getattr(self.noise_covar[i], f'raw_noise{i}').view(-1)
                for i in range(num_ann)]
                +
                [getattr(self.noise_covar[num_ann + i], f'raw_mu{i}').view(-1)
                for i in range(num_ann)]
            )


        
        elif self.kwargs['addnoise'] == 'add_1_bias_variance':
            return torch.cat((self.noise_covar[0].raw_noise0, self.noise_covar[1].raw_mu0))
        else:
            raise ValueError("Invalid noise type")



    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        if self.kwargs['addnoise'] == 'add':
            self.noise_covar.initialize(raw_noise=value)
        elif self.kwargs['addnoise'] == 'add3sigmas':
            self.noise_covar[0].initialize(raw_noise0=value[0])
            self.noise_covar[1].initialize(raw_noise1=value[1])
            self.noise_covar[2].initialize(raw_noise2=value[2])
        elif self.kwargs['addnoise'] == 'add3sigmas3mus':
           
            num_ann = self.kwargs['num_annotators']
            assert len(value) == 2 * num_ann

            for i in range(num_ann):
                self.noise_covar[i].initialize(
                    **{f'raw_noise{i}': value[i]}
                )

            for i in range(num_ann):
                self.noise_covar[num_ann + i].initialize(
                    **{f'raw_mu{i}': value[num_ann + i]}
                )


        elif self.kwargs['addnoise'] == 'add_1_bias_variance':
            self.noise_covar[0].initialize(raw_noise0=value[0])
            self.noise_covar[1].initialize(raw_mu0=value[1])

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)
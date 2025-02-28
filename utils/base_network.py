
# adapted from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import math

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, feat_dim, bilinear=False,FeatBN = 'noadd'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feat_dim = feat_dim
        self.FeatBN = FeatBN

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, feat_dim, bilinear))
        if self.FeatBN == 'add':
            self.BN = nn.BatchNorm2d(feat_dim)
        self.outc = (OutConv(feat_dim, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.FeatBN == 'add':
            x= self.BN(x)

        logits = self.outc(x)
        return logits

class UNet_linear(UNet):

    def __init__(self, n_channels, n_classes, feat_dim, bilinear=False,FeatBN = 'noadd',
                 noise_values=None,mu_values=None,
                 lower_bound_sigma = 1e-4, upper_bound_mu = 10000.0, lower_bound_mu = -10000.0):
        super(UNet_linear, self).__init__(n_channels, n_classes*2, feat_dim, bilinear,FeatBN)
        if noise_values is not None:
            self.noise_values = nn.parameter.Parameter(torch.from_numpy(noise_values).float(),requires_grad=True)
        else:
            self.noise_values = nn.parameter.Parameter(torch.zeros(3).float(),requires_grad=True)
        if mu_values is not None:
            self.mu_values = nn.parameter.Parameter(torch.from_numpy(mu_values).float(),requires_grad=True)
        else:
            self.mu_values = nn.parameter.Parameter(torch.zeros(3).float(),requires_grad=True)

        self.lower_bound_sigma = lower_bound_sigma
        self.upper_bound_mu = upper_bound_mu
        self.lower_bound_mu = lower_bound_mu
        self.mu_activation = lambda x,lower_bound, upper_bound: torch.sigmoid(x)*(upper_bound-lower_bound) + lower_bound
        self.Softplus = nn.Softplus()
    
    @property
    def transform_mu(self):
        return self.mu_activation(self.mu_values,self.lower_bound_mu,self.upper_bound_mu)

    @property    
    def transform_sigma(self):
        return self.Softplus(self.noise_values) + self.lower_bound_sigma


class UNetFeatureExtractor(nn.Module):

    def __init__(self, n_channels, n_classes, feat_dim, bilinear=False,FeatBN = 'noadd'):
        super(UNetFeatureExtractor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feat_dim = feat_dim
        self.FeatBN = FeatBN
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, feat_dim, bilinear))
        if self.FeatBN == 'add':
            self.BN = nn.BatchNorm2d(feat_dim)
        self.outc = (OutConv(feat_dim, n_classes))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        features = self.up4(x, x1)

        if self.FeatBN == 'add':
            features = self.BN(features)

        return features
    
""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, grid_bounds=(-10., 10.), grid_size=1):
        # Define the variational distribution for a single task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size ** input_dim
        )

        # Use a GridInterpolationVariationalStrategy for single-task GP
        variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=[grid_bounds]* input_dim,
            variational_distribution=variational_distribution
        )
        super().__init__(variational_strategy)

        # Define the covariance and mean modules
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        # Compute the mean and covariance of the GP
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


    
class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points,hyperparameter_fixed,kernel_type):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        
        # self.noise_observation = add_observation_noise
        # if self.noise_observation == 'homoscedasticity':
        #     self.raw_noise = torch.nn.Parameter(0.1 * torch.ones(1, 1), requires_grad=True)
        #     self.register_constraint("raw_noise", gpytorch.constraints.Positive())
        #     self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))

        
        
        #     variational_strategy = gpytorch.variational.VariationalStrategy(self, 
        #                                                                 inducing_points, 
        #                                                                 variational_distribution, 
        #                                                                 learn_inducing_locations=True,
        #                                                                 jitter_val = self.raw_noise)
        # elif self.noise_observation == 'noadd':
        variational_strategy = gpytorch.variational.VariationalStrategy(self, 
                                                                        inducing_points, 
                                                                        variational_distribution, 
                                                                        learn_inducing_locations=True
                                                                        )

        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        # # Latent noise as a trainable parameter over inducing points
        # latent_noise_init = torch.ones(inducing_points.size(0)) * 1e-2
        # self.latent_noise = torch.nn.Parameter(latent_noise_init)
        


        # if add_noise == 'add':
        #     self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()+ gpytorch.kernels.WhiteNoiseKernel(noise=0.1))
        # elif add_noise == 'noadd':
        #     self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # else:
        #     raise ValueError('add_noise must be either add or noadd')
        
        if kernel_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == 'Cosine':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        elif kernel_type == 'Linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_type == 'RBF_Linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel())

        else:
            raise ValueError('kernel_type must be either RBF or Cosine or Linear')

        # print("output_scale",self.covar_module.outputscale)
        # print("raw_output_scale",self.covar_module.raw_outputscale)
        # print("lengthscale",self.covar_module.base_kernel.lengthscale)
        # print("raw_lengthscale",self.covar_module.base_kernel.raw_lengthscale)
        # print("\n")
        
        if hyperparameter_fixed == 'fixed' and kernel_type == 'RBF':
            print('Hyperparameters fixed')

            # Fix kernel hyperparameters
            self.covar_module.outputscale = 1.0  # Fix variance
            self.covar_module.raw_outputscale.requires_grad = False

            self.covar_module.base_kernel.lengthscale = 2.0  # Fix lengthscale
            self.covar_module.base_kernel.raw_lengthscale.requires_grad = False  # Prevent updates

        elif hyperparameter_fixed == 'fixed' and kernel_type == 'Cosine':
            print('Hyperparameters fixed')

            # Fix kernel hyperparameters
            self.covar_module.outputscale = 1.0
            self.covar_module.raw_outputscale.requires_grad = False

            self.covar_module.base_kernel.period_length = 2.0
            self.covar_module.base_kernel.raw_period_length.requires_grad = False

        elif hyperparameter_fixed == 'fixed' and kernel_type == 'Linear':
            print('Hyperparameters fixed')

            # Fix kernel hyperparameters
            self.covar_module.outputscale = 1.0
            self.covar_module.raw_outputscale.requires_grad = False

            self.covar_module.base_kernel.variance = 2.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        elif hyperparameter_fixed == 'fixed' and kernel_type == 'RBF_Linear':
            print('Hyperparameters fixed')

            # Fix kernel hyperparameters
            self.covar_module.outputscale = 1.0
            self.covar_module.raw_outputscale.requires_grad = False

            self.covar_module.base_kernel.kernels[0].lengthscale = 2.0
            self.covar_module.base_kernel.kernels[0].raw_lengthscale.requires_grad = False

            self.covar_module.base_kernel.kernels[1].variance = 2.0
            self.covar_module.base_kernel.kernels[1].raw_variance.requires_grad = False

        # if hyperparameter_fixed == 'fixed' and add_observation_noise == 'homoscedasticity':
        #     print('Hyperparameters fixed')

            # Fix kernel hyperparameters
            # self.raw_noise = torch.nn.Parameter(0.1 * torch.ones(1, 1), requires_grad=True)
            # self.register_constraint("raw_noise", gpytorch.constraints.Positive())

    # @property
    # def noise(self):
    #     return self.raw_noise_constraint.transform(self.raw_noise)


    def forward(self, x):

        mean_x = self.mean_module(x)
        
        # if self.noise_observation == 'homoscedasticity':
        #     # add a constant term to the covariance and ensure that the noise term is positive
        #     covar_x = self.covar_module(x).add_diag(self.noise)
        # elif self.noise_observation == 'noadd':
        #     covar_x = self.covar_module(x)
        # elif self.noise_observation == 'hetroscedasticity':
        #     print('ToDo')
        covar_x = self.covar_module(x)


        # latent_noise_diag = gpytorch.lazify(torch.diag(self.latent_noise))
        # covar_x = covar_x + latent_noise_diag

        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class DKLModelInducingPts(gpytorch.Module):
    def __init__(self, feature_extractor, inducing_points, hyperparameter_fixed, kernel_type):
        super(DKLModelInducingPts, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPClassificationModel(inducing_points = inducing_points, hyperparameter_fixed = hyperparameter_fixed, kernel_type = kernel_type)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.permute(features,(0,2,3,1))
        features = features.reshape(-1,features.size(-1))
        res = self.gp_layer(features)
        return res
    

class DKLModelGrid(gpytorch.Module):
    def __init__(self, feature_extractor, input_dim, grid_size, grid_bounds=(-10., 10.)):
        super(DKLModelGrid, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(input_dim = input_dim, grid_bounds=grid_bounds,grid_size = grid_size)
        self.grid_bounds = grid_bounds

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        features = torch.permute(features,(0,2,3,1))
        features = features.reshape(-1,features.size(-1))

        res = self.gp_layer(features)
        return res



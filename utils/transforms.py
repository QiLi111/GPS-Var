
# adapted from: https://github.com/sophmrtn/RectAngle

import torch
from torch import nn
import random
from scipy.ndimage import measurements, gaussian_filter
from torchvision.transforms import RandomAffine
import numpy as np


# Pre- and post- processing:
#   * Pre-processing:
#     - Z-score normalisation
#   * Augmentation:
#     - Affine transformation
#     - Noise (speckle and gauss)
#     - Smoothing (gauss only, savitzky-golay in TODO)
#     - Flip (L-R only)
#   * Post-processing:
#     - Binarise (variable threshold)
#     - Keep-largest-connected-component


class z_score(object):
  """ Z-Score normalisation of image. Applied image-wise (not batch-wise) such
  that each image has a mean of 0 and standard deviation of 1.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
  """
  def __init__(self):
    super().__init__()

  def __call__(self, image):
    # batch_ = image.shape[0]
    # for batch_iter_ in range(batch_):
    #   image[batch_iter_,...] = (image[batch_iter_,...] - \
    #                             torch.mean(image[batch_iter_,...]))/ \
    #                             torch.std(image[batch_iter_,...])
      
    image= (image - torch.mean(image))/torch.std(image)
    
    return image


# def z_score(image):
#   """ Z-Score normalisation of image. Applied image-wise (not batch-wise) such
#   that each image has a mean of 0 and standard deviation of 1.
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#   """
#   batch_ = image.shape[0]
#   for batch_iter_ in range(batch_):
#     image[batch_iter_,...] = (image[batch_iter_,...] - \
#                               torch.mean(image[batch_iter_,...]) / \
#                               torch.std(image[batch_iter_,...]))
#   return image


class Affine(object):
  """ Affine augmentation of image. Wrapper for torchvision RandomAffine.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    prob : int, default = 0.3
           Probability of augmentation occuring at each pass.
    degrees : int, default = 5
              Range of possible rotation (-degrees, +degrees). Set to None for
              no rotation.
    translate : float, default = 0.1
                Range of possible translation. Set to None for no translation.
    scale : tuple, default = (0.9, 1.1)
            Range of possible scaling. Set to None for no scaling.
    shear : int, default = 5
            Range of possible shear rotation (-shear, +shear). Set to None for
            no shear.
  """
  def __init__(self, prob=0.3,\
    degrees=5, translate=0.1,
    scale=(0.9,1.1), shear=5):
    super().__init__() 
    self.prob = prob
    self.degrees = degrees
    self.translate = translate
    self.scale = scale
    self.shear = shear

  def __call__(self, image):
    rand_ = random.uniform(0,1)
    if rand_ < self.prob:
      RandAffine_ = RandomAffine(degrees=self.degrees, translate=(self.translate,self.translate),
                                scale=self.scale, shear=self.shear)
      image = RandAffine_(image)
    return image
           

# def Affine(image, prob=0.7,
#            degrees=10, translate=0.1,
#            scale=(0.9,1.1), shear=10):
#   """ Affine augmentation of image. Wrapper for torchvision RandomAffine.
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#     prob : int, default = 0.7
#            Probability of augmentation occuring at each pass.
#     degrees : int, default = 10
#               Range of possible rotation (-degrees, +degrees). Set to None for
#               no rotation.
#     translate : float, default = 0.1
#                 Range of possible translation. Set to None for no translation.
#     scale : tuple, default = (0.9, 1.1)
#             Range of possible scaling. Set to None for no scaling.
#     shear : int, default = 10
#             Range of possible shear rotation (-shear, +shear). Set to None for
#             no shear.
#   """
#   rand_ = random.uniform(0,1)
#   if rand_ < prob:
#     RandAffine_ = RandomAffine(degrees=degrees, translate=(translate,translate),
#                                scale=scale, shear=shear)
#     image = RandAffine_(image)
#   return image


class SpeckleNoise(object):
  """ Noise augmentation of image. Distribution scaled to image so 
  normalisation not essential.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    type : string, default = 'speckle'
           Type of noise to apply.
           Options:
                  * 'speckle': Multiplicative speckle noise
                  * 'gauss' : Additive gaussian noise
    mean : int, default = 0
           Mean of distribution for random sampling
    sigma : int, default = 0.01
            Standard deviation of distribution for random sampling
    prob : int, default = 0.3
           Probability of augmentation occuring at each pass.
  """
  def __init__(self, type='speckle', mean=0, sigma=0.01, prob=0.3):
    super().__init__()
    self.type = type
    self.mean = mean
    self.sigma = sigma
    self.prob = prob

  def __call__(self, image):
    device = image.device
    rand_ = random.uniform(0,1)
    if rand_ < self.prob:
      max = torch.amax(image, dim=(1,2,3))
      noise = torch.randn(image.shape).to(device)

      for i, max_ in enumerate(max):
        mean_ = max_ * self.mean
        sigma_ = max_ * self.sigma
        noise[i,...] = mean_ + (sigma_**0.5) * noise[i,...]

      if self.type == 'speckle':
        image = noise * image
      elif self.type == 'gauss':
        image = noise + image
      else:
        raise ValueError('Invalid noise type - \
        please enter either "speckle" or "gauss".')
    return image


# def SpeckleNoise(image, type='speckle', mean=0, sigma=1, prob=0.7):
#   """ Noise augmentation of image. Distribution scaled to image so 
#   normalisation not essential.
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#     type : string, default = 'speckle'
#            Type of noise to apply.
#            Options:
#                   * 'speckle': Multiplicative speckle noise
#                   * 'gauss' : Additive gaussian noise
#     mean : int, default = 0
#            Mean of distribution for random sampling
#     sigma : int, default = 1
#             Standard deviation of distribution for random sampling
#     prob : int, default = 0.7
#            Probability of augmentation occuring at each pass.
#   """
#   rand_ = random.uniform(0,1)
#   if rand_ < prob:
#     max = torch.amax(image, dim=(1,2,3))
#     noise = torch.randn(image.shape)

#     for i, max_ in enumerate(max):
#       mean_ = max_ * mean
#       sigma_ = max_ * sigma
#       noise[i,...] = mean_ + (sigma_**0.5) * noise[i,...]

#     if 'speckle':
#       image = noise * image
#     elif 'gauss':
#       image = noise + image
#     else:
#       raise ValueError('Invalid noise type - \
#       please enter either "speckle" or "gauss".')
#   return image
  

class Smooth(object): 
  """ Spatial smoothing of image. Currently only Gaussian smoothing.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    sigma : int, default = 0.01
            Standard deviation of smoothing kernel
    prob : int, default = 0.7
           Probability of augmentation occuring at each pass.
  """
  def __init__(self, sigma=0.01, prob=0.3): 
    super().__init__()
    self.sigma = sigma
    self.prob = prob

  def __call__(self, image):
    # Ideally add Savitzky-Golay filter instead of Gauss
    device = image.device
    rand_ = random.uniform(0,1)
    if rand_ < self.prob:
      max = torch.amax(image, dim=(1,2,3)).to(device)
      noise = torch.randn(image.shape).to(device)

      for i, max_ in enumerate(max):
        sigma_ = max_ * self.sigma
        image_ = torch.squeeze(image[i,0,...]).to(device)
        image_ = image_.detach().cpu().numpy()
        image_smooth_ = gaussian_filter(image_, sigma_)
        image[i,0,...] = torch.tensor(image_smooth_).to(device)
    return image


# def Smooth(image, sigma=1, prob=0.7):
#   """ Spatial smoothing of image. Currently only Gaussian smoothing.
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#     sigma : int, default = 1
#             Standard deviation of smoothing kernel
#     prob : int, default = 0.7
#            Probability of augmentation occuring at each pass.
#   """
#   # Ideally add Savitzky-Golay filter instead of Gauss
#   rand_ = random.uniform(0,1)
#   if rand_ < prob:
#     max = torch.amax(image, dim=(1,2,3))
#     noise = torch.randn(image.shape)

#     for i, max_ in enumerate(max):
#       sigma_ = max_ * sigma_
#       image_ = torch.squeeze(image[i,0,...])
#       image_ = image_.detach().cpu().numpy()
#       image_smooth_ = gaussian_filter(image_, sigma_)
#       image[i,0,...] = torch.tensor(image_smooth_)
#   return image


class Flip(object): 
  """ Randomly flip image in vertical axis (left and right)
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    prob : int, default = 0.3
          Probability of augmentation occuring at each pass.
  """
  def __init__(self, prob=0.3):
    super().__init__()
    self.prob = prob
    
  def __call__(self, image):
    if random.uniform(0,1) < self.prob:
      image = torch.fliplr(image)
      
    if random.uniform(0,1) < self.prob:
      image = torch.flipud(image)
    return image


# def Flip(image, prob=0.7):
#   """ Randomly flip image in vertical axis (left and right)
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#     prob : int, default = 0.7
#            Probability of augmentation occuring at each pass.
#   """
#   rand_ = random.uniform(0,1)
#   if rand_ < prob:
#     image = torch.fliplr(image)
#   return image


class KeepLargestComponent(object):
  """ Remove all regions of label except largest connected component.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
  """
  def __init__(self):
    super().__init__()

  def __call__(self, image):
    device = image.device
    image_batch_ = torch.squeeze(image, dim=1).to(device)
    image_batch_ = image_batch_.detach().cpu().numpy()
    batch_size_ = image_batch_.shape[0]

    for ix_ in range(batch_size_):
      image_ = np.squeeze(image_batch_[ix_,...])
      comp_, feat_ = measurements.label(image_)
      largest_ = (image_ == feat_).astype(int)
      image[ix_,...] = torch.unsqueeze(torch.tensor(largest_),dim=0).to(device)
    
    return image


# def KeepLargestComponent(image):
#   """ Remove all regions of label except largest connected component.
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#   """
#   image_batch_ = torch.squeeze(image, dim=1)
#   image_batch_ = image_batch_.detach().cpu().numpy()
#   batch_size_ = image_batch_.shape[0]

#   for ix_ in range(batch_size_):
#     image_ = np.squeeze(image_batch_[ix_,...])
#     comp_, feat_ = measurements.label(image_)
#     largest_ = (image_ == feat_).astype(int)
#     image[ix_,...] = torch.unsqueeze(torch.tensor(largest_),dim=0)
  
#   return image


class Binary(object):
  """ Convert float tensor to binary based on threshold value.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    threshold : float, default = 0.5
                Value above which all intensities are converted to 1, and
                below which all intensities are converted to 0.
  """
  def __init__(self, threshold=0.5):
    super().__init__()
    self.threshold = threshold

  # TODO: check for torch auto binarising
  def __call__(self, image):
    return (image > self.threshold).int()


# def Binary(image, threshold=0.5):
#   """ Convert float tensor to binary based on threshold value.
#   Input arguments:
#     image : Torch Tensor [B,C,H,W], dtype = int
#     threshold : float, default = 0.5
#                 Value above which all intensities are converted to 1, and
#                 below which all intensities are converted to 0.
#   """
#   return (image > threshold).int()

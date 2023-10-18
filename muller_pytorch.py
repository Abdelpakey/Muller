import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import cv2


class MullerResizer(nn.Module):
  """Learned Laplacian resizer in Keras Layer."""

  def __init__(
      self,
      target_size = None,
 
      antialias = False,
      kernel_size = 5,
      stddev = 1.0,
      num_layers = 2,
      avg_pool = False,
 
      init_weights = None,
      name = 'muller_resizer',
  ):
    """Applies a multilayer Laplacian filter on the input images.

    Args:
      target_size:  A tuple with target diemnsions (target_height,
        target_width).
 
      antialias:  Whether to use antialias in resizer. Only tf2 resizer supports
        this feature.
      kernel_size: Size of the Gaussian filter.
      stddev: An optional float stddev, if provided will be directly used
        otherwise is determined using kernel_size.
      num_layers: Specifies the number of Laplacian layers.
      avg_pool: Whether to apply an average pooling before the base image
        resizer. The average pooling is only effective when input is downsized.
 
      init_weights: Whether to use a pre-trained weights to initialize.
      name: name scope of this layer.
    """
    super(MullerResizer, self).__init__()

    self._target_size = target_size
    # self._base_resize_method = base_resize_method
    self._antialias = antialias
    self._kernel_size = kernel_size
    self._stddev = stddev
    self._num_layers = num_layers
    self._avg_pool = avg_pool
    self._dtype = torch.float32
    self.name = name
    self._init_weights = init_weights

    
    self._weights = []
    self._biases = []
    for layer in range(1, self._num_layers + 1):
 


        weight = nn.Parameter(
            torch.Tensor([self._init_weights[2 * layer - 2]])
            if self._init_weights
            else torch.Tensor([0.0]))

        # Optionally, initialize the weight with a constant value
        if self._init_weights:
                init.constant_(weight, self._init_weights[2 * layer - 2])

 


        bias = nn.Parameter(
            torch.Tensor([self._init_weights[2 * layer - 1]])
            if self._init_weights
            else torch.Tensor([0.0]))

            # Optionally, initialize the bias with a constant value
        if self._init_weights:
                init.constant_(bias, self._init_weights[2 * layer - 1])


        self._weights.append(weight)
        self._biases.append(bias)
        self.conv_depth = nn.Conv2d(3, 3, [1,5], 1, padding_mode='zeros',padding=(0,2),  groups=3, bias=False)
        self.conv_depth2 = nn.Conv2d(3, 3, [5,1], 1, padding_mode='zeros',padding=(2,0),  groups=3, bias=False)
 
 

  def _base_resizer(self, inputs):
                """Base resizer function for MullerResizer."""
                batch_size, num_channels, height, width = inputs.size()

                if self._avg_pool:
                    stride_h = height // self._target_size[0]
                    stride_w = width // self._target_size[1]

                    if stride_h > 1 and stride_w > 1:
                        pooling_shape = (1, stride_h, stride_w, 1)
                        inputs = F.avg_pool2d(inputs, pooling_shape, stride=(stride_h, stride_w), padding=0)

                # Resizing using F.interpolate
                resized_inputs = F.interpolate(inputs, size=self._target_size, mode='bilinear')

                return resized_inputs


  def _gaussian_blur(self, inputs):
    """Gaussian blur function for muller."""
 
 
    stddev =torch.tensor(self._stddev).to(self._dtype)# tf2.cast(self._stddev, self._dtype)
    size = self._kernel_size
    radius = size // 2
    x = torch.arange(-radius, radius + 1).to(self._dtype)#tf2.cast(tf2.range(-radius, radius + 1), self._dtype) # # [-2,-1,0,2] float32
    # blur_filter = tf2.exp(-tf2.pow(x, 2.0) / (2.0 * tf2.pow(stddev, 2.0))) 
    blur_filter = torch.exp(-torch.pow(x, 2.0) / (2.0 * torch.pow(stddev, 2.0)))
    # blur_filter /= tf2.reduce_sum(blur_filter) 
    blur_filter /= torch.sum(blur_filter)
    # cast to dtype
    num_channels = inputs.shape[1]
    blur_h = blur_filter.view(1,1, 1, size)#, 1)# tf2.reshape(blur_filter, [size, 1, 1, 1])
    blur_v = blur_filter.view( 1, 1, size,1)#tf2.reshape(blur_filter, [1, size, 1, 1])
    blur_h = blur_h.repeat(num_channels,1,1, 1)# tf2.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = blur_v.repeat(  num_channels,1,1, 1)#tf2.tile(blur_v, [1, 1, num_channels, 1])
 
    self.conv_depth.weight = nn.Parameter(blur_h,requires_grad=False)
    blurred_py = self.conv_depth(inputs)
    
    self.conv_depth2.weight = nn.Parameter(blur_v,requires_grad=False)
    blurred_py2 = self.conv_depth2(blurred_py)

    #####################################################

 

 
    return blurred_py2


  def forward(self, inputs):
        if inputs.dtype != self._dtype:
            inputs = inputs.to(self._dtype)

        # Creates the base resized image.
        net = self._base_resizer(inputs)

        # Multi Laplacian resizer.
        for weight, bias in zip(self._weights, self._biases):
            # Gaussian blur.
            blurred = self._gaussian_blur(inputs)
            # Residual image
            residual_image = blurred - inputs
            # Resize residual image.
            resized_residual = self._base_resizer(residual_image)
            # Add the residual to the input image.
            net = net + torch.tanh(weight * resized_residual + bias)
            inputs = blurred

        return net
 


def inference(image, model):
 
 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.asarray(image) / 255.
 

    image = np.transpose(image, (2, 0, 1))  # Change to PyTorch format (C, H, W)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)  # Add batch dimension
    # model.eval()
    preds = model(image)
    preds = np.array(preds[0].detach().numpy(), np.float32)
    return  np.array(np.clip(preds, 0.0, 1.0))
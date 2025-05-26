#' @title Temporally Shared Block
#'
#' @description
#' Base module for applying 2D convolutions across each time step in a 5D tensor of shape [B, T, C, H, W].
#' Handles optional masking based on a padding value and supports single-timestep inputs ([B, C, H, W]).
#'
#' Used as a parent class for ConvBlock and DownConvBlock.
#'
#' @param pad_value Optional scalar value (e.g., 0) indicating padding in input tensors.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
TemporallySharedBlock <- nn_module(
  classname = "TemporallySharedBlock",
  
  initialize = function(pad_value = NULL) {
    self$pad_value <- pad_value
    self$out_shape <- NULL
  },
  
  smart_forward = function(x) {
    if (x$ndim == 4) {
      return(self$forward(x))  # Already [B, C, H, W]
    }
    
    # x shape: [B, T, C, H, W]
    b <- x$size(1)
    t <- x$size(2)
    c <- x$size(3)
    h <- x$size(4)
    w <- x$size(5)
    
    x_reshaped <- x$view(c(b * t, c, h, w))  # [B*T, C, H, W]
    
    if (!is.null(self$pad_value)) {
      if (is.null(self$out_shape)) {
        dummy <- torch_zeros_like(x_reshaped)
        self$out_shape <- self$forward(dummy)$shape
      }
      
      # Create mask (then move to correct device)
      pad_mask <- x_reshaped$eq(self$pad_value)$all(dim = 3)$all(dim = 2)$all(dim = 2)
      pad_mask <- pad_mask$to(device = x$device)
      
      if (pad_mask$any()$item()) {
        temp <- torch_ones(self$out_shape, device = x$device) * self$pad_value
        temp[!pad_mask, ..] <- self$forward(x_reshaped[!pad_mask, ..])
        out <- temp
      } else {
        out <- self$forward(x_reshaped)
      }
    } else {
      out <- self$forward(x_reshaped)
    }
    
    # Reshape back to [B, T, C, H, W]
    out <- out$view(c(b, t, out$size(2), out$size(3), out$size(4)))
    return(out)
  }
  
)


#' @title DownConvBlock
#'
#' @description
#' A U-Net-style encoder block that downsamples spatial resolution, then applies a skip connection followed by residual conv layers.
#' Internally uses a stack of ConvLayers. Each timestep is processed independently (weight sharing).
#'
#' @param d_in Number of input channels.
#' @param d_out Number of output channels.
#' @param k Kernel size.
#' @param s Stride.
#' @param p Padding.
#' @param pad_value Optional scalar used for masking padded inputs.
#' @param norm Type of normalization: "batch", "instance", "group", or NULL.
#' @param padding_mode Padding type: "zeros", "reflect", etc.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
DownConvBlock <- nn_module(
  classname = "DownConvBlock",
  inherit = TemporallySharedBlock,
  
  initialize = function(d_in,
                        d_out,
                        k,
                        s,
                        p,
                        pad_value = NULL,
                        norm = "batch",
                        padding_mode = "reflect") {
    self$pad_value <- pad_value
    
    self$down <- ConvLayer(
      nkernels = list(d_in, d_in),
      norm = norm,
      k = k,
      s = s,
      p = p,
      padding_mode = padding_mode
    )
    
    self$conv1 <- ConvLayer(
      nkernels = list(d_in, d_out),
      norm = norm,
      padding_mode = padding_mode
    )
    
    # Adds the skip connection
    self$conv2 <- ConvLayer(
      nkernels = list(d_out, d_out),
      norm = norm,
      padding_mode = padding_mode
    )
  },
  
  # Runs predictor x through U-Net-style encoder
  forward = function(x) {
    out <- self$down(x)
    out <- self$conv1(out)
    out <- out + self$conv2(out)
    out
  }
)


#' @title ConvBlock
#'
#' @description
#' Applies a shared ConvLayer to each time step of a time series tensor. Each temporal slice [B, C, H, W] is
#' processed independently using the same convolutional block.
#'
#' Inherits from TemporallySharedBlock.
#'
#' @param nkernels A list of channel sizes defining the ConvLayer stack (e.g., [32, 64, 128]).
#' @param pad_value Optional scalar padding value.
#' @param norm Type of normalization: "batch", "instance", "group", or NULL.
#' @param last_relu Logical. If TRUE, applies ReLU after final conv layer.
#' @param padding_mode Padding type: "zeros", "reflect", etc.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
ConvBlock <- nn_module(
  classname = "ConvBlock",
  inherit = TemporallySharedBlock,
  
  initialize = function(nkernels,
                        pad_value = NULL,
                        norm = "batch",
                        last_relu = TRUE,
                        padding_mode = "reflect") {
    self$pad_value <- pad_value
    
    # Define the 2d conv layers
    self$conv <- ConvLayer(
      nkernels = nkernels,
      norm = norm,
      last_relu = last_relu,
      padding_mode = padding_mode
    )
  },
  
  # Runs predictor x through the block
  forward = function(x) {
    self$conv(x)
  }
)



##' @title ConvLayer
#'
#' @description
#' Constructs a sequence of 2D convolutional layers with optional normalization and ReLU activations. 
#' Used as a building block for ConvBlock and DownConvBlock.
#'
#' @param nkernels A list of integers specifying the number of channels in each conv layer.
#' @param norm Type of normalization: "batch", "instance", "group", or NULL.
#' @param k Kernel size.
#' @param s Stride.
#' @param p Padding.
#' @param n_groups Number of groups for group normalization (only if norm = "group").
#' @param last_relu Logical. If TRUE, applies ReLU to the final conv layer.
#' @param padding_mode Padding type: "zeros", "reflect", etc.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
ConvLayer <- nn_module(
  classname = "ConvLayer",
  inherit = TemporallySharedBlock,
  
  initialize = function(nkernels,
                        norm = "batch",
                        k = 3,
                        s = 1,
                        p = 1,
                        n_groups = 4,
                        last_relu = TRUE,
                        padding_mode = "reflect") {
    # List of layers to be added sequentially
    layers <- list()
    
    # Choose normalization layer
    get_norm <- function(num_feats) {
      if (norm == "batch") {
        nn_batch_norm2d(num_feats)
      } else if (norm == "instance") {
        nn_instance_norm2d(num_feats)
      } else if (norm == "group") {
        nn_group_norm(num_channels = num_feats, num_groups = n_groups)
      } else {
        NULL
      }
    }
    # Iterate through nkernels and defines the conv layers
    for (i in seq_len(length(nkernels) - 1)) {
      in_ch <- nkernels[[i]]      # Layer i's input channels
      out_ch <- nkernels[[i + 1]] # Layer i's output channels
      
      # Define a torch 2d conv module
      conv <- nn_conv2d(
        in_channels = in_ch,
        out_channels = out_ch,
        kernel_size = k,
        stride = s,
        padding = p,
        padding_mode = padding_mode
      )
      
      # Add layer to the encoder
      layers <- append(layers, list(conv))
      
      # Define the type of normalization the layer will use
      norm_layer <- get_norm(out_ch)
      if (!is.null(norm_layer)) {
        layers <- append(layers, list(norm_layer))
      }
      
      # Append a final relu layer in case one is asked
      if (last_relu || i < length(nkernels) - 1) {
        layers <- append(layers, list(nn_relu()))
      }
    }
    
    self$conv <- nn_sequential(!!!layers)
  },
  
  # Runs predictor x through the layer
  forward = function(x) {
    stopifnot(x$size(2) == self$conv[[1]]$in_channels) # Sanity check
    self$conv(x)                                       # Feedforward step
  }
)

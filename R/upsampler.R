#' @title UpConvBlock
#'
#' @description
#' A U-Net-style decoder block that upsamples spatial resolution using transposed convolutions,
#' applies skip connections from the encoder, and refines the output using residual ConvLayers.
#'
#' @param d_in Number of input channels from the previous decoder stage.
#' @param d_out Number of output channels for the current stage.
#' @param k Kernel size for the transposed convolution.
#' @param s Stride for the transposed convolution.
#' @param p Padding for the transposed convolution.
#' @param norm Normalization method: "batch" or NULL.
#' @param d_skip Number of channels in the skip connection (optional).
#' @param padding_mode Padding type for ConvLayers: "zeros", "reflect", etc.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
UpConvBlock <- nn_module(
  classname = "UpConvBlock",
  
  initialize = function(d_in, d_out, k, s, p, norm = "batch", d_skip = NULL, padding_mode = "reflect") {
    d <- if (is.null(d_skip)) d_out else d_skip
    
    self$skip_conv <- nn_sequential(
      nn_conv2d(d, d, kernel_size = 1),
      if (norm == "batch") nn_batch_norm2d(d) else nn_identity(),
      nn_relu()
    )
    
    self$up <- nn_sequential(
      nn_conv_transpose2d(
        in_channels = d_in,
        out_channels = d_out,
        kernel_size = k,
        stride = s,
        padding = p
      ),
      if (norm == "batch") nn_batch_norm2d(d_out) else nn_identity(),
      nn_relu()
    )
    
    self$conv1 <- ConvLayer(nkernels = c(d_out + d, d_out), norm = norm, padding_mode = padding_mode)
    self$conv2 <- ConvLayer(nkernels = c(d_out, d_out), norm = norm, padding_mode = padding_mode)
  },
  
  forward = function(input, skip) {
    up_out <- self$up(input)
    skip_proj <- self$skip_conv(skip)
    out <- torch_cat(list(up_out, skip_proj), dim = 2)
    out <- self$conv1(out)
    out <- out + self$conv2(out)
    out
  }
)

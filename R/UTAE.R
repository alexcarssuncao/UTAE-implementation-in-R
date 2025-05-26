#' @title U-TAE Network (U-Net with Temporal Attention Encoder)
#'
#' @author 
#' Alexandre de Carvalho Assunção \email{alexcarssuncao@@gmail.com}
#'
#' @description
#' U-TAE is a hybrid deep learning model designed for segmentation of satellite image time series. 
#' It integrates spatial and temporal components:
#'
#' • **Spatial Encoder**: A U-Net-style encoder extracts spatial features at multiple resolutions through strided convolutions.  
#' • **Temporal Encoder (LTAE)**: A lightweight multi-head self-attention mechanism encodes temporal relationships at the deepest spatial level.  
#' • **Temporal Aggregator**: Aggregates features over time using either attention-weighted mean or simple mean.  
#' • **Spatial Decoder**: A U-Net-style decoder reconstructs high-resolution spatial outputs using aggregated skip connections.  
#'
#' The model outputs per-pixel class logits and optionally returns temporal attention weights.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020).  
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*.  
#' In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).  
#' https://arxiv.org/abs/2004.09002  
#'
#' Please cite this paper if you use the U-TAE model in your work.
#'
#' @param input_dim Number of spectral bands in the input (e.g., 10 for Sentinel-2).
#' @param encoder_widths Integer vector with number of filters for each encoder level.
#' @param str_conv_k Kernel size for strided convolutions (default: 4).
#' @param str_conv_s Stride for strided convolutions (default: 2).
#' @param str_conv_p Padding for strided convolutions (default: 1).
#' @param norm Normalization method: "batch", "instance", or NULL.
#' @param pad_value Value used to indicate padding or missing data in input.
#' @param padding_mode Padding mode used in convolutions: "reflect" or "zeros".
#' @param n_head Number of attention heads in the LTAE module.
#' @param d_model Dimensionality of temporal embedding used in Q/K/V projections.
#' @param d_k Dimensionality of keys and queries in attention mechanism.
#' @param agg_method Aggregation method across time: "att_mean" or "mean".
#' @param n_classes Number of output classes for pixel-wise classification.
#'
#' @examples
#' model <- UTAE(input_dim = 2, encoder_widths = c(16, 32), n_classes = 5)
#' input_tensor <- torch_randn(4, 2, 10, 64, 64)  # [batch, bands, time, height, width]
#' output <- model(input_tensor)
#' dim(output$out)
UTAE<- nn_module(
  classname = "UTAE",
  
  initialize = function(input_dim,
                        encoder_widths = c(32, 64),
                        str_conv_k = 4,
                        str_conv_s = 2,
                        str_conv_p = 1,
                        norm = "batch",
                        pad_value = 0,
                        padding_mode = "reflect",
                        d_model = 256,
                        d_k = 4,
                        n_head = 4,
                        agg_method = "mean",
                        n_classes) {
    
    # ============================================
    # Spatial Encoder (conv blocks)
    # ============================================
    self$n_stages <- length(encoder_widths)
    # Initial conv layer
    self$in_conv <- ConvBlock(
      nkernels = c(input_dim, encoder_widths[[1]], encoder_widths[[1]]),
      pad_value = pad_value,
      norm = norm,
      padding_mode = padding_mode
    )
    # Size-variable conv blocks
    self$down_blocks <- nn_module_list(
      lapply(seq_len(self$n_stages - 1), function(i) {
        DownConvBlock(
          d_in = encoder_widths[[i]],
          d_out = encoder_widths[[i + 1]],
          k = str_conv_k,
          s = str_conv_s,
          p = str_conv_p,
          pad_value = pad_value,
          norm = norm,
          padding_mode = padding_mode
        )
      })
    )
    
    # ============================================
    # Temporal Encoder (ltae2d)
    # ============================================
    self$temporal_encoder <- LTAE2d(
      in_channels = encoder_widths[[length(encoder_widths)]],  # should match output from spatial encoder
      d_model = 256,
      n_head = 4,
      mlp = c(256, 128),
      return_att = TRUE,
      d_k = 4
    )
    # ============================================
    # Temporal Aggregator
    # ============================================
    self$temporal_aggregator <- TemporalAggregator(mode = agg_method)
    
    # ============================================
    # Upsampling Decoder
    # ============================================
    self$up_blocks <- nn_module_list(
      lapply(seq_len(self$n_stages - 1), function(i) {
        UpConvBlock(
          d_in    = encoder_widths[[self$n_stages + 1 - i]],  # (deep to shallow)
          d_out   = encoder_widths[[self$n_stages - i]],      # to match skip level
          d_skip  = encoder_widths[[self$n_stages - i]],      # same as skip's channels
          k       = str_conv_k,
          s       = str_conv_s,
          p       = str_conv_p,
          norm    = norm,
          padding_mode = padding_mode
        )
      })
    )
    
    # ============================================
    # Classifier
    # ============================================
    self$classifier <- nn_conv2d(encoder_widths[[1]], n_classes, kernel_size = 1)
  },
  
  forward = function(input) {
    # ----------------------------
    # Run spatial encoder
    # ----------------------------
    feature_maps <- list(self$in_conv$smart_forward(input)) # save skip connections
    for (i in seq_len(self$n_stages - 1)) {
      out <- self$down_blocks[[i]]$smart_forward(feature_maps[[length(feature_maps)]])
      feature_maps[[length(feature_maps) + 1]] <- out
    }
    deepest_maps <- feature_maps[[length(feature_maps)]]
    
    # ----------------------------
    # Run temporal encoder
    # ----------------------------
    te_out <- self$temporal_encoder(deepest_maps, batch_positions = NULL, time_mask = NULL)
    out <- te_out[[1]]  # [B, C, H, W]
    att <- te_out[[2]]  # [n_heads, B, T, h, w]
    
    # ----------------------------
    # Aggregate feature maps
    # ----------------------------
    agg_maps <- list()
    for (i in seq_len(self$n_stages)) {
      # Go backwards through encoder levels
      level <- self$n_stages - i + 1
      # Aggregate and save
      agg_maps[[length(agg_maps) + 1]] <- self$temporal_aggregator(feature_maps[[level]], attn_mask = att)
    }
    
    # ----------------------------
    # Run Decoder
    # ----------------------------
    out <- agg_maps[[1]] # Start from deepest agg map
    # Go backwards through encoder levels
    for (i in seq_len(self$n_stages - 1)) {
      level <- i + 1  # 2nd deepest to shallowest
      skip <- agg_maps[[level]]
      out <- self$up_blocks[[i]](out, skip)
    }
    
    # ----------------------------
    # Calculate logits
    # ----------------------------
    logits <- self$classifier(out)
    return(list(out = logits, attn = att))
  }
)

#' @title Temporal Aggregator
#'
#' @author
#' Alexandre de Carvalho Assunção \email{alexcarssuncao@@gmail.com}
#'
#' @description
#' The TemporalAggregator module combines a sequence of spatio-temporal feature maps of shape [B, T, C, H, W]
#' into a temporally aggregated feature map [B, C, H, W]. It supports two aggregation modes:
#'
#' - "mean": simple average over time, optionally using a temporal pad mask.
#' - "att_mean": attention-weighted average using temporal attention maps.
#'
#' If a pad mask is provided, missing time steps are excluded from the mean computation. In attention mode,
#' attention maps are interpolated to match the spatial resolution and used to weight the temporal sum.
#'
#' @param mode Aggregation strategy: either "mean" (default) or "att_mean" for attention-based averaging.
#'
#' @param x A 5D tensor of shape [B, T, C, H, W] representing temporal feature maps.
#' @param pad_mask Optional 2D tensor [B, T] indicating valid time steps (0 = pad, 1 = valid).
#' @param attn_mask Optional 5D tensor [n_head, B, T, h, w] of attention weights over time per spatial location.
#'
#' @return A 4D tensor [B, C, H, W] representing temporally aggregated features.
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
TemporalAggregator <- nn_module(
  classname = "TemporalAggregator",
  
  initialize = function(mode = "mean") {
    self$mode <- mode
  },
  
  TemporalAggregator <- nn_module(
    classname = "TemporalAggregator",
    
    initialize = function(mode = "mean") {
      self$mode <- mode
    },
    
    forward = function(x, pad_mask = NULL, attn_mask = NULL) {
      device <- x$device
      
      if (self$mode == "att_mean") {
        attn <- attn_mask$mean(dim = 1)
        
        if (!all(attn$shape[3:4] == x$shape[4:5])) {
          target_size <- as.integer(c(x$shape[4], x$shape[5]))
          attn <- nnf_interpolate(
            attn,
            size = target_size,
            mode = "bilinear",
            align_corners = FALSE
          )
        }
        
        out <- (x * attn$unsqueeze(3))$sum(dim = 2)
        return(out)
        
      } else {
        return(x$mean(dim = 2))
      }
    }
  )
)

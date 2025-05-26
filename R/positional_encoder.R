#' @title Positional Encoder
#'
#' @description
#' The PositionalEncoder module computes sinusoidal positional encodings, as introduced in the Transformer architecture,
#' to inject temporal ordering into input sequences. It transforms a [B, T] tensor of positions into a [B, T, d] tensor
#' of position encodings using sine and cosine functions of different frequencies.
#'
#' This encoder is typically added to the temporal input embeddings before applying attention.
#'
#' @param d Dimensionality of the positional encoding.
#' @param T Maximum expected number of time steps (used to scale frequencies).
#' @param repeat_ If set, the resulting positional encoding is repeated along the last dimension.
#' @param offset Optional integer offset to shift the frequency range.
#'
#' @return A tensor of shape [B, T, d] or [B, T, d * repeat_] with sinusoidal positional encodings.
#'
#' @references
#' Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
#' Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.
#' https://arxiv.org/abs/1706.03762
PositionalEncoder <- nn_module(
  classname = "PositionalEncoder",
  
  initialize = function(d, T = 1000, repeat_ = NULL, offset = 0) {
    self$d <- d
    self$T <- T
    self$repeat_ <- repeat_
    self$offset <- offset
    
    # denom = T^(2 * (offset + 0:d-1) // 2 / d)
    denom_idx <- torch_arange(offset, offset + d - 1, dtype = torch_float())
    self$denom_base <- T^(2 * (denom_idx %/% 2) / d)
  },
  
  forward = function(batch_positions) {
    device <- batch_positions$device
    denom <- self$denom_base$to(device = device)
    
    # batch_positions: [B, T]
    # sinusoid_table: [B, T, d]
    sinusoid_table <- batch_positions$unsqueeze(3) / denom$unsqueeze(1)$unsqueeze(1)
    
    # Apply sin to even indices and cos to odd indices
    sinusoid_table[,,seq(1, self$d, by = 2)] <- torch_sin(sinusoid_table[,,seq(1, self$d, by = 2)])
    sinusoid_table[,,seq(2, self$d, by = 2)] <- torch_cos(sinusoid_table[,,seq(2, self$d, by = 2)])
    
    if (!is.null(self$repeat_)) {
      sinusoid_table <- torch_cat(rep(list(sinusoid_table), self$repeat_), dim = -1)
    }
    
    sinusoid_table
  }
)

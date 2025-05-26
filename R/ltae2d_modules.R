#' @title LTAE2d (Lightweight Temporal Attention Encoder 2D)
#'
#' @author
#' Alexandre de Carvalho Assunção \email{alexcarssuncao@@gmail.com}
#'
#' @description
#' The LTAE2d module implements a lightweight temporal attention encoder adapted to 2D spatial grids. 
#' It reshapes spatio-temporal feature maps into pixel-wise time series and applies multi-head self-attention 
#' at each pixel location. It is used in the U-TAE model for encoding temporal dependencies across satellite image time steps.
#'
#' Features are projected via 1x1 convolutions, normalized, and then processed with attention and an MLP head.
#'
#' @param in_channels Number of input channels per pixel (e.g., final encoder output).
#' @param n_head Number of attention heads.
#' @param d_k Dimensionality of each attention head (key/query).
#' @param mlp Integer vector with hidden layer sizes for the MLP head.
#' @param dropout Dropout rate applied after MLP.
#' @param d_model Dimensionality of projected features (input to attention).
#' @param T Maximum time steps (unused unless positional encoding is active).
#' @param return_att If TRUE, also returns temporal attention weights.
#' @param positional_encoding If TRUE, enables positional encoding (not implemented here).
#'
#' @references
#' Sainte Fare Garnot, V., & Landrieu, L. (2020). 
#' *Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*. 
#' https://arxiv.org/abs/2004.09002
LTAE2d <- nn_module(
  classname = "LTAE2d",
  
  initialize = function(in_channels = 128,
                        n_head = 16,
                        d_k = 4,
                        mlp = c(256, 128),
                        dropout = 0.2,
                        d_model = 256,
                        T = 1000,
                        return_att = FALSE,
                        positional_encoding = FALSE) {
    
    self$in_channels <- in_channels
    self$n_head <- n_head
    self$d_k <- d_k
    self$return_att <- return_att
    self$d_model <- if (!is.null(d_model)) d_model else in_channels
    self$mlp_dims <- mlp
    
    # Optional 1x1 conv to project input dim to d_model
    if (!is.null(d_model)) {
      self$inconv <- nn_conv1d(in_channels, self$d_model, kernel_size = 1)
    } else {
      self$inconv <- NULL
    }
    
    stopifnot(mlp[1] == self$d_model)
    
    # Positional encoder placeholder
    self$positional_encoder <- NULL 
    
    # Normalization
    self$in_norm <- nn_group_norm(num_groups = n_head, num_channels = in_channels)
    self$out_norm <- nn_batch_norm1d(num_features = self$d_model)
    
    # Multi-head temporal attention
    self$attention_heads <- MultiHeadAttention(n_head = n_head, d_k = d_k, d_in = self$d_model)
    
    # MLP
    layers <- list()
    for (i in seq_len(length(mlp) - 1)) {
      layers <- append(layers, list(
        nn_linear(mlp[i], mlp[i + 1]),
        nn_batch_norm1d(mlp[i + 1]),
        nn_relu()
      ))
    }
    self$mlp <- nn_sequential(!!!layers)
    
    self$dropout <- nn_dropout(dropout)
  },
  
  forward = function(x, batch_positions = NULL, time_mask = NULL, return_comp = FALSE) {
    sz_b <- x$size(1)
    seq_len <- x$size(2)
    d <- x$size(3)
    h <- x$size(4)
    w <- x$size(5)
    
    # --- Pad Mask ---
    if (!is.null(time_mask)) {
      time_mask <- time_mask$to(device = device)
      time_mask_orig <- time_mask
      time_mask <- time_mask$unsqueeze(-1)$`repeat`(c(1, 1, h))$
        unsqueeze(-1)$`repeat`(c(1, 1, 1, w))$
        permute(c(1, 3, 4, 2))$contiguous()$
        view(c(sz_b * h * w, seq_len))
    }
    
    # --- Reshape to pixel time series: [B, T, D, H, W] → [BHW, T, D]
    out <- x$permute(c(1, 4, 5, 2, 3))$contiguous()$view(c(sz_b * h * w, seq_len, d))
    
    # --- GroupNorm ---
    out <- self$in_norm(out$permute(c(1, 3, 2)))$permute(c(1, 3, 2))
    
    # --- 1x1 conv (optional) ---
    if (!is.null(self$inconv)) {
      out <- self$inconv(out$permute(c(1, 3, 2)))$permute(c(1, 3, 2))
    }
    
    # --- Positional encoding skipped for now ---
    
    # --- Multi-head attention ---
    attn_out <- self$attention_heads(out, time_mask = time_mask, return_comp = return_comp)
    out <- attn_out$output  # Shape: [n_head * B, 1, d_k]
    attn <- attn_out$attn
    
    # --- Merge heads: [n_head, B, d_k] → [BHW, d_model]
    if (out$ndim == 3) {
      out <- out$permute(c(2, 1, 3))$contiguous()$view(c(sz_b * h * w, -1))
    } else {
      out <- out$view(c(sz_b * h * w, -1))  # Already 2D, reshape only
    }
    
    # --- output norm + MLP + dropout ---
    out <- self$out_norm(out)
    out <- self$dropout(self$mlp(out))
    out <- out$view(c(sz_b, h, w, -1))$permute(c(1, 4, 2, 3))
    
    if (self$return_att) {
      attn <- attn$view(c(self$n_head, sz_b, h, w, seq_len))$permute(c(1, 2, 5, 3, 4))
      return(list(out = out, attn = attn))
    } else {
      return(out)
    }
  }
  
)

#' @title Multi-Head Attention Module
#'
#' @description
#' Implements multi-head attention using a learned master query and shared key/value projections. Each head
#' attends over the temporal dimension for a given spatial location. Used inside LTAE2d to encode temporal dependencies.
#'
#' @param d_k Dimension of key and query vectors per head.
#' @param d_in Input feature dimension.
#' @param n_head Number of attention heads.
#'
#' @details
#' A single learnable master query is shared across all time steps per head. Keys are linearly projected 
#' from input, and values are split across heads. Scaled dot-product attention is applied per head.
#'
#' @seealso ScaledDotProductAttention
MultiHeadAttention <- nn_module(
  classname = "MultiHeadAttention",
  
  initialize = function(n_head, d_k, d_in) {
    self$n_head <- n_head # number of attention heads
    self$d_k <- d_k       # dimension per head (key/query)
    self$d_in <- d_in     # input feature dimension
    
    # Master query: [n_head, d_k]
    self$Q <- nn_parameter(torch_empty(n_head, d_k))
    self$Q <- nn_init_normal_(self$Q, mean = 0, std = sqrt(2.0 / d_k))
    
    # Linear projection for keys
    self$fc1_k <- nn_linear(d_in, n_head * d_k)
    self$fc1_k$weight <- nn_init_normal_(self$fc1_k$weight, mean = 0, std = sqrt(2.0 / d_k))
    
    # Scaled dot-product attention
    self$attention <- ScaledDotProductAttention(temperature = sqrt(d_k))
  },
  
  forward = function(v, time_mask = NULL, return_comp = FALSE) {
    device <- v$device
    
    d_k <- self$d_k
    d_in <- self$d_in
    n_head <- self$n_head
    
    sz_b <- v$size(1)     # Batch size
    seq_len <- v$size(2)  # Time steps
    
    # Safely move Q to correct device without reassigning self$Q
    Q <- self$Q$to(device = device)
    
    repeats <- torch_tensor(sz_b, dtype = torch_int64(), device = Q$device)
    q <- Q$unsqueeze(2)$repeat_interleave(repeats, dim = 2)
    
    q <- q$permute(c(2, 1, 3))$contiguous()$view(c(-1, 1, d_k))  # [(n_head * B), 1, d_k]
    
    # Project keys and permute
    k <- self$fc1_k(v)$view(c(sz_b, seq_len, n_head, d_k))
    k <- k$permute(c(3, 1, 2, 4))$contiguous()$view(c(-1, seq_len, d_k))  # [(n_head * B), T, d_k]
    
    # Split and reshape values
    v_split <- v$split(d_in %/% n_head, dim = 3)
    v <- torch_stack(v_split, dim = 1)$view(c(-1, seq_len, d_in %/% n_head))  # [(n_head * B), T, d_k]
    
    # Fix time_mask repeat
    if (!is.null(time_mask)) {
      repeats <- torch_tensor(n_head, dtype = torch_int64(), device = time_mask$device)
      time_mask <- time_mask$repeat_interleave(repeats, dim = 1)
    }
    
    # Run attention
    if (return_comp) {
      attn_out <- self$attention(q = q, k = k, v = v, time_mask = time_mask, return_comp = TRUE)
      output <- attn_out$output
      attn <- attn_out$attn
      comp <- attn_out$comp
    } else {
      attn_out <- self$attention(q = q, k = k, v = v, time_mask = time_mask)
      output <- attn_out$output
      attn <- attn_out$attn
    }
    
    # Final reshape
    attn <- attn$view(c(n_head, sz_b, 1, seq_len))$squeeze(3)              # [n_head, B, T]
    output <- output$view(c(n_head, sz_b, 1, d_in %/% n_head))$squeeze(3)  # [n_head, B, d_out]
    output <- output$permute(c(2, 1, 3))$contiguous()$view(c(sz_b, -1))    # [B, d_in]
    
    if (return_comp) {
      return(list(output = output, attn = attn, comp = comp))
    } else {
      return(list(output = output, attn = attn))
    }
  }
)

#' @title Scaled Dot-Product Attention
#'
#' @description
#' Computes scaled dot-product self-attention: softmax(q k^T / \eqn{\sqrt{d_k}}) * v.
#' A padding mask can be applied to ignore specific time steps.
#'
#' @param q Query tensor of shape [B, T_q, D].
#' @param k Key tensor of shape [B, T_k, D].
#' @param v Value tensor of shape [B, T_k, D].
#' @param temperature Scaling factor for logits (commonly sqrt(d_k)).
#' @param time_mask Optional mask [B, T_k] to ignore specific time steps.
#' @param return_comp If TRUE, also returns the raw (pre-softmax) attention scores.
#'
#' @return A list containing:
#' \item{output}{Attention-weighted result tensor}
#' \item{attn}{Normalized attention weights}
#' \item{comp}{(Optional) Raw unnormalized attention scores}
ScaledDotProductAttention <- nn_module(
  classname = "ScaledDotProductAttention",
  
  initialize = function(temperature, attn_dropout = 0.1) {
    self$temperature <- temperature
    self$dropout <- nn_dropout(attn_dropout)
    self$softmax <- nn_softmax(dim = -1)
  },
  
  forward = function(q, k, v, time_mask = NULL, return_comp = FALSE) {
    device <- q$device
    k <- k$to(device = device)
    v <- v$to(device = device)
    
    q_unsqueezed <- q$unsqueeze(2)
    k_transposed <- k$transpose(2, 3)$unsqueeze(2)
    v_unsqueezed <- v$unsqueeze(2)
    
    attn <- torch_matmul(q_unsqueezed, k_transposed) / self$temperature  # [B, 1, T_q, T_k]
    
    if (!is.null(time_mask)) {
      mask <- time_mask$unsqueeze(2)$unsqueeze(2)$to(device = attn$device)  # [B, 1, 1, T_k]
      attn <- attn$masked_fill(mask, -1e3)
    }
    
    if (return_comp) {
      comp <- attn$clone()
    }
    
    attn <- self$softmax(attn)
    attn <- self$dropout(attn)
    
    output <- torch_matmul(attn, v_unsqueezed)  # [B, 1, T_q, D]
    
    if (return_comp) {
      return(list(output = output, attn = attn$squeeze(2), comp = comp$squeeze(2)))
    } else {
      return(list(output = output, attn = attn$squeeze(2)))
    }
  }
)



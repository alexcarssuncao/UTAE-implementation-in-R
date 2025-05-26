test_that("ScaledDotProductAttention returns correct shapes and masks correctly", {
  torch_manual_seed(42)
  
  batch_size <- 2
  T_q <- 3
  T_k <- 5
  D <- 4
  temperature <- sqrt(D)
  
  q <- torch_randn(batch_size, T_q, D)
  k <- torch_randn(batch_size, T_k, D)
  v <- torch_randn(batch_size, T_k, D)
  
  pad_mask <- torch_zeros(batch_size, T_k)$to(dtype = torch_bool())
  pad_mask[ , T_k] <- TRUE
  
  attn_module <- ScaledDotProductAttention(temperature = temperature)
  result <- attn_module(q = q, k = k, v = v, pad_mask = pad_mask, return_comp = TRUE)
  
  # Shape checks
  expect_s3_class(result$output, "torch_tensor")
  expect_equal(result$output$size(), c(batch_size, 1, T_q, D))
  expect_equal(result$attn$size(),  c(batch_size, T_q, T_k))
  expect_equal(result$comp$size(),  c(batch_size, T_q, T_k))
  
  # Masked keys should be very negative
  masked_vals <- result$comp[ , , T_k]  # [B, T_q]
  expect_true((masked_vals < -100)$all()$item())
})


test_that("MultiHeadAttention returns correct shapes and applies mask", {
  torch_manual_seed(42)
  
  # Hyperparameters
  batch_size <- 2
  seq_len <- 5
  d_in <- 64
  n_head <- 4
  d_k <- 16
  temperature <- sqrt(d_k)
  
  # Dummy input
  v <- torch_randn(batch_size, seq_len, d_in)
  
  # Create mask: mask only last key position
  pad_mask <- torch_zeros(batch_size, seq_len, dtype = torch_bool())
  pad_mask[ , seq_len] <- TRUE  # mask last key position
  
  # Initialize module
  mha <- MultiHeadAttention(n_head = n_head, d_k = d_k, d_in = d_in)
  
  # Run forward pass
  result <- mha(v = v, pad_mask = pad_mask, return_comp = TRUE)
  
  output <- result$output
  attn <- result$attn
  comp <- result$comp
  
  # === SHAPE TESTS ===
  expect_s3_class(output, "torch_tensor")
  expect_equal(output$size(), c(batch_size, d_in))
  
  expect_s3_class(attn, "torch_tensor")
  expect_equal(attn$size(), c(n_head, batch_size, seq_len))
  
  expect_s3_class(comp, "torch_tensor")
  expect_equal(comp$size(), c(n_head * batch_size, 1, seq_len))
  
  # === MASKING TEST ===
  # Extract last key column from comp (should be -1e3)
  masked_vals <- comp[ , 1, seq_len]  # shape: [n_head * batch_size]
  expect_true((masked_vals < -100)$all()$item())
})


test_that("LTAE2d returns correct shapes and applies attention masking", {
  torch_manual_seed(42)
  
  # Dimensions
  batch_size <- 2
  T <- 2
  D <- 128
  H <- 4
  W <- 4
  
  # Instantiate module
  ltae <- LTAE2d(
    in_channels = D,
    n_head = 1,
    d_k = 16,
    mlp = c(256, 128),
    d_model = 256,
    dropout = 0.1,
    return_att = TRUE,
    positional_encoding = FALSE
  )
  
  # Dummy input
  x <- torch_randn(batch_size, T, D, H, W)
  
  # Dummy pad mask: mask last time step
  pad_mask <- torch_zeros(batch_size, T, dtype = torch_bool())
  pad_mask[ , T] <- TRUE
  
  # Forward pass
  result <- ltae(x = x, pad_mask = pad_mask)
  out <- result$out
  attn <- result$attn
  
  # === Output shape ===
  expect_s3_class(out, "torch_tensor")
  expect_equal(out$size(), c(batch_size, 128, H, W))  # Final output: [n_heads, B, D_out, H, W]
  
  # === Attention shape ===
  expect_s3_class(attn, "torch_tensor")
  expect_equal(attn$size(), c(1, batch_size, T, H, W))  # [n_head, B, T, H, W]
  
  # === Mask test: attention on last time step should be ~0
  masked_vals <- attn[ , , T, 1:H, 1:W]
  expect_true((masked_vals < 1e-3)$all()$item())  # all masked values close to 0
})


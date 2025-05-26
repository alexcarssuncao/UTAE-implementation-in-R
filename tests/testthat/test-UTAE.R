library(torch)
library(testthat)
source("dataset_module.R")
source("encoder_module.R")
source("positional_encoder.R")
source("ltae2d_modules.R")
source("temporal_aggregator.R")
source("upsampler.R")
source("UTAE.R")


test_that("UTAE end-to-end returns correct output shape", {
  # Parameters
  batch_size <- 2
  input_dim <- 2
  time_steps <- 12
  height <- 32
  width <- 32
  n_classes <- 5
  
  # Create dummy input
  input_tensor <- torch_randn(batch_size, time_steps, input_dim, height, width)
  
  # Instantiate model
  model <- UTAE(
    input_dim = input_dim,
    encoder_widths = c(32, 64, 128),  # 3-stage encoder
    str_conv_k = 4,
    str_conv_s = 2,
    str_conv_p = 1,
    norm = "batch",
    d_model = 256,
    d_k = 4,
    n_head = 4,
    agg_method = "att_mean",
    n_classes = n_classes
  )
  
  # Forward pass
  result <- model(input_tensor)
  
  # Extract output
  out <- result$out
  attn <- result$attn
  
  # Test shape
  expect_equal(as.integer(out$shape), c(batch_size, n_classes, height, width))
  
  # Optional: Check attention shape (if returned)
  expect_true(length(attn$shape) >= 4)  # [n_head, B, T, H, W] or similar
})

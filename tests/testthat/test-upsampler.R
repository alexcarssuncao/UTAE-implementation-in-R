test_that("UpConvBlock upsamples and fuses skip correctly (U-TAE style)", {
  library(torch)
  
  # Decoder dimensions
  d_in <- 64   # input feature depth (lower resolution)
  d_out <- 32  # output feature depth after upsampling
  d_skip <- 32 # feature depth of skip connection
  
  # Create UpConvBlock
  block <- UpConvBlock(d_in = d_in, d_out = d_out,
                       k = 2, s = 2, p = 0,
                       d_skip = d_skip, norm = "batch")
  
  # Input tensor: [B, C_in, H_in, W_in]
  input <- torch_randn(1, d_in, 8, 8)  # low-res input
  
  # Skip tensor: [B, C_skip, H_skip, W_skip]
  skip <- torch_randn(1, d_skip, 16, 16)  # higher-res skip connection
  
  # Forward pass
  out <- block(input, skip)
  
  # Output should match skip's spatial resolution and be [B, d_out, H, W]
  expect_s3_class(out, "torch_tensor")
  expect_equal(out$shape, c(1, d_out, 16, 16))
  
  # Check that residual connection works and output isn't identical to conv1
  out_direct <- block$conv1(torch_cat(list(block$up(input), block$skip_conv(skip)), dim = 2))
  expect_false(torch_allclose(out, out_direct))
})

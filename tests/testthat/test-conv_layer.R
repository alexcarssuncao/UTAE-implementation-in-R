library(torch)
library(testthat)


test_that("ConvLayer outputs correct shape", {
  
  # Define dummy ConvLayer
  conv_layer <- ConvLayer(
    nkernels = list(2, 16, 32),
    norm = "batch",
    k = 3,
    s = 1,
    p = 1,
    last_relu = TRUE
  )
  
  # Input shape: [batch_size, channels, height, width]
  input <- torch_randn(2, 2, 256, 256)  # [2, 2, 256, 256]
  
  # Run through layer
  output <- conv_layer(input)
  
  # Should result in output with 32 channels, same height and width
  expect_s3_class(output, "torch_tensor")
  expect_equal(output$size(), c(2, 32, 256, 256))
})


test_that("ConvBlock outputs correct shape", {
  
  # Define dummy ConvBlock
  conv_block <- ConvBlock(
    nkernels = list(2, 16, 32, 64),
    pad_value = 0,
    norm = "batch",
    last_relu = TRUE
  )
  
  # Input shape: [batch_size, channels, height, width]
  input <- torch_randn(2, 2, 256, 256)  # [2, 2, 256, 256]
  
  # Run through block
  output <- conv_block(input)
  
  # Should result in output with 32 channels, same height and width
  expect_s3_class(output, "torch_tensor")
  expect_equal(output$size(), c(2, 64, 256, 256))
})


test_that("DownConvBlock downsamples correctly and outputs expected shape", {
  
  block <- DownConvBlock(
    d_in = 2,
    d_out = 32,
    k = 3,
    s = 2,
    p = 1,
    pad_value = 0,
    norm = "batch",
    padding_mode = "reflect"
  )
  
  # Input: [B, C, H, W]
  input <- torch_randn(2, 2, 256, 256)
  
  # Forward pass
  output <- block(input)
  
  # Check output type and shape
  expect_s3_class(output, "torch_tensor")
  expect_equal(output$size(), c(2, 32, 128, 128))  # 256 → 128 due to stride=2
})


test_that("TemporallySharedBlock smart_forward matches manual time loop without batch norm", {
  
  # Dummy temporally shared block using ConvLayer internally
  MySharedBlock <- nn_module(
    classname = "MySharedBlock",
    inherit = TemporallySharedBlock,
    
    initialize = function() {
      super$initialize()
      self$conv <- ConvLayer(nkernels = list(2, 4), norm = "none", last_relu = FALSE)
    },
    
    forward = function(x) {
      self$conv(x)
    }
  )
  
  # Create input: [B, T, C, H, W]
  input <- torch_randn(2, 3, 2, 32, 32)
  block <- MySharedBlock()
  
  # Use smart_forward
  output_smart <- block$smart_forward(input)
  
  # Use manual for-loop over time
  b <- input$size(1)
  t <- input$size(2)
  manual_output <- torch_empty_like(output_smart)
  
  for (i in 1:t) {
    # Slice one time step [B, C, H, W]
    x_slice <- input[, i, ..]
    manual_output[, i, ..] <- block(x_slice)
  }
  
  expect_s3_class(output_smart, "torch_tensor")
  expect_true(torch_allclose(output_smart, manual_output, atol = 1e-5))
})


test_that("TemporallySharedBlock smart_forward matches manual time loop with batch norm", {
  
  # Dummy temporally shared block using ConvLayer internally
  MySharedBlock <- nn_module(
    classname = "MySharedBlock",
    inherit = TemporallySharedBlock,
    
    initialize = function() {
      super$initialize()
      self$conv <- ConvLayer(nkernels = list(2, 4), norm = "batch", last_relu = FALSE)
    },
    
    forward = function(x) {
      self$conv(x)
    }
  )
  
  # Create input: [B, T, C, H, W]
  input <- torch_randn(2, 3, 2, 32, 32)
  block <- MySharedBlock()
  block$eval()
  
  t <- input$size(2)
  
  output_smart <- block$smart_forward(input)
  
  manual_output <- torch_empty_like(output_smart)
  for (i in 1:t) {
    manual_output[, i, ..] <- block(input[, i, ..])
  }
  
  expect_s3_class(output_smart, "torch_tensor")
  expect_true(torch_allclose(output_smart, manual_output, atol = 1e-5))
})


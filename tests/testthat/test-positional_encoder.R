test_that("PositionalEncoder returns a tensor with correct shape", {
  batch_size <- 4
  seq_len <- 10
  d <- 6
  
  encoder <- PositionalEncoder(d = d)
  
  pos <- torch_arange(0, seq_len - 1, dtype = torch_float())$`repeat`(c(batch_size, 1))
  out <- encoder(pos)
  
  expect_s3_class(out, "torch_tensor")
  expect_equal(out$shape, c(batch_size, seq_len, d))
})

test_that("Output contains no NaNs", {
  encoder <- PositionalEncoder(d = 8)
  pos <- torch_arange(0, 9, dtype = torch_float())$`repeat`(c(3, 1))
  out <- encoder(pos)
  
  expect_false(out$isnan()$any()$item())
})

test_that("Repeat argument increases last dimension correctly", {
  d <- 4
  repeat_factor <- 3
  encoder <- PositionalEncoder(d = d, repeat_ = repeat_factor)
  pos <- torch_arange(0, 9, dtype = torch_float())$`repeat`(c(2, 1))
  out <- encoder(pos)
  
  expect_equal(out$shape, c(2, 10, d * repeat_factor))
})

test_that("Encoding varies across time positions", {
  encoder <- PositionalEncoder(d = 4)
  pos <- torch_tensor(matrix(0:9, nrow = 1))$to(dtype = torch_float())
  out <- encoder(pos)
  
  # The encoding at position t=0 and t=1 should differ
  expect_false(torch_allclose(out[1,1,], out[1,2,]))
})

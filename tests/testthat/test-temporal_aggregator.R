test_that("TemporalAggregator (mean) reduces temporal dimension correctly", {
  agg <- TemporalAggregator(mode = "mean")
  
  B <- 2
  T <- 4
  C <- 3
  H <- 5
  W <- 5
  
  x <- torch_randn(B, T, C, H, W)
  out <- agg(x)
  
  expect_s3_class(out, "torch_tensor")
  expect_equal(out$shape, c(B, C, H, W))
  
  # Check that it's actually averaging (manual mean test)
  x_mean <- x$mean(dim = 2)
  expect_true(torch_allclose(out, x_mean))
})


test_that("TemporalAggregator (mean) handles pad_mask correctly", {
  agg <- TemporalAggregator(mode = "mean")
  
  B <- 1
  T <- 3
  C <- 1
  H <- 2
  W <- 2
  
  x <- torch_tensor(array(1:12, dim = c(B, T, C, H, W)), dtype = torch_float())
  pad_mask <- torch_tensor(matrix(c(FALSE, TRUE, FALSE), nrow = B), dtype = torch_bool())
  
  out <- agg(x, pad_mask = pad_mask)
  
  # Only x[1,1,...] and x[1,3,...] should be averaged
  expected <- (x[1,1,,,] + x[1,3,,,]) / 2
  
  expect_true(torch_allclose(out[1,,,], expected, atol = 1e-5))
})


test_that("TemporalAggregator (att_mean) correctly computes weighted temporal average", {
  
  B <- 1
  T <- 2
  C <- 1
  H <- 2
  W <- 2
  n_heads <- 2
  
  x <- torch_tensor(array(c(
    1, 2, 1, 2,   # t = 1
    1, 2, 1, 2    # t = 2
  ), dim = c(B, T, C, H, W)), dtype = torch_float())
  
  # Base attention maps for t=1 and t=2
  attn_t1 <- torch_full(c(H, W), 0.3)
  attn_t2 <- torch_full(c(H, W), 0.7)
  
  # Stack into [T, H, W], then [1, T, H, W]
  base_attn <- torch_stack(list(attn_t1, attn_t2))$unsqueeze(1)
  
  # Repeat for n_heads → [n_heads, B, T, H, W]
  attn_mask <- base_attn$`repeat`(c(n_heads, 1, 1, 1, 1))
  
  # Define aggregator (assumes interpolate is shape-checked inside)
  agg <- TemporalAggregator(mode = "att_mean")
  
  # Forward
  out <- agg(x, attn_mask = attn_mask)
  
  # Expected result: 0.3*1 + 0.7*2 = 1.7
  expected <- torch_full(c(B, C, H, W), 1.7)
  
  # Check
  expect_true(torch_allclose(out, expected, atol = 1e-5))
})


test_that("TemporalAggregator (att_mean) interpolates attention correctly", {
  
  B <- 1
  T <- 2
  C <- 1
  H <- 2
  W <- 2
  n_heads <- 2
  
  # x with alternating pattern — shape [1, 2, 1, 2, 2]
  x <- torch_tensor(array(c(
    1, 1, 1, 1,   # t = 1
    2, 2, 2, 2    # t = 2
  ), dim = c(B, T, C, H, W)), dtype = torch_float())
  
  # Create low-res attention → shape [n_heads, B, T, 1, 1]
  attn_t1 <- torch_full(c(1, 1), 0.25)
  attn_t2 <- torch_full(c(1, 1), 0.75)
  
  base_attn <- torch_stack(list(attn_t1, attn_t2))$unsqueeze(1)  # [T, 1, 1, 1]
  attn_mask <- base_attn$`repeat`(c(n_heads, B, 1, 1, 1))        # [n_heads, B, T, 1, 1]
  
  # Aggregator (assumes correct spatial upsampling logic)
  agg <- TemporalAggregator(mode = "att_mean")
  out <- agg(x, attn_mask = attn_mask)
  
  # Should be 0.25 * t1 + 0.75 * t2 = 1 * x, since x[1,t=1]=x[1,t=2]
  expect_true(torch_allclose(out, x[ ,1,, ,]$clone(), atol = 1e-5))
})


test_that("TemporalAggregator (att_mean) applies pad_mask correctly", {
  B <- 1; T <- 2; C <- 1; H <- 2; W <- 2; n_heads <- 2
  
  # x with distinct values at t=1 and t=2
  x <- torch_tensor(array(c(
    1, 2, 1, 2,   # t = 1
    9, 8, 9, 8    # t = 2
  ), dim = c(1, 2, 1, 2, 2)), dtype = torch_float())
  
  # attn_mask: t=1 = 0.4, t=2 = 0.6 → shape [n_heads, B, T, H, W]
  attn_t1 <- torch_full(c(H, W), 0.4)
  attn_t2 <- torch_full(c(H, W), 0.6)
  base_attn <- torch_stack(list(attn_t1, attn_t2))$unsqueeze(1)  # [T, 1, H, W]
  attn_mask <- base_attn$`repeat`(c(n_heads, B, 1, 1, 1))  # [n_heads, B, T, H, W]
  
  # pad_mask: mask out timestep 2
  pad_mask <- torch_tensor(matrix(c(FALSE, TRUE), nrow = 1), dtype = torch_bool())
  
  # Expected output = x[t=1] * 0.4
  expected <- x[ ,1,, ,] * 0.4
  
  agg <- TemporalAggregator(mode = "att_mean")
  out <- agg(x, attn_mask = attn_mask, pad_mask = pad_mask)
  
  expect_true(torch_allclose(out, expected, atol = 1e-5))
})


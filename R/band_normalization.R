
#' @title Compute Global NDVI and EVI Normalization Statistics
#'
#' @description
#' Computes the global mean and standard deviation for NDVI and EVI values
#' across a set of satellite image patches stored as `.tif` files. The function
#' uses `convert_tif_to_pt_ndvi_evi()` to extract NDVI and EVI tensors from each patch.
#' Zero values are excluded from the statistics to avoid bias from masked or invalid pixels.
#'
#' @param patch_info A data frame with at least one column named `file` containing
#' the relative filenames (e.g., `"patch_00001.tif"`) of the `.tif` image patches
#' to be processed. These files are assumed to be located in `"E:/downloads/segmentation/patches"`.
#'
#' @return A numeric vector of length 4 containing the estimated global mean and standard
#' deviation for NDVI and EVI in the following order:
#' \code{c(ndvi_mean, ndvi_std, evi_mean, evi_std)}.
#'
#' @examples
#' \dontrun{
#' patch_info <- data.frame(file = c("patch_00001.tif", "patch_00002.tif"))
#' band_stats <- normalization(patch_info)
#' }
normalization <- function(patch_info, sample_size = 1000, seed = 42) {
  set.seed(seed)
  
  # Sample rows if there are more than `sample_size`
  if (nrow(patch_info) > sample_size) {
    patch_info <- patch_info[sample(1:nrow(patch_info), sample_size), ]
  }
  
  ndvi_sum <- 0
  ndvi_sq_sum <- 0
  evi_sum <- 0
  evi_sq_sum <- 0
  n_pixels_total_ndvi <- 0
  n_pixels_total_evi <- 0
  
  for (file in patch_info$file) {
    
    # Get tensor: [2, T, H, W]
    tensor <- torch_load(file.path("./data/patches_pt", file))
    
    ndvi_vals <- tensor[1, , , ]$view(-1)
    evi_vals  <- tensor[2, , , ]$view(-1)
    
    # Exclude zero values
    ndvi_vals <- ndvi_vals[ndvi_vals != 0]
    evi_vals  <- evi_vals[evi_vals != 0]
    
    ndvi_sum <- ndvi_sum + ndvi_vals$sum()$item()
    ndvi_sq_sum <- ndvi_sq_sum + (ndvi_vals^2)$sum()$item()
    n_pixels_total_ndvi <- n_pixels_total_ndvi + ndvi_vals$numel()
    
    evi_sum <- evi_sum + evi_vals$sum()$item()
    evi_sq_sum <- evi_sq_sum + (evi_vals^2)$sum()$item()
    n_pixels_total_evi <- n_pixels_total_evi + evi_vals$numel()
  }
  
  ndvi_mean <- ndvi_sum / n_pixels_total_ndvi
  ndvi_std  <- sqrt(ndvi_sq_sum / n_pixels_total_ndvi - ndvi_mean^2)
  
  evi_mean <- evi_sum / n_pixels_total_evi
  evi_std  <- sqrt(evi_sq_sum / n_pixels_total_evi - evi_mean^2)
  
  return(c(ndvi_mean, ndvi_std, evi_mean, evi_std))
}


    

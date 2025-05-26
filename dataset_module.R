PatchDataset <- dataset(
  name = "PatchDataset",
  
  initialize = function(data_dir = "./data/patches_pt",
                        file_names,
                        label_vector,
                        label_map,
                        band_stats = c(0, 1, 0, 1)) {
    self$names <- file_names         # list of .tif file names
    self$labels <- label_vector      # character vector, one label per file
    self$label_map <- label_map      # named list: label string -> int
    self$band_stats <- band_stats    # c(ndvi_mean, ndvi_std, evi_mean, evi_std)
    self$data_dir <- data_dir
  },
  
  .getitem = function(i) {
    # Load tensor: [C, T, H, W]
    x <- torch_load(file.path(self$data_dir, self$names[[i]]))
    
    # Normalize NDVI (channel 1) and EVI (channel 2)
    x[1,,,] <- (x[1,,,] - self$band_stats[1]) / self$band_stats[2]
    x[2,,,] <- (x[2,,,] - self$band_stats[3]) / self$band_stats[4]
    
    # Rearrange to [T, C, H, W]
    x <- x$permute(c(2, 1, 3, 4))
    
    # Convert label string to integer
    y_chr <- self$labels[[i]]
    y_idx <- torch_tensor(self$label_map[[y_chr]], dtype = torch_long())
    
    return(list(x = x$contiguous(), y = y_idx))
  },
  
  .length = function() {
    length(self$names)
  }
)

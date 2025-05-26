library(torch)
source("./R/dataset_module.R")
source("./R/encoder_module.R")
source("./R/positional_encoder.R")
source("./R/ltae2d_modules.R")
source("./R/temporal_aggregator.R")
source("./R/upsampler.R")
source("./R/UTAE.R")
source("./R/band_normalization.R")

# Open patch info file
rds <- read.csv("./data/patches_pt/patch_info.csv")

# Count total number of unique class labels
unique_classes <- sort(unique(unlist(rds$label)))

# Label encoder: char -> int
class_to_index <- setNames(seq_along(unique_classes), unique_classes)
index_to_class <- setNames(unique_classes, seq_along(unique_classes))

patch_names <- rds$file
labels_list <- lapply(rds$label, function(x) {return(x[1])})
band_stats <- normalization(rds, sample_size = 50)

# Training hyperparameters
model <- UTAE(input_dim = 2,
              encoder_widths = c(32, 64, 128),
              str_conv_k = 4,
              str_conv_s = 2,
              str_conv_p = 1,
              norm = "batch",
              pad_value = 0,
              padding_mode = "reflect",
              d_model = 256,
              d_k = 4,
              n_head = 4,
              agg_method = "att_mean",
              n_classes = length(unique_classes))

epochs <- 1
lr <- 0.01
loss_fn <- nn_cross_entropy_loss()
optimizer <- optim_adam(model$parameters, lr = lr)

# Create dataset
ds <- PatchDataset(file_names = patch_names,
                   label_vector = labels_list,
                   label_map = class_to_index,
                   band_stats = band_stats)

# Get train, dev, and test idx
set.seed(42)
#indices <- sample(valid_indices)
indices <- sample(seq_len(length(ds)))
n <- length(indices)
n_train <- floor(0.7 * n)
n_dev   <- floor(0.15 * n)

train_idx <- indices[1:n_train]
dev_idx   <- indices[(n_train + 1):(n_train + n_dev)]
test_idx  <- indices[(n_train + n_dev + 1):n]

# Create the train, dev, and test subdatasets
train_ds <- dataset_subset(ds, indices = train_idx)
dev_ds   <- dataset_subset(ds, indices = dev_idx)
test_ds  <- dataset_subset(ds, indices = test_idx)

# Create dataloaders
batch_size <- 1
train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
dev_dl   <- dataloader(dev_ds,   batch_size = batch_size)

# Check GPU availability
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
print(device)
model <- model$to(device = device)

# Store losses for plotting
losses <- data.frame(epoch = integer(),
                     train_loss = numeric(),
                     dev_loss = numeric())

model <- UTAE(input_dim = 2,
              encoder_widths = c(32, 64, 128),
              str_conv_k = 4,
              str_conv_s = 2,
              str_conv_p = 1,
              norm = "batch",
              pad_value = 0,
              padding_mode = "reflect",
              d_model = 256,
              d_k = 4,
              n_head = 4,
              agg_method = "att_mean",
              n_classes = length(unique_classes))

model <- model$to(device = device)

#
# TRAINING LOOP
#
for (epoch in 1:epochs) {
  start_time <- Sys.time()
  
  model$train()
  acc_train_loss <- 0
  acc_dev_loss <- 0
  train_batch_count <- 0
  dev_batch_count <- 0
  
  # ================================
  # TRAINING
  # ================================
  coro::loop(for (batch in train_dl) {
    optimizer$zero_grad()
    
    x <- batch$x$to(device = device)
    y <- batch$y$to(dtype = torch_long(), device = device)$view(-1)  # shape [B]
    
    logits <- model(x)$out
    logits_avg <- logits$mean(dim = c(3, 4))  # [1, C]
    loss <- loss_fn(logits_avg, y)
     
     loss$backward()
     optimizer$step()
     
     acc_train_loss <- acc_train_loss + loss$item()
     train_batch_count <- train_batch_count + 1
   })

  # ================================
  # VALIDATION
  # ================================
  model$eval()

  coro::loop(for (batch in dev_dl) {
    x <- batch$x$to(device = device)
    y <- batch$y$to(dtype = torch_long(), device = device)$view(-1)

    logits <- model(x)$out
    logits_avg <- logits$mean(dim = c(3, 4))  # [1, C]
    loss <- loss_fn(logits_avg, y)

    acc_dev_loss <- acc_dev_loss + loss$item()
    dev_batch_count <- dev_batch_count + 1
  })

  end_time <- Sys.time()
  cat(sprintf(
    "Epoch %d | Avg Training Loss: %.4f | Avg Validation Loss: %.4f | Time: %.2f sec\n",
    epoch,
    acc_train_loss / train_batch_count,
    acc_dev_loss   / dev_batch_count,
    as.numeric(end_time - start_time, units = "secs")
  ))

  losses <- rbind(
    losses,
    data.frame(epoch = epoch,
               train_loss = acc_train_loss / train_batch_count,
               dev_loss = acc_dev_loss / train_batch_count)
  )
}



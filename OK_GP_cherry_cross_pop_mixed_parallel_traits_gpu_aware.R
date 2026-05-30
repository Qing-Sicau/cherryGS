# =====================================================================================
# Part 0: User Configuration and Environment Setup
# =====================================================================================

# ----------------------------- USER CONFIG -------------------------------------------
# Edit variables here instead of passing environment variables from the shell.

# Trait selection: "fixed" uses TRAITS_TO_RUN_FIXED; "all" analyzes all phenotype columns.
TRAIT_SELECTION_MODE <- "fixed"

TRAITS_TO_RUN_FIXED <- c(
  "BLUP_booming_day",
  "BLUP_fruit_maturation_period",
  "BLUP_fruit_weight",
  "BLUP_longitudinal_diameter",
  "BLUP_transverse_diameter",
  "BLUP_lateral_diameter",
  "BLUP_fruit_shape_index",
  "BLUP_fruit_stalk_length",
  "BLUP_soluble_solid_content",
  "BLUP_titratable_acid",
  "BLUP_ratio_of_TSS_TA"
)

# Parallel strategy:
#   "auto"   : CPU-only -> parallel; GPU detected -> serial, safest default.
#   "serial" : always serial.
#   "cpu"    : force CPU-style parallelism even if nvidia-smi reports GPUs.
PARALLEL_MODE <- "auto"
REQUESTED_N_WORKERS <- 12L

# GPU/CUDA handling:
# When CPU parallelism is used, hide CUDA from TensorFlow inside this R process and workers.
# This avoids repeated cuInit warnings and prevents accidental GPU memory use.
HIDE_CUDA_FOR_CPU_PARALLEL <- TRUE

# Optional TensorFlow GPU detection fallback. Keep FALSE by default because it can initialize
# CUDA and print cuInit warnings on CPU-only nodes with CUDA-enabled TensorFlow builds.
ENABLE_TF_GPU_DETECT <- FALSE

# Low-level math/TensorFlow thread settings inside each R process/worker.
BLAS_THREADS_PER_WORKER <- 1L
TF_INTRAOP_THREADS_PER_WORKER <- 1L
TF_INTEROP_THREADS_PER_WORKER <- 1L
TF_FORCE_GPU_ALLOW_GROWTH_VALUE <- "true"

# Suppress TensorFlow C++ INFO/WARNING logs in CPU-only CUDA-enabled builds.
# 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = filter INFO+WARNING+ERROR.
# Use 2 by default; set to 3 only if you want to hide CUDA cuInit ERROR noise.
TF_CPP_MIN_LOG_LEVEL_VALUE <- "2"

# Suppress Keras warnings about legacy input_shape usage after models are updated to layer_input().
SUPPRESS_KERAS_WARNINGS <- TRUE
# -------------------------------------------------------------------------------------

Sys.setenv(
  OPENBLAS_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  MKL_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  OMP_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  TF_NUM_INTRAOP_THREADS = TF_INTRAOP_THREADS_PER_WORKER,
  TF_NUM_INTEROP_THREADS = TF_INTEROP_THREADS_PER_WORKER,
  TF_FORCE_GPU_ALLOW_GROWTH = TF_FORCE_GPU_ALLOW_GROWTH_VALUE,
  TF_CPP_MIN_LOG_LEVEL = TF_CPP_MIN_LOG_LEVEL_VALUE
)

# Lightweight early GPU check before loading TensorFlow/Keras.
# On CPU-only nodes, hide CUDA from TensorFlow to avoid repeated cuInit warnings.
early_has_visible_gpu <- function() {
  cuda_visible <- Sys.getenv("CUDA_VISIBLE_DEVICES", unset = NA_character_)
  if (!is.na(cuda_visible) && trimws(cuda_visible) %in% c("", "-1")) return(FALSE)
  nvidia_smi <- Sys.which("nvidia-smi")
  if (!nzchar(nvidia_smi)) return(FALSE)
  smi_out <- tryCatch(
    system2(nvidia_smi, args = "-L", stdout = TRUE, stderr = FALSE),
    error = function(e) character(0)
  )
  length(smi_out) > 0 && any(grepl("GPU", smi_out, fixed = TRUE))
}

if (tolower(PARALLEL_MODE) %in% c("auto", "cpu") && HIDE_CUDA_FOR_CPU_PARALLEL && !early_has_visible_gpu()) {
  Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
}
if (isTRUE(SUPPRESS_KERAS_WARNINGS)) options(warn = -1)

# --- Enhanced Logging Function ---
log_message <- function(..., indent = 0) {
  prefix <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ")
  indent_space <- paste(rep(" ", indent * 2), collapse = "")
  message(paste0(prefix, indent_space, ...))
}

log_message("SETUP: Loading required R packages except TensorFlow/Keras...")
# Load reticulate first, bind it to the intended conda environment, then load
# tensorflow/keras3. This prevents reticulate from auto-binding to a different
# Python and avoids TensorFlow initialization before CUDA/thread settings.
required_packages_pre_tf <- c(
  "tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
  "reticulate", "caret", "cowplot", "sommer", "Cairo", "tidyr"
)
suppressPackageStartupMessages({ lapply(required_packages_pre_tf, library, character.only = TRUE) })
log_message("SETUP: Non-TensorFlow packages loaded.")

# =====================================================================================
# Part 1: Find Python Path Before TensorFlow/Keras Loading and Parallelization
# =====================================================================================
log_message("SETUP: Finding Python executable from the 'poly_map' conda environment...")
tryCatch({
  reticulate::use_condaenv("poly_map", required = TRUE)
  python_exe_path <- reticulate::py_config()$python
  log_message("SETUP: Found python executable to be used by workers: ", python_exe_path)
}, error = function(e) {
  log_message("SETUP: ERROR - Could not find the 'poly_map' conda environment. Please ensure it is correctly installed and configured.")
  stop(e)
})

log_message("SETUP: Loading TensorFlow/Keras packages after Python binding...")
suppressPackageStartupMessages({
  library(tensorflow)
  library(keras3)
  library(tfdatasets)
})
log_message("SETUP: TensorFlow/Keras packages loaded.")


# --- Define Global Parameters and Plotting Theme ---
theme_publication <- function(base_size = 12, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10, color = "black"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
    )
}

# Create output directories if they don't exist
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)
if (!dir.exists("./results")) dir.create("./results", recursive = TRUE)
if (!dir.exists("./matrix_cache")) dir.create("./matrix_cache", recursive = TRUE) # Directory for cached matrices
log_message("SETUP: Environment setup complete.")


# =====================================================================================
# Part 2: Load, Clean, and Preprocess Data
# =====================================================================================
log_message("DATA: Loading and preprocessing data...")
t_start_data <- proc.time()

GENOTYPE_FILE  <- "genotype.dosages.tsv"
PHENOTYPE_FILE <- "phenotype_BLUPs.csv"
TRAINING_EPOCHS <- 100

# --- Mixed-subpopulation training design parameters ---
# Mixed 1: NH_HN + 25% HP -> remaining HP
# Mixed 2: NH_HN + 50% HP -> remaining HP
# Mixed 3: HP + 25% NH_HN -> remaining NH_HN
# Mixed 4: HP + 50% NH_HN -> remaining NH_HN
MIXED_TARGET_PROPORTIONS <- c(0.25, 0.50)
MIXED_N_REPEATS <- 20
MIXED_RANDOM_SEED <- 20260519

# --- Load Genotype Data ---
dosage_raw <- read.csv(GENOTYPE_FILE, check.names = FALSE, sep = "\t", header = TRUE)
marker_ids <- paste0(dosage_raw$CHROM, ":", dosage_raw$POS)
dosage_only <- dosage_raw[, 5:ncol(dosage_raw)]
dosage_only_matrix <- as.matrix(dosage_only)
storage.mode(dosage_only_matrix) <- "numeric"
genotypeMatrix_raw <- t(dosage_only_matrix)
rownames(genotypeMatrix_raw) <- colnames(dosage_only)
colnames(genotypeMatrix_raw) <- marker_ids
log_message("DATA: Loaded genotype data with ", nrow(genotypeMatrix_raw), " individuals and ", ncol(genotypeMatrix_raw), " markers.")

# --- Load Phenotype Data ---
pheno_raw <- read.csv(PHENOTYPE_FILE, row.names = 1, check.names = FALSE, sep = ",", header = TRUE)
pheno_df <- as.data.frame(pheno_raw)
ALL_TRAITS_IN_FILE <- colnames(pheno_df)
log_message("DATA: Found ", length(ALL_TRAITS_IN_FILE), " traits in phenotype file: ", paste(ALL_TRAITS_IN_FILE, collapse=", "))

# --- User-selectable traits ---
# TRAIT_SELECTION_MODE and TRAITS_TO_RUN_FIXED are defined in the USER CONFIG section at the top.
resolve_traits_to_run <- function(all_traits, fixed_traits, selection_mode = TRAIT_SELECTION_MODE) {
  selection_mode <- tolower(selection_mode)
  if (!selection_mode %in% c("fixed", "all")) {
    stop("TRAIT_SELECTION_MODE must be one of: fixed, all")
  }

  if (selection_mode == "all") {
    traits_requested <- all_traits
    source_label <- "all traits because TRAIT_SELECTION_MODE = 'all'"
  } else if (length(fixed_traits) > 0) {
    traits_requested <- fixed_traits
    source_label <- "TRAITS_TO_RUN_FIXED"
  } else {
    traits_requested <- all_traits
    source_label <- "all traits because TRAITS_TO_RUN_FIXED is empty"
  }

  traits_requested <- unique(traits_requested)
  missing_traits <- setdiff(traits_requested, all_traits)
  if (length(missing_traits) > 0) {
    stop(
      "The following requested trait(s) were not found in the phenotype file: ",
      paste(missing_traits, collapse = ", "),
      "\nAvailable traits are: ", paste(all_traits, collapse = ", ")
    )
  }

  log_message("TRAITS: Selected ", length(traits_requested), " trait(s) from ", source_label, ": ",
              paste(traits_requested, collapse = ", "))
  return(traits_requested)
}

TRAITS_TO_RUN <- resolve_traits_to_run(ALL_TRAITS_IN_FILE, TRAITS_TO_RUN_FIXED)

# --- Data Alignment and Cleaning ---
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
log_message("DATA: Found ", length(common_individuals), " individuals with both genotype and phenotype data.")
genotypeMatrix <- genotypeMatrix_raw[common_individuals, ]
pheno_df_aligned <- pheno_df[common_individuals, , drop = FALSE]

# --- Handle Missing Genotype Values (Mean Imputation) ---
log_message("DATA: Handling missing genotype values using mean imputation...")
col_means <- colMeans(genotypeMatrix, na.rm = TRUE)
col_means[is.nan(col_means)] <- 0
missing_indices <- which(is.na(genotypeMatrix), arr.ind = TRUE)
genotypeMatrix[missing_indices] <- round(col_means[missing_indices[, 2]])
numIndividuals <- nrow(genotypeMatrix); numMarkers <- ncol(genotypeMatrix)

t_end_data <- proc.time()
log_message("DATA: Data preparation complete: ", numIndividuals, " individuals, ", numMarkers, " SNP markers.")
log_message("DATA: Time elapsed for data loading and processing: ", round((t_end_data - t_start_data)[3], 2), " seconds.")

# =====================================================================================
# Part 3: Split Data into Defined Populations
# =====================================================================================
log_message("DATA: Splitting data into specified populations based on ID prefixes...")
all_ids <- rownames(genotypeMatrix)
pop_NH_HN_ids <- all_ids[startsWith(all_ids, "NH") | startsWith(all_ids, "HN")]
pop_HP_ids <- all_ids[startsWith(all_ids, "HP")]
log_message("  - Population 1 (NH/HN): ", length(pop_NH_HN_ids), " individuals.", indent = 1)
log_message("  - Population 2 (HP): ", length(pop_HP_ids), " individuals.", indent = 1)
if(length(pop_NH_HN_ids) == 0 | length(pop_HP_ids) == 0) {
  stop("One or both defined populations have zero individuals.")
}

# =====================================================================================
# Part 4: Population Structure Analysis via PCA (Trait-Independent)
# =====================================================================================
log_message("PCA: Performing Principal Component Analysis to visualize population structure...")
t_start_pca <- proc.time()
non_zero_var_cols <- which(apply(genotypeMatrix, 2, var, na.rm = TRUE) > 1e-6)
pca_results <- prcomp(genotypeMatrix[, non_zero_var_cols], center = TRUE, scale. = TRUE)
pca_df <- as.data.frame(pca_results$x) %>%
  dplyr::mutate(ID = rownames(.), Population = case_when(ID %in% pop_NH_HN_ids ~ "NH_HN", ID %in% pop_HP_ids ~ "HP", TRUE ~ "Other"))

pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Population)) +
  geom_point(alpha = 0.8, size = 3) +
  labs(
    title = "Population Structure (PCA)",
    x = paste0("PC1 (", round(summary(pca_results)$importance[2, 1] * 100, 2), "%)"),
    y = paste0("PC2 (", round(summary(pca_results)$importance[2, 2] * 100, 2), "%)")
  ) +
  stat_ellipse(aes(group = Population), type = "t", level = 0.95) +
  scale_color_manual(values = c("NH_HN" = "#0072B2", "HP" = "#D55E00", "Other" = "grey50")) +
  theme_publication()

ggsave("plots/PCA_Plot_CrossPop_Structure.pdf", pca_plot, width = 8, height = 6, device = cairo_pdf)
# =====================================================================================
# PCA within-group dispersion analysis
# =====================================================================================

pca_disp_input <- pca_df %>%
  dplyr::filter(Population %in% c("HP", "NH_HN"))

# 1. Distance to group centroid
centroids <- pca_disp_input %>%
  dplyr::group_by(Population) %>%
  dplyr::summarise(
    centroid_PC1 = mean(PC1, na.rm = TRUE),
    centroid_PC2 = mean(PC2, na.rm = TRUE),
    .groups = "drop"
  )

pca_disp <- pca_disp_input %>%
  dplyr::left_join(centroids, by = "Population") %>%
  dplyr::mutate(
    distance_to_centroid = sqrt(
      (PC1 - centroid_PC1)^2 +
      (PC2 - centroid_PC2)^2
    )
  )

centroid_summary <- pca_disp %>%
  dplyr::group_by(Population) %>%
  dplyr::summarise(
    n = dplyr::n(),
    mean_distance_to_centroid = mean(distance_to_centroid, na.rm = TRUE),
    median_distance_to_centroid = median(distance_to_centroid, na.rm = TRUE),
    variance_distance_to_centroid = var(distance_to_centroid, na.rm = TRUE),
    sd_distance_to_centroid = sd(distance_to_centroid, na.rm = TRUE),
    se_distance_to_centroid = sd_distance_to_centroid / sqrt(n),
    .groups = "drop"
  )

# 2. Average pairwise distance within each population
calc_pairwise <- function(df) {
  mat <- as.matrix(df[, c("PC1", "PC2")])
  d <- as.numeric(dist(mat, method = "euclidean"))
  data.frame(
    n_pairwise = length(d),
    average_pairwise_distance = mean(d, na.rm = TRUE),
    median_pairwise_distance = median(d, na.rm = TRUE),
    variance_pairwise_distance = var(d, na.rm = TRUE),
    sd_pairwise_distance = sd(d, na.rm = TRUE)
  )
}

pairwise_summary <- pca_disp_input %>%
  dplyr::group_by(Population) %>%
  dplyr::group_modify(~ calc_pairwise(.x)) %>%
  dplyr::ungroup()

# 3. Combine summaries
pca_dispersion_summary <- centroid_summary %>%
  dplyr::left_join(pairwise_summary, by = "Population")

write.csv(
  pca_dispersion_summary,
  "results/PCA_within_group_dispersion_summary.csv",
  row.names = FALSE
)

write.csv(
  pca_disp,
  "results/PCA_distance_to_centroid_per_sample.csv",
  row.names = FALSE
)

write.csv(
  pca_df,
  "results/PCA_coordinates_CrossPop_Structure.csv",
  row.names = FALSE
)

print(pca_dispersion_summary)

t_end_pca <- proc.time()
log_message("PCA: PCA plot saved as 'plots/PCA_Plot_CrossPop_Structure.pdf'.")
log_message("PCA: Time elapsed for PCA: ", round((t_end_pca - t_start_pca)[3], 2), " seconds.")

# =====================================================================================
# Part 5: Pre-calculate Relationship Matrices and Pedigree
# =====================================================================================
log_message("PREP: Building or loading relationship matrices for the ENTIRE population...")
t_start_matrices <- proc.time()
source("get_DomRel_matrix.R")
parents <- c("HF1", "NZH2", "PJHH")
all_ped_ids <- unique(c(parents, all_ids))
ped_df <- data.frame(ID = all_ped_ids, Sire = 0, Dam = 0, stringsAsFactors = FALSE)
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) { ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) { ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "PJHH" }
}

# --- A-matrix ---
A_full <- Amatrix(ped_df, ploidy = 4)
log_message("PREP: Full A-matrix (pedigree) built.")

# --- G-matrix ---
if (file.exists("matrix_cache/G_full.rds")) {
  G_full <- readRDS("matrix_cache/G_full.rds")
  log_message("PREP: Loaded cached G-matrix from file.")
} else {
  t_start_g <- proc.time()
  G_full <- Gmatrix(genotypeMatrix, method = "VanRaden", ploidy = 4)
  G_full <- G_full + diag(nrow(G_full)) * 1e-4
  saveRDS(G_full, "matrix_cache/G_full.rds")
  t_end_g <- proc.time()
  log_message("PREP: Full G-matrix (additive) built and cached. Time: ", round((t_end_g - t_start_g)[3], 2), "s.")
}

# --- D-matrix ---
if (file.exists("matrix_cache/D_full.rds")) {
  D_full <- readRDS("matrix_cache/D_full.rds")
  log_message("PREP: Loaded cached D-matrix from file.")
} else {
  t_start_d <- proc.time()
  D_raw <- get_DomRel(genotypeMatrix, ploidy = 4)
  Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw))
  D_full <- (Ic %*% D_raw %*% Ic) + diag(nrow(D_raw)) * 1e-4
  rownames(D_full) <- rownames(genotypeMatrix); colnames(D_full) <- rownames(genotypeMatrix)
  saveRDS(D_full, "matrix_cache/D_full.rds")
  t_end_d <- proc.time()
  log_message("PREP: Full D-matrix (dominance) built and cached. Time: ", round((t_end_d - t_start_d)[3], 2), "s.")
}

# --- H-matrix  ---
if (file.exists("matrix_cache/H_full.rds")) {
  H_full <- readRDS("matrix_cache/H_full.rds")
  log_message("PREP: Loaded cached H-matrix from file.")
} else {
  t_start_h <- proc.time()
  H_full <- sommer::H.mat(A=A_full, G=G_full)
  saveRDS(H_full, "matrix_cache/H_full.rds")
  t_end_h <- proc.time()
  log_message("PREP: Full H-matrix (single-step) built and cached. Time: ", round((t_end_h - t_start_h)[3], 2), "s.")
}
t_end_matrices <- proc.time()
log_message("PREP: Total time for matrix preparations: ", round((t_end_matrices - t_start_matrices)[3], 2), " seconds.")


# =====================================================================================
# Part 6: Standalone Hyperparameter Tuning Function
# =====================================================================================
tune_dl_hyperparameters <- function(genotype_data, phenotype_data, numMarkers) {
  log_message("DL_TUNE: Starting hyperparameter tuning for the current trait...")
  t_start_dl_tune <- proc.time()

  # --- Prepare data for tuning (using an 80/20 split of the input data) ---
  set.seed(123)
  tuning_indices <- createDataPartition(phenotype_data, p = 0.8, list = FALSE)
  tuning_genotype <- genotype_data[tuning_indices, ]
  validation_genotype <- genotype_data[-tuning_indices, ]
  tuning_phenotype <- phenotype_data[tuning_indices]
  validation_phenotype <- phenotype_data[-tuning_indices]

  # --- Scale data based on the tuning set ---
  train_mean_tune <- colMeans(tuning_genotype)
  train_sd_tune <- apply(tuning_genotype, 2, sd)
  train_sd_tune[train_sd_tune == 0] <- 1
  tuning_genotype_scaled <- scale(tuning_genotype, center = train_mean_tune, scale = train_sd_tune)
  validation_genotype_scaled <- scale(validation_genotype, center = train_mean_tune, scale = train_sd_tune)

  # Force R objects into plain numeric matrices/vectors before passing them to Python/Keras.
  tuning_genotype_scaled <- as.matrix(tuning_genotype_scaled)
  validation_genotype_scaled <- as.matrix(validation_genotype_scaled)
  storage.mode(tuning_genotype_scaled) <- "double"
  storage.mode(validation_genotype_scaled) <- "double"
  tuning_phenotype <- as.numeric(tuning_phenotype)
  validation_phenotype <- as.numeric(validation_phenotype)

  # --- Define common training parameters ---

  callbacks_list <- list(
      callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)
  )

  # --- MLP Tuning ---
  best_mlp_params <- list(neurons=64, dropout_rate=0.4, learning_rate=0.005) # Defaults
  tryCatch({
      mlp_param_grid <- expand.grid(neurons=c(64, 128), dropout_rate=c(0.4, 0.6), learning_rate=c(0.005, 0.001))
      best_val_loss <- Inf
      for(j in 1:nrow(mlp_param_grid)) {
          params <- mlp_param_grid[j,]
          model <- keras_model_sequential(input_shape = c(numMarkers)) %>%
            layer_dense(units=as.integer(params$neurons), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=as.numeric(params$dropout_rate)) %>%
            layer_dense(units=as.integer(round(params$neurons/2)), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=as.numeric(params$learning_rate)))
          hist <- model %>% fit(tuning_genotype_scaled, tuning_phenotype, epochs=TRAINING_EPOCHS, batch_size=32, validation_data=list(validation_genotype_scaled, validation_phenotype), verbose=0, callbacks=callbacks_list)
          val_loss <- min(hist$metrics$val_loss, na.rm=TRUE)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
      }
      log_message("DL_TUNE: Best MLP params found: neurons=", best_mlp_params$neurons, ", dropout=", best_mlp_params$dropout_rate, ", lr=", best_mlp_params$learning_rate, indent = 1)
  }, error=function(e){log_message("DL_TUNE: WARNING - MLP tuning failed. Using defaults.", indent = 1)})

  # --- CNN Tuning ---
  best_cnn_params <- list(filters=32, kernel_size=8, learning_rate=0.005) # Defaults
  tryCatch({
      xtrain_cnn_tune <- array(as.numeric(tuning_genotype_scaled), dim=c(nrow(tuning_genotype_scaled), numMarkers, 1))
      xval_cnn_tune <- array(as.numeric(validation_genotype_scaled), dim=c(nrow(validation_genotype_scaled), numMarkers, 1))
      cnn_param_grid <- expand.grid(filters=c(32,64), kernel_size=c(8,12), learning_rate=c(0.005,0.001))
      best_val_loss <- Inf
      for(j in 1:nrow(cnn_param_grid)) {
          params <- cnn_param_grid[j,]
          model <- keras_model_sequential(input_shape = c(numMarkers, 1)) %>%
            layer_conv_1d(filters=as.integer(params$filters), kernel_size=as.integer(params$kernel_size), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
            layer_dense(units=64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=as.numeric(params$learning_rate)))
          hist <- model %>% fit(xtrain_cnn_tune, tuning_phenotype, epochs=TRAINING_EPOCHS, batch_size=32, validation_data=list(xval_cnn_tune, validation_phenotype), verbose=0, callbacks=callbacks_list)
          val_loss <- min(hist$metrics$val_loss, na.rm=TRUE)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_cnn_params <- params }
      }
      log_message("DL_TUNE: Best CNN params found: filters=", best_cnn_params$filters, ", kernel_size=", best_cnn_params$kernel_size, ", lr=", best_cnn_params$learning_rate, indent = 1)
  }, error=function(e){log_message("DL_TUNE: WARNING - CNN tuning failed. Using defaults.", indent = 1)})

  t_end_dl_tune <- proc.time()
  log_message("DL_TUNE: DL hyperparameter tuning complete. Time: ", round((t_end_dl_tune - t_start_dl_tune)[3], 2), "s.")
  
  return(list(mlp = best_mlp_params, cnn = best_cnn_params))
}


# =====================================================================================
# Part 7: Main Analysis Function
# =====================================================================================
perform_cross_population_prediction <- function(train_ids, pred_ids, train_pop_name, pred_pop_name,
                                                phenotypeVector, genotypeMatrix,
                                                A_full, G_full, D_full, H_full,
                                                dl_params) { 

  log_message("----------------------------------------------------------------------")
  log_message("START SCENARIO: Train on '", train_pop_name, "' -> Predict on '", pred_pop_name, "'")
  log_message("----------------------------------------------------------------------")

  # --- Step 1: Prepare data subsets ---
  train_ids <- unique(train_ids)
  pred_ids <- unique(pred_ids)
  overlap_ids <- intersect(train_ids, pred_ids)
  if (length(overlap_ids) > 0) {
    stop("Training and prediction sets overlap. Please check scenario construction.")
  }
  if (length(train_ids) < 5 || length(pred_ids) < 5) {
    stop("Training or prediction set is too small for a stable prediction scenario.")
  }

  geno_train <- genotypeMatrix[train_ids, ]; pheno_train <- phenotypeVector[train_ids]
  geno_pred <- genotypeMatrix[pred_ids, ]; pheno_pred <- phenotypeVector[pred_ids]
  numMarkers <- ncol(geno_train)
  results_list <- list()
  
  # --- Step 2: Prepare data for BGLR and sommer ---
  combined_ids <- c(train_ids, pred_ids)
  pheno_with_NAs <- phenotypeVector; pheno_with_NAs[pred_ids] <- NA
  pheno_combined_bglr <- pheno_with_NAs[combined_ids]

  log_message("EXEC: Training all models and predicting...", indent = 1)
  
  run_model <- function(model_name, expr) {
    t_start_model <- proc.time()
    log_message(paste0("  - Running model: ", model_name, "..."), indent = 2)
    tryCatch({
      eval(expr)
      t_end_model <- proc.time()
      log_message(paste0("  - ", model_name, " finished. Accuracy = ", round(results_list[[model_name]], 4), 
                         ". Time = ", round((t_end_model - t_start_model)[3], 2), "s."), indent = 2)
    }, error = function(e) {
      log_message(paste0("  - ERROR in ", model_name, " model: ", e$message), indent = 2)
    })
  }

  # --- 1. glmnet Family ---
  run_model("Ridge", {
    fit_ridge <- cv.glmnet(geno_train, pheno_train, alpha = 0, family="gaussian")
    pred_ridge <- predict(fit_ridge, newx = geno_pred, s = "lambda.min")[, 1]
    results_list[["Ridge"]] <- cor(pred_ridge, pheno_pred, use="complete.obs")
  })
  run_model("LASSO", {
    fit_lasso <- cv.glmnet(geno_train, pheno_train, alpha = 1, family="gaussian")
    pred_lasso <- predict(fit_lasso, newx = geno_pred, s = "lambda.min")[, 1]
    results_list[["LASSO"]] <- cor(pred_lasso, pheno_pred, use="complete.obs")
  })
  run_model("Elastic Net", {
    fit_en <- cv.glmnet(geno_train, pheno_train, alpha = 0.5, family="gaussian")
    pred_en <- predict(fit_en, newx = geno_pred, s = "lambda.min")[, 1]
    results_list[["Elastic Net"]] <- cor(pred_en, pheno_pred, use="complete.obs")
  })
  
  # --- 2. BGLR Family ---
  models_bglr <- list(
    BRR = list(list(X = genotypeMatrix[combined_ids,], model = "BRR")),
    BayesA = list(list(X = genotypeMatrix[combined_ids,], model = "BayesA")),
    BayesB = list(list(X = genotypeMatrix[combined_ids,], model = "BayesB")),
    BayesC = list(list(X = genotypeMatrix[combined_ids,], model = "BayesC")),
    `Bayes G-BLUP` = list(list(K = G_full[combined_ids, combined_ids], model = "RKHS"))
  )
  for(m_name in names(models_bglr)){
    run_model(m_name, {
      fit_bglr <- BGLR(y = pheno_combined_bglr, ETA = models_bglr[[m_name]], nIter = 10000, burnIn = 2500, verbose = FALSE)
      pred_indices_bglr <- (length(train_ids) + 1):length(combined_ids)
      pred_bglr <- fit_bglr$yHat[pred_indices_bglr]
      results_list[[m_name]] <- cor(pred_bglr, pheno_pred, use = "complete.obs")
    })
  }
    
  # --- 3. sommer Family ---
  run_model("GBLUP", {
    data_sommer <- data.frame(ID=combined_ids, y=pheno_with_NAs[combined_ids])
    data_sommer$ID <- factor(data_sommer$ID, levels = rownames(G_full))
    fit_gblup <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=G_full), rcov=~units, data=data_sommer, naMethodY="include", verbose=F, dateWarning=FALSE)
    pred_table <- predict(fit_gblup, D = "ID")
    pred_gblup <- pred_table$pvals[pred_ids, "predicted.value"]
    results_list[["GBLUP"]] <- cor(pred_gblup, pheno_pred, use="complete.obs")
  })
  
  run_model("AD-GBLUP", {
    data_sommer <- data.frame(ID=combined_ids, y=pheno_with_NAs[combined_ids])
    data_sommer$ID_A <- factor(data_sommer$ID, levels=rownames(G_full))
    data_sommer$ID_D <- factor(data_sommer$ID, levels=rownames(D_full))
    fit_ad <- mmes(fixed=y~1, random=~vsm(ism(ID_A), Gu=G_full) + vsm(ism(ID_D), Gu=D_full), rcov=~units, data=data_sommer, naMethodY="include", verbose=F, dateWarning=FALSE)
 
    intercept <- fit_ad$b[1, 1]
    u_A <- fit_ad$uList[[1]][pred_ids, , drop=FALSE] 
    u_D <- fit_ad$uList[[2]][pred_ids, , drop=FALSE]
    pred_ad <- intercept + u_A + u_D
    results_list[["AD-GBLUP"]] <- cor(pred_ad, pheno_pred, use="complete.obs")
  })
  
  run_model("ssGBLUP", {
    pheno_ssgblup <- rep(NA, nrow(H_full)); names(pheno_ssgblup) <- rownames(H_full)
    pheno_ssgblup[train_ids] <- pheno_train
    data_sommer <- data.frame(ID = names(pheno_ssgblup), y = pheno_ssgblup)
    data_sommer$ID <- factor(data_sommer$ID, levels = rownames(H_full))
    fit_ss <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=H_full), rcov=~units, data=data_sommer, naMethodY="include", verbose=F, dateWarning=FALSE)
    pred_table <- predict(fit_ss, D = "ID")
    pred_ss <- pred_table$pvals[pred_ids, "predicted.value"]
    results_list[["ssGBLUP"]] <- cor(pred_ss, pheno_pred, use="complete.obs")
  })
  
  # --- 4. Deep Learning Family ---
  best_mlp_params <- dl_params$mlp
  best_cnn_params <- dl_params$cnn

  train_mean <- colMeans(geno_train); train_sd <- apply(geno_train, 2, sd); train_sd[train_sd == 0] <- 1
  geno_train_scaled <- scale(geno_train, center = train_mean, scale = train_sd)
  geno_pred_scaled <- scale(geno_pred, center = train_mean, scale = train_sd)

  # Force R objects into plain numeric matrices/vectors before passing them to Python/Keras.
  geno_train_scaled <- as.matrix(geno_train_scaled)
  geno_pred_scaled <- as.matrix(geno_pred_scaled)
  storage.mode(geno_train_scaled) <- "double"
  storage.mode(geno_pred_scaled) <- "double"
  pheno_train <- as.numeric(pheno_train)
  pheno_pred <- as.numeric(pheno_pred)
  
  run_model("MLP", {
    final_model <- keras_model_sequential(input_shape = c(numMarkers)) %>%
      layer_dense(units=as.integer(best_mlp_params$neurons), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=as.numeric(best_mlp_params$dropout_rate)) %>%
      layer_dense(units=as.integer(round(best_mlp_params$neurons/2)), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
    final_model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=as.numeric(best_mlp_params$learning_rate)))
    final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
    final_model %>% fit(geno_train_scaled, pheno_train, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
    pred_mlp <- final_model %>% predict(geno_pred_scaled, verbose=0)
    results_list[["MLP"]] <- cor(pred_mlp[,1], pheno_pred, use="complete.obs")
  })
  
  run_model("CNN", {
    xtrain_cnn <- array(as.numeric(geno_train_scaled), dim=c(nrow(geno_train_scaled), numMarkers, 1))
    xtest_cnn <- array(as.numeric(geno_pred_scaled), dim=c(nrow(geno_pred_scaled), numMarkers, 1))
    final_model <- keras_model_sequential(input_shape = c(numMarkers, 1)) %>%
      layer_conv_1d(filters=as.integer(best_cnn_params$filters), kernel_size=as.integer(best_cnn_params$kernel_size), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
      layer_dense(units=64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
    final_model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=as.numeric(best_cnn_params$learning_rate)))
    final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
    final_model %>% fit(xtrain_cnn, pheno_train, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
    pred_cnn <- final_model %>% predict(xtest_cnn, verbose=0)
    results_list[["CNN"]] <- cor(pred_cnn[,1], pheno_pred, use="complete.obs")
  })

  # --- Step 4: Collate and return results ---
  results_df <- tibble::enframe(unlist(results_list), name = "Model", value = "Accuracy") %>%
    mutate(
      Training_Population = train_pop_name,
      Prediction_Population = pred_pop_name
    ) %>%
    dplyr::select(Training_Population, Prediction_Population, Model, Accuracy)
    
  log_message("----------------------------------------------------------------------")
  log_message("END SCENARIO: Train on '", train_pop_name, "' -> Predict on '", pred_pop_name, "'")
  log_message("----------------------------------------------------------------------")
  return(results_df)
}

# =====================================================================================
# Part 8: Mixed-Subpopulation Training Helper
# =====================================================================================
run_mixed_subpopulation_design <- function(source_ids, target_ids,
                                           source_name, target_name,
                                           target_fraction,
                                           design_name,
                                           phenotypeVector, genotypeMatrix,
                                           A_full, G_full, D_full, H_full,
                                           dl_params,
                                           n_repeats = MIXED_N_REPEATS,
                                           base_seed = MIXED_RANDOM_SEED) {

  log_message("----------------------------------------------------------------------")
  log_message("START MIXED DESIGN: ", design_name,
              " | Train ", source_name, " + ", target_fraction * 100, "% ", target_name,
              " -> Predict remaining ", target_name)
  log_message("----------------------------------------------------------------------")

  target_n <- length(target_ids)
  n_target_train <- max(1, floor(target_n * target_fraction))

  if ((target_n - n_target_train) < 5) {
    stop("Too few target-population individuals remain for prediction. Reduce target_fraction.")
  }

  repeated_results <- list()

  for (rep_i in seq_len(n_repeats)) {
    set.seed(base_seed + rep_i)

    sampled_target_train_ids <- sample(target_ids, size = n_target_train, replace = FALSE)
    mixed_train_ids <- c(source_ids, sampled_target_train_ids)
    mixed_pred_ids <- setdiff(target_ids, sampled_target_train_ids)

    log_message("MIXED REP ", rep_i, "/", n_repeats,
                ": source n = ", length(source_ids),
                ", added target-training n = ", length(sampled_target_train_ids),
                ", prediction n = ", length(mixed_pred_ids), indent = 1)

    res_rep <- perform_cross_population_prediction(
      train_ids = mixed_train_ids,
      pred_ids = mixed_pred_ids,
      train_pop_name = paste0(source_name, "+", round(target_fraction * 100), "%_", target_name),
      pred_pop_name = paste0("Remaining_", target_name),
      phenotypeVector = phenotypeVector,
      genotypeMatrix = genotypeMatrix,
      A_full = A_full,
      G_full = G_full,
      D_full = D_full,
      H_full = H_full,
      dl_params = dl_params
    ) %>%
      dplyr::mutate(
        Scenario_Type = "Mixed-subpopulation training",
        Design = design_name,
        Source_Population = source_name,
        Target_Population = target_name,
        Target_Fraction_Added_To_Training = target_fraction,
        Source_N = length(source_ids),
        Target_Training_N = length(sampled_target_train_ids),
        Prediction_N = length(mixed_pred_ids),
        Replicate = rep_i
      )

    repeated_results[[rep_i]] <- res_rep
  }

  mixed_results <- dplyr::bind_rows(repeated_results)

  log_message("----------------------------------------------------------------------")
  log_message("END MIXED DESIGN: ", design_name)
  log_message("----------------------------------------------------------------------")

  return(mixed_results)
}

# =====================================================================================
# Part 9: Execute Strict Cross-Population and Mixed-Subpopulation Scenarios for Selected Traits
#         GPU-aware PARALLEL VERSION
# =====================================================================================
# Recommended on CPU-only servers: do not set REQUESTED_N_WORKERS to a very large number immediately.
# Each worker loads large matrices and may run TensorFlow/Keras. Start with 16-24,
# then increase after checking RAM and CPU usage.
# Parallel options are configured in the USER CONFIG section at the top of the script.

# Keep low-level math libraries single-threaded inside each worker to prevent oversubscription.
Sys.setenv(
  OPENBLAS_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  MKL_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  OMP_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  TF_NUM_INTRAOP_THREADS = TF_INTRAOP_THREADS_PER_WORKER,
  TF_NUM_INTEROP_THREADS = TF_INTEROP_THREADS_PER_WORKER,
  TF_FORCE_GPU_ALLOW_GROWTH = TF_FORCE_GPU_ALLOW_GROWTH_VALUE,
  TF_CPP_MIN_LOG_LEVEL = TF_CPP_MIN_LOG_LEVEL_VALUE
)

detect_available_gpus <- function() {
  cuda_visible <- Sys.getenv("CUDA_VISIBLE_DEVICES", unset = NA_character_)
  if (!is.na(cuda_visible) && trimws(cuda_visible) %in% c("", "-1")) {
    log_message("PARALLEL: CUDA_VISIBLE_DEVICES is empty/-1, treating GPU as unavailable.")
    return(FALSE)
  }

  # Prefer nvidia-smi for detection. Calling TensorFlow just to detect GPUs can print
  # CUDA/cuInit errors on CPU-only nodes when TensorFlow was compiled with CUDA support.
  nvidia_smi <- Sys.which("nvidia-smi")
  if (nzchar(nvidia_smi)) {
    smi_out <- tryCatch(
      system2(nvidia_smi, args = "-L", stdout = TRUE, stderr = FALSE),
      error = function(e) character(0)
    )
    if (length(smi_out) > 0 && any(grepl("GPU", smi_out, fixed = TRUE))) {
      log_message("PARALLEL: GPU detected by nvidia-smi. For safety, auto mode will disable multi-worker parallelism.")
      return(TRUE)
    }
  }

  # Optional fallback. Disabled by default to avoid noisy CUDA initialization errors.
  if (isTRUE(ENABLE_TF_GPU_DETECT)) {
    tf_gpus <- tryCatch({
      devices <- tensorflow::tf$config$list_physical_devices("GPU")
      length(devices) > 0
    }, error = function(e) FALSE)
    if (isTRUE(tf_gpus)) {
      log_message("PARALLEL: GPU detected by TensorFlow. For safety, auto mode will disable multi-worker parallelism.")
      return(TRUE)
    }
  }

  log_message("PARALLEL: No visible GPU detected. CPU parallelism is allowed in auto mode.")
  return(FALSE)
}

PARALLEL_MODE <- tolower(PARALLEL_MODE)
if (!PARALLEL_MODE %in% c("auto", "serial", "cpu")) {
  stop("PARALLEL_MODE must be one of: auto, serial, cpu")
}

requested_workers <- as.integer(REQUESTED_N_WORKERS)
requested_workers <- max(1, requested_workers)
max_safe_workers <- max(1, parallel::detectCores(logical = TRUE) - 2)
gpu_detected <- detect_available_gpus()

if (PARALLEL_MODE == "serial") {
  N_WORKERS <- 1L
  log_message("PARALLEL: PARALLEL_MODE=serial, forcing N_WORKERS=1.")
} else if (PARALLEL_MODE == "auto" && gpu_detected) {
  N_WORKERS <- 1L
  log_message("PARALLEL: GPU detected and PARALLEL_MODE=auto, forcing N_WORKERS=1 to avoid GPU memory overuse.")
} else {
  N_WORKERS <- min(requested_workers, max_safe_workers)
  if (HIDE_CUDA_FOR_CPU_PARALLEL) {
    Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
    log_message("PARALLEL: CPU-style execution selected; setting CUDA_VISIBLE_DEVICES=-1 inside this R process and workers.")
  }
  if (PARALLEL_MODE == "cpu" && gpu_detected) {
    log_message("PARALLEL: PARALLEL_MODE=cpu, allowing CPU-style parallelism even though a GPU was detected.")
  }
}

log_message("PARALLEL: Final worker count = ", N_WORKERS,
            " | PARALLEL_MODE = ", PARALLEL_MODE,
            " | GPU detected = ", gpu_detected)

run_one_trait_analysis <- function(TRAIT_OF_INTEREST) {

  t_start_trait <- proc.time()
  log_message("\n######################################################################")
  log_message("###   STARTING ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###")
  log_message("######################################################################\n")

  phenotypeVector_current <- pheno_df_aligned[[TRAIT_OF_INTEREST]]
  names(phenotypeVector_current) <- rownames(pheno_df_aligned)

  # Remove individuals with missing phenotype for the current trait.
  valid_trait_ids <- names(phenotypeVector_current)[!is.na(phenotypeVector_current)]
  pop_NH_HN_ids_trait <- intersect(pop_NH_HN_ids, valid_trait_ids)
  pop_HP_ids_trait <- intersect(pop_HP_ids, valid_trait_ids)

  log_message("TRAIT: Valid NH_HN individuals = ", length(pop_NH_HN_ids_trait),
              "; valid HP individuals = ", length(pop_HP_ids_trait), indent = 1)

  if (length(pop_NH_HN_ids_trait) < 10 || length(pop_HP_ids_trait) < 10) {
    log_message("WARNING: Skipping trait '", TRAIT_OF_INTEREST,
                "' because one population has fewer than 10 phenotyped individuals.", indent = 1)
    return(NULL)
  }

  # --- Perform one-time DL hyperparameter tuning for the current trait ---
  tuning_pop_ids <- if (length(pop_NH_HN_ids_trait) > length(pop_HP_ids_trait)) pop_NH_HN_ids_trait else pop_HP_ids_trait
  best_dl_params <- tune_dl_hyperparameters(
    genotype_data = genotypeMatrix[tuning_pop_ids, ],
    phenotype_data = phenotypeVector_current[tuning_pop_ids],
    numMarkers = numMarkers
  )

  # -------------------------------------------------------------------
  # A. Original strict reciprocal cross-subpopulation prediction
  # -------------------------------------------------------------------
  result_strict_A <- perform_cross_population_prediction(
    train_ids = pop_NH_HN_ids_trait,
    pred_ids = pop_HP_ids_trait,
    train_pop_name = "NH_HN",
    pred_pop_name = "HP",
    phenotypeVector = phenotypeVector_current,
    genotypeMatrix = genotypeMatrix,
    A_full = A_full,
    G_full = G_full,
    D_full = D_full,
    H_full = H_full,
    dl_params = best_dl_params
  ) %>%
    dplyr::mutate(
      Scenario_Type = "Strict cross-subpopulation prediction",
      Design = "Strict_1_NH_HN_to_HP",
      Source_Population = "NH_HN",
      Target_Population = "HP",
      Target_Fraction_Added_To_Training = 0,
      Source_N = length(pop_NH_HN_ids_trait),
      Target_Training_N = 0,
      Prediction_N = length(pop_HP_ids_trait),
      Replicate = 1
    )

  result_strict_B <- perform_cross_population_prediction(
    train_ids = pop_HP_ids_trait,
    pred_ids = pop_NH_HN_ids_trait,
    train_pop_name = "HP",
    pred_pop_name = "NH_HN",
    phenotypeVector = phenotypeVector_current,
    genotypeMatrix = genotypeMatrix,
    A_full = A_full,
    G_full = G_full,
    D_full = D_full,
    H_full = H_full,
    dl_params = best_dl_params
  ) %>%
    dplyr::mutate(
      Scenario_Type = "Strict cross-subpopulation prediction",
      Design = "Strict_2_HP_to_NH_HN",
      Source_Population = "HP",
      Target_Population = "NH_HN",
      Target_Fraction_Added_To_Training = 0,
      Source_N = length(pop_HP_ids_trait),
      Target_Training_N = 0,
      Prediction_N = length(pop_NH_HN_ids_trait),
      Replicate = 1
    )

  # -------------------------------------------------------------------
  # B. Mixed-subpopulation training scenarios
  # -------------------------------------------------------------------
  # Mixed 1: NH_HN + 25% HP -> remaining HP
  result_mixed_1 <- run_mixed_subpopulation_design(
    source_ids = pop_NH_HN_ids_trait,
    target_ids = pop_HP_ids_trait,
    source_name = "NH_HN",
    target_name = "HP",
    target_fraction = MIXED_TARGET_PROPORTIONS[1],
    design_name = "Mixed_1_NH_HN_plus_25pct_HP_to_remaining_HP",
    phenotypeVector = phenotypeVector_current,
    genotypeMatrix = genotypeMatrix,
    A_full = A_full,
    G_full = G_full,
    D_full = D_full,
    H_full = H_full,
    dl_params = best_dl_params,
    n_repeats = MIXED_N_REPEATS,
    base_seed = MIXED_RANDOM_SEED
  )

  # Mixed 2: NH_HN + 50% HP -> remaining HP
  result_mixed_2 <- run_mixed_subpopulation_design(
    source_ids = pop_NH_HN_ids_trait,
    target_ids = pop_HP_ids_trait,
    source_name = "NH_HN",
    target_name = "HP",
    target_fraction = MIXED_TARGET_PROPORTIONS[2],
    design_name = "Mixed_2_NH_HN_plus_50pct_HP_to_remaining_HP",
    phenotypeVector = phenotypeVector_current,
    genotypeMatrix = genotypeMatrix,
    A_full = A_full,
    G_full = G_full,
    D_full = D_full,
    H_full = H_full,
    dl_params = best_dl_params,
    n_repeats = MIXED_N_REPEATS,
    base_seed = MIXED_RANDOM_SEED + 1000
  )

  # Mixed 3: HP + 25% NH_HN -> remaining NH_HN
  result_mixed_3 <- run_mixed_subpopulation_design(
    source_ids = pop_HP_ids_trait,
    target_ids = pop_NH_HN_ids_trait,
    source_name = "HP",
    target_name = "NH_HN",
    target_fraction = MIXED_TARGET_PROPORTIONS[1],
    design_name = "Mixed_3_HP_plus_25pct_NH_HN_to_remaining_NH_HN",
    phenotypeVector = phenotypeVector_current,
    genotypeMatrix = genotypeMatrix,
    A_full = A_full,
    G_full = G_full,
    D_full = D_full,
    H_full = H_full,
    dl_params = best_dl_params,
    n_repeats = MIXED_N_REPEATS,
    base_seed = MIXED_RANDOM_SEED + 2000
  )

  # Mixed 4: HP + 50% NH_HN -> remaining NH_HN
  result_mixed_4 <- run_mixed_subpopulation_design(
    source_ids = pop_HP_ids_trait,
    target_ids = pop_NH_HN_ids_trait,
    source_name = "HP",
    target_name = "NH_HN",
    target_fraction = MIXED_TARGET_PROPORTIONS[2],
    design_name = "Mixed_4_HP_plus_50pct_NH_HN_to_remaining_NH_HN",
    phenotypeVector = phenotypeVector_current,
    genotypeMatrix = genotypeMatrix,
    A_full = A_full,
    G_full = G_full,
    D_full = D_full,
    H_full = H_full,
    dl_params = best_dl_params,
    n_repeats = MIXED_N_REPEATS,
    base_seed = MIXED_RANDOM_SEED + 3000
  )

  # --- Combine and save results for the CURRENT trait ---
  trait_results <- dplyr::bind_rows(
    result_strict_A,
    result_strict_B,
    result_mixed_1,
    result_mixed_2,
    result_mixed_3,
    result_mixed_4
  ) %>%
    dplyr::mutate(Trait = TRAIT_OF_INTEREST) %>%
    dplyr::select(
      Trait, Scenario_Type, Design, Replicate,
      Source_Population, Target_Population,
      Target_Fraction_Added_To_Training,
      Source_N, Target_Training_N, Prediction_N,
      Training_Population, Prediction_Population,
      Model, Accuracy
    )

  log_message("RESULTS: Completed detailed results for trait '", TRAIT_OF_INTEREST, "'.")
  trait_output_file <- file.path(
      "results",
      paste0("Trait_Result_", TRAIT_OF_INTEREST, ".csv")
    )

    write.csv(trait_results, trait_output_file, row.names = FALSE)

  t_end_trait <- proc.time()
  log_message("###   FINISHED ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###")
  log_message("###   Time elapsed for this trait: ", round((t_end_trait - t_start_trait)[3], 2), " seconds.   ###\n")
  return(trait_results)

}

incremental_results_file <- "results/GS_Cross_and_Mixed_Population_Incremental_Results.csv"
if (file.exists(incremental_results_file)) file.remove(incremental_results_file)

if (N_WORKERS <= 1) {
  log_message("PARALLEL: N_WORKERS <= 1, running traits sequentially.")
  all_results_list <- lapply(TRAITS_TO_RUN, run_one_trait_analysis)
  names(all_results_list) <- TRAITS_TO_RUN
} else {
  log_message("PARALLEL: Starting PSOCK cluster with ", N_WORKERS, " workers.")

  cl <- parallel::makeCluster(N_WORKERS, type = "PSOCK", outfile = "results/parallel_workers.log")
  on.exit({
  try(parallel::stopCluster(cl), silent = TRUE)
}, add = TRUE)

  # Export Python path before workers load TensorFlow/Keras, so reticulate is bound correctly.
  parallel::clusterExport(
    cl,
    varlist = c(
      "python_exe_path",
      "BLAS_THREADS_PER_WORKER", "TF_INTRAOP_THREADS_PER_WORKER",
      "TF_INTEROP_THREADS_PER_WORKER", "TF_FORCE_GPU_ALLOW_GROWTH_VALUE",
      "TF_CPP_MIN_LOG_LEVEL_VALUE", "SUPPRESS_KERAS_WARNINGS",
      "HIDE_CUDA_FOR_CPU_PARALLEL"
    ),
    envir = environment()
  )

  # Worker initialization: set thread/CUDA environment BEFORE loading TensorFlow,
  # then bind reticulate to the same Python executable. Do not call
  # tf$config$threading setters after TensorFlow has initialized; that causes
  # "Intra op parallelism cannot be modified after initialization".
  parallel::clusterEvalQ(cl, {
    Sys.setenv(
      OPENBLAS_NUM_THREADS = BLAS_THREADS_PER_WORKER,
      MKL_NUM_THREADS = BLAS_THREADS_PER_WORKER,
      OMP_NUM_THREADS = BLAS_THREADS_PER_WORKER,
      TF_NUM_INTRAOP_THREADS = TF_INTRAOP_THREADS_PER_WORKER,
      TF_NUM_INTEROP_THREADS = TF_INTEROP_THREADS_PER_WORKER,
      TF_FORCE_GPU_ALLOW_GROWTH = TF_FORCE_GPU_ALLOW_GROWTH_VALUE,
      TF_CPP_MIN_LOG_LEVEL = TF_CPP_MIN_LOG_LEVEL_VALUE
    )
    if (isTRUE(HIDE_CUDA_FOR_CPU_PARALLEL)) {
      Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
    }
    if (isTRUE(SUPPRESS_KERAS_WARNINGS)) options(warn = -1)
    suppressPackageStartupMessages(library(reticulate))
    reticulate::use_python(python_exe_path, required = TRUE)
    suppressPackageStartupMessages({
      lapply(c(
        "tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
        "tensorflow", "keras3", "tfdatasets", "caret", "cowplot",
        "sommer", "Cairo", "tidyr"
      ), library, character.only = TRUE)
    })
    NULL
  })

  parallel::clusterExport(
    cl,
    varlist = c(
      "python_exe_path", "log_message", "theme_publication",
      "TRAINING_EPOCHS", "MIXED_TARGET_PROPORTIONS", "MIXED_N_REPEATS", "MIXED_RANDOM_SEED",
      "ALL_TRAITS_IN_FILE", "TRAITS_TO_RUN", "pheno_df_aligned", "genotypeMatrix", "numMarkers",
      "pop_NH_HN_ids", "pop_HP_ids", "A_full", "G_full", "D_full", "H_full",
      "tune_dl_hyperparameters", "perform_cross_population_prediction",
      "run_mixed_subpopulation_design", "run_one_trait_analysis"
    ),
    envir = environment()
  )

  all_results_list <- parallel::parLapplyLB(cl, TRAITS_TO_RUN, function(trait) {
    set.seed(MIXED_RANDOM_SEED + match(trait, TRAITS_TO_RUN) * 100000L)
    run_one_trait_analysis(trait)
  })
  names(all_results_list) <- TRAITS_TO_RUN

  try(parallel::stopCluster(cl), silent = TRUE)
}

all_results_list <- all_results_list[!vapply(all_results_list, is.null, logical(1))]

# Write incremental results once in the master process to avoid concurrent file writes.
if (length(all_results_list) > 0) {
  readr::write_csv(dplyr::bind_rows(all_results_list), incremental_results_file)
  log_message("RESULTS: Incremental combined results written to '", incremental_results_file, "'.")
} else {
  stop("No trait produced results. Check phenotype missingness and population sizes.")
}

# =====================================================================================
# Part 10: Final Summary and Cleanup
# =====================================================================================
final_summary_all_traits <- dplyr::bind_rows(all_results_list)

log_message("\n====================================================================")
log_message("     FINAL CROSS- AND MIXED-POPULATION PREDICTION SUMMARY")
log_message("====================================================================\n")

# Detailed long-format output: every replicate and every model.
detailed_output_filename <- "results/GS_Cross_and_Mixed_Population_Detailed_All_Traits.csv"
write.csv(final_summary_all_traits, detailed_output_filename, row.names = FALSE)

# Mean/SD/SE summary across mixed-training replicates.
# Strict scenarios have only one replicate, so SD/SE are NA.
final_summary_stats <- final_summary_all_traits %>%
  dplyr::group_by(
    Trait, Scenario_Type, Design,
    Source_Population, Target_Population,
    Target_Fraction_Added_To_Training,
    Model
  ) %>%
  dplyr::summarise(
    n_replicates = dplyr::n(),
    mean_accuracy = mean(Accuracy, na.rm = TRUE),
    sd_accuracy = ifelse(dplyr::n() > 1, sd(Accuracy, na.rm = TRUE), NA_real_),
    se_accuracy = ifelse(dplyr::n() > 1, sd(Accuracy, na.rm = TRUE) / sqrt(dplyr::n()), NA_real_),
    .groups = "drop"
  )

print(as.data.frame(final_summary_stats))

summary_output_filename <- "results/GS_Cross_and_Mixed_Population_Summary_Stats.csv"
write.csv(final_summary_stats, summary_output_filename, row.names = FALSE)

# Wide-format summary table for easy manuscript plotting.
final_summary_wide <- final_summary_stats %>%
  dplyr::select(Trait, Scenario_Type, Design, Source_Population, Target_Population,
                Target_Fraction_Added_To_Training, Model, mean_accuracy, se_accuracy) %>%
  tidyr::pivot_wider(
    names_from = Model,
    values_from = c(mean_accuracy, se_accuracy)
  )

wide_output_filename <- "results/GS_Cross_and_Mixed_Population_Summary_Wide.csv"
write.csv(final_summary_wide, wide_output_filename, row.names = FALSE)

log_message(paste0("\n\n[--- FINISHED ---] Analysis complete."))
log_message(paste0("Detailed replicate-level results saved to '", detailed_output_filename, "'"))
log_message(paste0("Mean/SD/SE summary saved to '", summary_output_filename, "'"))
log_message(paste0("Wide summary saved to '", wide_output_filename, "'"))
log_message(paste0("Incremental results for all completed traits are available in '", incremental_results_file, "'"))

# =====================================================================================
# Part 0: Environment Setup
# =====================================================================================
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)
Sys.setenv(OMP_NUM_THREADS = 1)

# --- Enhanced Logging Function ---
log_message <- function(..., indent = 0) {
  prefix <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ")
  indent_space <- paste(rep(" ", indent * 2), collapse = "")
  message(paste0(prefix, indent_space, ...))
}

log_message("SETUP: Loading required R packages...")
required_packages <- c(
  "tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
  "tensorflow", "keras3", "tfdatasets", "caret", "cowplot", "reticulate",
  "sommer", "Cairo", "tidyr" 
)
suppressPackageStartupMessages({ lapply(required_packages, library, character.only = TRUE) })
log_message("SETUP: Packages loaded.")

# =====================================================================================
# Part 1: Find Python Path Before Parallelization (CRITICAL STEP)
# =====================================================================================
log_message("SETUP: Finding Python executable from the 'poly_map' conda environment...")
tryCatch({
  use_condaenv("poly_map", required = TRUE)
  python_exe_path <- reticulate::py_config()$python
  log_message("SETUP: Found python executable to be used by workers: ", python_exe_path)
}, error = function(e) {
  log_message("SETUP: ERROR - Could not find the 'poly_map' conda environment. Please ensure it is correctly installed and configured.")
  stop(e)
})


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

# --- G-matrix (cached) ---
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

# --- D-matrix (cached) ---
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

# --- H-matrix (cached) ---
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
          model <- keras_model_sequential() %>%
            layer_dense(units=params$neurons, input_shape=numMarkers, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=params$dropout_rate) %>%
            layer_dense(units=round(params$neurons/2), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=params$learning_rate))
          hist <- model %>% fit(tuning_genotype_scaled, tuning_phenotype, epochs=TRAINING_EPOCHS, batch_size=32, validation_data=list(validation_genotype_scaled, validation_phenotype), verbose=0, callbacks=callbacks_list)
          val_loss <- min(hist$metrics$val_loss, na.rm=TRUE)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
      }
      log_message("DL_TUNE: Best MLP params found: neurons=", best_mlp_params$neurons, ", dropout=", best_mlp_params$dropout_rate, ", lr=", best_mlp_params$learning_rate, indent = 1)
  }, error=function(e){log_message("DL_TUNE: WARNING - MLP tuning failed. Using defaults.", indent = 1)})

  # --- CNN Tuning ---
  best_cnn_params <- list(filters=32, kernel_size=8, learning_rate=0.005) # Defaults
  tryCatch({
      xtrain_cnn_tune <- array(tuning_genotype_scaled, dim=c(nrow(tuning_genotype_scaled), numMarkers, 1))
      xval_cnn_tune <- array(validation_genotype_scaled, dim=c(nrow(validation_genotype_scaled), numMarkers, 1))
      cnn_param_grid <- expand.grid(filters=c(32,64), kernel_size=c(8,12), learning_rate=c(0.005,0.001))
      best_val_loss <- Inf
      for(j in 1:nrow(cnn_param_grid)) {
          params <- cnn_param_grid[j,]
          model <- keras_model_sequential() %>%
            layer_conv_1d(filters=params$filters, kernel_size=params$kernel_size, input_shape=c(numMarkers,1), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
            layer_dense(units=64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=params$learning_rate))
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
    fit_gblup <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=G_full), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
    pred_table <- predict(fit_gblup, D = "ID")
    pred_gblup <- pred_table$pvals[pred_ids, "predicted.value"]
    results_list[["GBLUP"]] <- cor(pred_gblup, pheno_pred, use="complete.obs")
  })
  
  run_model("AD-GBLUP", {
    data_sommer <- data.frame(ID=combined_ids, y=pheno_with_NAs[combined_ids])
    data_sommer$ID_A <- factor(data_sommer$ID, levels=rownames(G_full))
    data_sommer$ID_D <- factor(data_sommer$ID, levels=rownames(D_full))
    fit_ad <- mmes(fixed=y~1, random=~vsm(ism(ID_A), Gu=G_full) + vsm(ism(ID_D), Gu=D_full), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
 
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
    fit_ss <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=H_full), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
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
  
  run_model("MLP", {
    final_model <- keras_model_sequential() %>%
      layer_dense(units=best_mlp_params$neurons, input_shape=numMarkers, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=best_mlp_params$dropout_rate) %>%
      layer_dense(units=round(best_mlp_params$neurons/2), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
    final_model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=best_mlp_params$learning_rate))
    final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
    final_model %>% fit(geno_train_scaled, pheno_train, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
    pred_mlp <- final_model %>% predict(geno_pred_scaled, verbose=0)
    results_list[["MLP"]] <- cor(pred_mlp[,1], pheno_pred, use="complete.obs")
  })
  
  run_model("CNN", {
    xtrain_cnn <- array(geno_train_scaled, dim=c(nrow(geno_train_scaled), numMarkers, 1))
    xtest_cnn <- array(geno_pred_scaled, dim=c(nrow(geno_pred_scaled), numMarkers, 1))
    final_model <- keras_model_sequential() %>%
      layer_conv_1d(filters=best_cnn_params$filters, kernel_size=best_cnn_params$kernel_size, input_shape=c(numMarkers,1), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
      layer_dense(units=64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
      layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
    final_model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=best_cnn_params$learning_rate))
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
# Part 8: Execute Scenarios for All Traits
# =====================================================================================
all_results_list <- list()
incremental_results_file <- "results/GS_Cross_Population_Incremental_Results.csv"
if (file.exists(incremental_results_file)) file.remove(incremental_results_file)

for (TRAIT_OF_INTEREST in ALL_TRAITS_IN_FILE) {
  t_start_trait <- proc.time()
  log_message("\n######################################################################")
  log_message("###   STARTING ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###")
  log_message("######################################################################\n")
  
  phenotypeVector_current <- pheno_df_aligned[[TRAIT_OF_INTEREST]]
  names(phenotypeVector_current) <- rownames(pheno_df_aligned)
  
  # --- Perform one-time DL hyperparameter tuning for the current trait ---
  tuning_pop_ids <- if (length(pop_NH_HN_ids) > length(pop_HP_ids)) pop_NH_HN_ids else pop_HP_ids
  best_dl_params <- tune_dl_hyperparameters(
    genotype_data = genotypeMatrix[tuning_pop_ids, ],
    phenotype_data = phenotypeVector_current[tuning_pop_ids],
    numMarkers = numMarkers
  )
  
  # --- Run scenarios using the tuned DL parameters ---
  result_A <- perform_cross_population_prediction(
    train_ids = pop_NH_HN_ids, pred_ids = pop_HP_ids, train_pop_name = "NH_HN", pred_pop_name = "HP",
    phenotypeVector = phenotypeVector_current, genotypeMatrix = genotypeMatrix,
    A_full=A_full, G_full=G_full, D_full=D_full, H_full=H_full,
    dl_params = best_dl_params # Pass the tuned parameters
  )
  
  result_B <- perform_cross_population_prediction(
    train_ids = pop_HP_ids, pred_ids = pop_NH_HN_ids, train_pop_name = "HP", pred_pop_name = "NH_HN",
    phenotypeVector = phenotypeVector_current, genotypeMatrix = genotypeMatrix,
    A_full=A_full, G_full=G_full, D_full=D_full, H_full=H_full,
    dl_params = best_dl_params # Pass the same tuned parameters
  )
  
  # --- Combine and save results for the CURRENT trait ---
  trait_results <- bind_rows(result_A, result_B)
  trait_results$Trait <- TRAIT_OF_INTEREST
  
  write.table(
    trait_results,
    file = incremental_results_file,
    append = TRUE,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(incremental_results_file)
  )
  log_message("RESULTS: Successfully saved results for trait '", TRAIT_OF_INTEREST, "' to '", incremental_results_file, "'.")
  
  all_results_list[[TRAIT_OF_INTEREST]] <- trait_results
  
  t_end_trait <- proc.time()
  log_message("###   FINISHED ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###")
  log_message("###   Time elapsed for this trait: ", round((t_end_trait - t_start_trait)[3], 2), " seconds.   ###\n")
}

# =====================================================================================
# Part 9: Final Summary and Cleanup 
# =====================================================================================
final_summary_all_traits <- bind_rows(all_results_list)

log_message("\n====================================================================")
log_message("            FINAL CROSS-POPULATION PREDICTION SUMMARY (ALL TRAITS)")
log_message("====================================================================\n")

final_summary_wide <- final_summary_all_traits %>%
  dplyr::select(Trait, everything()) %>%
  tidyr::pivot_wider(names_from = Model, values_from = Accuracy)

print(as.data.frame(final_summary_wide))

output_filename <- "results/GS_Cross_Population_Summary_All_Traits_Wide.csv"
write.csv(final_summary_wide, output_filename, row.names = FALSE)
log_message(paste0("\n\n[--- FINISHED ---] Analysis complete. Final summary saved to '", output_filename, "'"))
log_message(paste0("Incremental results for all completed traits are available in '", incremental_results_file, "'"))
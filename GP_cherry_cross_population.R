# =====================================================================================
#
#   Cross-Population Genomic Prediction Script (Enhanced & Robust)
#
# Description:
#   This script evaluates cross-population genomic prediction accuracy by merging a
#   robust validation framework with advanced model optimization techniques.
#   1. It performs a PCA to visualize population structure.
#   2. It identifies the best model via parallelized, repeated cross-validation
#      within a designated training population, using REML for variance components
#      and hyperparameter tuning for deep learning models.
#   3. It trains the identified best model on the entire training population.
#   4. It assesses prediction accuracy on the separate prediction population using
#      a repeated sampling strategy.
#
# Author: [Your Name]
# Last Modified: Aug 03, 2025
#
# =====================================================================================

# =====================================================================================
# Part 0: Environment Setup
# =====================================================================================
# Set environment variables to prevent over-threading with underlying BLAS/MKL libraries
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)
Sys.setenv(OMP_NUM_THREADS = 1)

cat("Loading required R packages...\n")
# If packages are not installed, uncomment and run the line below once.
# install.packages(c("tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix",
# "ggpubr", "tensorflow", "keras", "tfdatasets", "caret", "cowplot", "reticulate", "Cairo", "future", "furrr", "sommer"))

required_packages <- c(
  "tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
  "tensorflow", "keras", "tfdatasets", "caret", "cowplot", "reticulate",
  "future", "furrr", "sommer", "Cairo"
)
lapply(required_packages, library, character.only = TRUE)

# =====================================================================================
# Part 0.1: Find Python Path Before Parallelization (CRITICAL STEP)
# =====================================================================================
# Configure reticulate in the main session to find the correct python path.
# This path will be explicitly passed to each parallel worker.
cat("Finding Python executable from the 'reseq' conda environment...\n")
use_condaenv("reseq", required = TRUE)

# Get the exact path to the python binary. This is the most important variable for parallel DL.
python_exe_path <- reticulate::py_config()$python
cat(paste("Found python executable to be used by workers:", python_exe_path, "\n"))


# --- Define Global Parameters and Plotting Theme ---
theme_publication <- function(base_size = 12, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      axis.title = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9, color = "black"),
      legend.title = element_text(size = 9, face = "bold"),
      legend.text = element_text(size = 8),
      plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
    )
}

# Create a dedicated directory for outputs
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)


# =====================================================================================
# Part 0.5: Load, Clean, and Preprocess Data
# =====================================================================================
cat("\n[DATA] Loading and preprocessing data...\n")

# --- User Configuration ---
GENOTYPE_FILE     <- "genotype.dosages.tsv"
PHENOTYPE_FILE    <- "phenotype_BLUPs.csv"
TRAIT_OF_INTEREST <- "fruit_weight" # <-- !!! EDIT THIS !!!

# --- Load Genotype Data ---
dosage_raw <- read.csv(GENOTYPE_FILE, check.names = FALSE, sep = "\t", header = TRUE)
marker_ids <- paste0(dosage_raw$CHROM, ":", dosage_raw$POS)
dosage_only_matrix <- as.matrix(dosage_raw[, 5:ncol(dosage_raw)])
genotypeMatrix_raw <- t(dosage_only_matrix)
colnames(genotypeMatrix_raw) <- marker_ids
storage.mode(genotypeMatrix_raw) <- "numeric"
cat(paste0("[DATA] Loaded genotype data with ", nrow(genotypeMatrix_raw), " individuals and ", ncol(genotypeMatrix_raw), " markers.\n"))

# --- Load Phenotype Data ---
pheno_raw <- read.csv(PHENOTYPE_FILE, row.names = 1, check.names = FALSE, sep = ",", header = TRUE)
pheno_df <- as.data.frame(pheno_raw)
cat(paste0("[DATA] Loaded phenotype data for ", nrow(pheno_df), " individuals.\n"))

# --- Data Alignment and Cleaning ---
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
cat(paste0("[DATA] Found ", length(common_individuals), " individuals with both genotype and phenotype data.\n"))

genotypeMatrix <- genotypeMatrix_raw[common_individuals, ]

if (!TRAIT_OF_INTEREST %in% colnames(pheno_df)) {
  stop(paste("Error: The specified trait '", TRAIT_OF_INTEREST, "' was not found in the phenotype file."))
}
phenotype_full_df <- pheno_df[common_individuals, , drop = FALSE] %>%
  dplyr::select(all_of(TRAIT_OF_INTEREST))

phenotypeVector <- phenotype_full_df[[TRAIT_OF_INTEREST]]
names(phenotypeVector) <- rownames(phenotype_full_df)

# --- Handle Missing Genotype Values (Mean Imputation) ---
cat("[DATA] Handling missing genotype values using mean imputation...\n")
impute_mean <- function(x) {
  mean_val <- mean(x, na.rm = TRUE)
  if (is.nan(mean_val)) mean_val <- 0
  x[is.na(x)] <- round(mean_val)
  return(x)
}
genotypeMatrix <- apply(genotypeMatrix, 2, impute_mean)
if(sum(is.na(genotypeMatrix)) > 0) {
  warning("Missing values still exist after imputation. This can happen if a marker is missing for all individuals.")
}

numIndividuals <- nrow(genotypeMatrix)
numMarkers <- ncol(genotypeMatrix)
cat(paste0("[DATA] Data preparation complete: ", numIndividuals, " individuals, ", numMarkers, " SNP markers.\n"))


# =====================================================================================
# Part 0.6: Split Data into Defined Populations
# =====================================================================================
cat("\n[DATA] Splitting data into specified populations...\n")
all_ids <- rownames(genotypeMatrix)
pop_NH_HN_ids <- all_ids[startsWith(all_ids, "NH") | startsWith(all_ids, "HN")]
pop_HP_ids <- all_ids[startsWith(all_ids, "HP")]

cat(paste0("  - Population 1 (NH/HN): ", length(pop_NH_HN_ids), " individuals.\n"))
cat(paste0("  - Population 2 (HP): ", length(pop_HP_ids), " individuals.\n"))

if(length(pop_NH_HN_ids) == 0 | length(pop_HP_ids) == 0) {
  stop("One or both populations have zero individuals. Check sample ID prefixes.")
}

# =====================================================================================
# Part 0.7: Population Structure Analysis via PCA
# =====================================================================================
cat("\n[PCA] Performing Principal Component Analysis...\n")
# Filter out markers with zero variance before PCA
non_zero_var_cols <- which(apply(genotypeMatrix, 2, var, na.rm = TRUE) > 1e-6)
pca_results <- prcomp(genotypeMatrix[, non_zero_var_cols], center = TRUE, scale. = TRUE)

pca_df <- as.data.frame(pca_results$x) %>%
  dplyr::mutate(ID = rownames(.),
                Population = case_when(
                  ID %in% pop_NH_HN_ids ~ "NH_HN",
                  ID %in% pop_HP_ids    ~ "HP",
                  TRUE                  ~ "Other"
                ))

pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Population)) +
  geom_point(alpha = 0.8, size = 3) +
  theme_bw() +
  labs(
    title = "Population Structure (PCA)",
    x = paste0("PC1 (", round(summary(pca_results)$importance[2, 1] * 100, 2), "%)"),
    y = paste0("PC2 (", round(summary(pca_results)$importance[2, 2] * 100, 2), "%)")
  ) +
  stat_ellipse(aes(group = Population), type = "t") +
  scale_color_manual(values = c("NH_HN" = "#0072B2", "HP" = "#D55E00", "Other" = "grey50")) +
  theme_publication()

pca_filename <- paste0("plots/PCA_Plot_", TRAIT_OF_INTEREST, ".pdf")
ggsave(pca_filename, pca_plot, width = 8, height = 6, device = cairo_pdf)
cat(paste0("[PCA] PCA plot saved as '", pca_filename, "'.\n"))


# =====================================================================================
# Part 1: Helper Functions and Pedigree/Matrix Preparation
# =====================================================================================
cat("\n[PREP] Building pedigree and relationship matrices for the entire population...\n")
# This script is required for the dominance matrix calculation.
source("get_DomRel_matrix.R")

# --- Build Pedigree Data Frame ---
parents <- c("HF1", "NZH2", "PJHH")
all_ped_ids <- unique(c(parents, all_ids))
ped_df <- data.frame(ID = all_ped_ids, Sire = 0, Dam = 0, stringsAsFactors = FALSE)
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) {
    ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) {
    ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "PJHH"
  }
}

# --- A-matrix (Pedigree Relationship) ---
A_full <- Amatrix(ped_df, ploidy = 4)
cat("[PREP] A-matrix built successfully.\n")

# --- G-matrix (Additive Genomic Relationship) ---
G_full <- Gmatrix(genotypeMatrix, method = "VanRaden", ploidy = 4)
G_full <- G_full + diag(nrow(G_full)) * 1e-4 # Regularization for stability
cat("[PREP] G-matrix (Additive) built successfully.\n")

# --- D-matrix (Dominance Genomic Relationship) ---
D_raw <- get_DomRel(genotypeMatrix, ploidy = 4)
Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw)) # Centering matrix
D_centered <- Ic %*% D_raw %*% Ic
D_full <- D_centered + diag(nrow(D_centered)) * 1e-4 # Regularization
rownames(D_full) <- rownames(genotypeMatrix); colnames(D_full) <- rownames(genotypeMatrix)
cat("[PREP] D-matrix (Dominance) built successfully.\n")


# --- H-inverse Matrix Helper Function (for ssGBLUP) ---
# This helper function calculates the H-inverse matrix from A and G matrices.
doH_inverse <- function(A_matrix, G_matrix) {
  genotyped_ids <- rownames(G_matrix)
  A_inv <- solve(A_matrix)
  G_inv <- solve(G_matrix)
  genotyped_indices_in_A <- match(genotyped_ids, rownames(A_matrix))
  
  # Handle case where no genotyped individuals are in the A matrix (unlikely but safe)
  if (length(genotyped_indices_in_A) == 0 || any(is.na(genotyped_indices_in_A))) {
    return(A_inv)
  }
  
  A22_inv <- solve(A_matrix[genotyped_indices_in_A, genotyped_indices_in_A])
  
  H_inv <- A_inv
  # This line implements the core formula: H_inv = A_inv + [0, 0; 0, G_inv - A22_inv]
  H_inv[genotyped_indices_in_A, genotyped_indices_in_A] <-
    H_inv[genotyped_indices_in_A, genotyped_indices_in_A] + G_inv - A22_inv
  
  # CRITICAL: Tell sommer this is an inverse matrix to prevent it from inverting it again.
  attr(H_inv, 'inverse') <- TRUE
  return(H_inv)
}

# Pre-calculate the H-inverse for the entire population.
# This will be subsetted or used as-is in the final prediction step.
Hinv_full <- doH_inverse(A_full, G_full)
cat("[PREP] H-inverse matrix built successfully for the full population.\n")

# =====================================================================================
# Part 2: Robust Parallel Backend Setup (NEW IMPLEMENTATION)
# =====================================================================================
# --- User Configuration for Analysis ---
CV_REPETITIONS    <- 20
NUM_CORES_TO_USE  <- 20 # <-- !!! SET THE NUMBER OF CPU CORES TO USE !!!
PRED_SAMPLES      <- 50
PRED_SAMPLE_FRAC  <- 0.8

cat("\n[PARALLEL] Setting up robust parallel cluster with direct path injection...\n")

# 1. Create a persistent cluster of independent R sessions.
cl <- parallel::makeCluster(NUM_CORES_TO_USE)

# 2. Export the exact python executable path to each worker.
#    This avoids any search/discovery issues inside the workers.
parallel::clusterExport(cl, "python_exe_path")

# 3. Pre-configure each worker to use the injected Python path and load Keras.
#    This is the most reliable way to prepare workers for deep learning tasks.
parallel::clusterEvalQ(cl, {
  library(reticulate)
  # Use the direct path to python, which is the most robust method.
  use_python(python_exe_path, required = TRUE)
  
  # Pre-load keras to finalize the Python-R connection on each worker.
  library(keras)
})

# 4. Tell the 'future' framework to use our pre-configured cluster for all parallel tasks.
plan(cluster, workers = cl)

cat("[PARALLEL] Cluster is ready. Starting computations.\n")


# =====================================================================================
# Part 3: Main Analysis Function for a Single Cross-Population Scenario
# =====================================================================================

perform_cross_population_prediction <- function(train_ids,
                                                pred_ids,
                                                train_pop_name,
                                                pred_pop_name,
                                                cv_repeats = 20,
                                                pred_samples = 50,
                                                pred_sample_frac = 0.8) {
  
  cat(paste0("\n\n======================================================================\n"))
  cat(paste0("  Scenario: Train on '", train_pop_name, "' -> Predict on '", pred_pop_name, "'\n"))
  cat(paste0("======================================================================\n"))
  
  # --- Step 1: Find the best model using repeated CV within the training population ---
  cat(paste0("\n[STEP 1] Performing ", cv_repeats, " repeats of 5-fold CV on '", train_pop_name,
             "' using ", NUM_CORES_TO_USE, " cores to find the best model...\n"))
  
  # Prepare data subsets for the training population
  geno_train_pop <- genotypeMatrix[train_ids, ]
  pheno_train_pop <- phenotypeVector[train_ids]
  
  # Subset relationship matrices for the training population
  G_train_pop <- G_full[train_ids, train_ids]
  D_train_pop <- D_full[train_ids, train_ids]
  
  # This function runs ONE full repetition of 5-fold CV. It will be sent to the parallel workers.
  run_one_repetition <- function(rep_id) {
    
    # List to store results for this single repetition
    rep_results_list <- list()
    
    # Ensure all necessary packages are available on each parallel worker
    suppressPackageStartupMessages({
      library(tidyverse); library(BGLR); library(Matrix); library(glmnet);
      library(keras); library(caret); library(sommer)
    })
    
    # Helper function to standardize result data frames for robust binding.
    # This ensures every data frame has the same columns in the same order.
    all_possible_columns <- c(
      "Repetition", "Fold", "Model", "Cor", "alpha", "varA", "varD",
      "mlp_neurons", "mlp_dropout", "cnn_filters", "cnn_kernel_size"
    )
    
    standardize_df <- function(df) {
      missing_cols <- setdiff(all_possible_columns, names(df))
      if (length(missing_cols) > 0) {
        df[missing_cols] <- NA
      }
      return(df[, all_possible_columns])
    }
    
    cat(paste0("  [CV] Starting Repetition ", rep_id, "/", cv_repeats, "...\n"))
    folds <- createFolds(pheno_train_pop, k = 5, list = TRUE, returnTrain = FALSE)
    
    for (i in 1:length(folds)) {
      test_indices_local <- folds[[i]]
      train_indices_local <- setdiff(1:length(pheno_train_pop), test_indices_local)
      
      # Use local indices to get IDs for this fold
      fold_train_ids <- names(pheno_train_pop[train_indices_local])
      fold_test_ids <- names(pheno_train_pop[test_indices_local])
      
      phenoTrain <- pheno_train_pop[fold_train_ids]
      phenoTest <- pheno_train_pop[fold_test_ids]
      genoTrain <- geno_train_pop[fold_train_ids, ]
      genoTest <- geno_train_pop[fold_test_ids, ]
      
      # Create phenotype vector with NAs in the test set positions for BGLR/sommer
      pheno_with_NAs_cv <- pheno_train_pop
      pheno_with_NAs_cv[test_indices_local] <- NA
      
      # --- 1. glmnet Family ---
      tryCatch({
        cv_ridge <- cv.glmnet(genoTrain, phenoTrain, alpha = 0, family = "gaussian")
        pred_ridge <- predict(cv_ridge, newx = genoTest, s = "lambda.min")[, 1]
        df_ridge <- data.frame(Repetition = rep_id, Fold = i, Model = "Ridge", Cor = cor(pred_ridge, phenoTest, use = "complete.obs"), alpha = 0)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ridge)
        
        cv_lasso <- cv.glmnet(genoTrain, phenoTrain, alpha = 1, family = "gaussian")
        pred_lasso <- predict(cv_lasso, newx = genoTest, s = "lambda.min")[, 1]
        df_lasso <- data.frame(Repetition = rep_id, Fold = i, Model = "LASSO", Cor = cor(pred_lasso, phenoTest, use = "complete.obs"), alpha = 1)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_lasso)
        
        best_alpha <- NA; best_lambda <- NA; best_mse <- Inf
        for (a in seq(0, 1, by = 0.2)) {
          cv_fit <- cv.glmnet(genoTrain, phenoTrain, alpha = a, family="gaussian")
          current_mse <- min(cv_fit$cvm, na.rm = TRUE)
          if (is.finite(current_mse) && current_mse < best_mse) {
            best_mse <- current_mse; best_alpha <- a; best_lambda <- cv_fit$lambda.min
          }
        }
        fit_en <- glmnet(genoTrain, phenoTrain, alpha = best_alpha, lambda = best_lambda, family="gaussian")
        pred_en <- predict(fit_en, newx = genoTest)[, 1]
        df_en <- data.frame(Repetition = rep_id, Fold = i, Model = "Elastic Net", Cor = cor(pred_en, phenoTest, use = "complete.obs"), alpha = best_alpha)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_en)
      }, error = function(e) { cat(paste("      - glmnet failed in Rep", rep_id, "Fold", i, "\n")) })
      
      # --- 2. BGLR Family ---
      tryCatch({
        models_bglr <- list(
          BRR = list(list(X = geno_train_pop, model = "BRR")),
          BayesA = list(list(X = geno_train_pop, model = "BayesA")),
          BayesB = list(list(X = geno_train_pop, model = "BayesB")),
          BayesC = list(list(X = geno_train_pop, model = "BayesC")),
          `Bayes G-BLUP` = list(list(K = G_train_pop, model = "RKHS"))
        )
        for(m_name in names(models_bglr)) {
          fit_bglr <- BGLR(y = pheno_with_NAs_cv, ETA = models_bglr[[m_name]], nIter = 10000, burnIn = 2500, verbose = FALSE)
          pred_bglr <- fit_bglr$yHat[test_indices_local]
          df_bglr <- data.frame(Repetition = rep_id, Fold = i, Model = m_name, Cor = cor(pred_bglr, phenoTest, use = "complete.obs"))
          rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_bglr)
        }
      }, error = function(e) { cat(paste("      - BGLR failed in Rep", rep_id, "Fold", i, "\n")) })
      
      # --- 3. sommer GBLUP Family ---
      tryCatch({ # GBLUP
        sommer_data_gblup <- data.frame(ID = names(pheno_train_pop), y = pheno_with_NAs_cv)
        sommer_data_gblup$ID <- factor(sommer_data_gblup$ID, levels = rownames(G_train_pop))
        
        fit_gblup <- mmes(fixed = y ~ 1, random = ~vsm(ism(ID), Gu = G_train_pop), rcov = ~units,
                          data = sommer_data_gblup, naMethodY = "include", verbose = FALSE)
        pred_gblup <- fitted(fit_gblup)[test_indices_local]
        df_gblup <- data.frame(Repetition = rep_id, Fold = i, Model = "GBLUP", Cor = cor(pred_gblup, phenoTest, use = "complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_gblup)
      }, error = function(e) { cat(paste("      - GBLUP failed in Rep", rep_id, "Fold", i, "\n")) })
      
      tryCatch({ # AD-GBLUP
        sommer_data_ad <- data.frame(ID = names(pheno_train_pop), y = pheno_with_NAs_cv)
        sommer_data_ad$ID_A <- factor(sommer_data_ad$ID, levels = rownames(G_train_pop))
        sommer_data_ad$ID_D <- factor(sommer_data_ad$ID, levels = rownames(D_train_pop))
        
        fit_ad <- mmes(fixed = y ~ 1, random = ~vsm(ism(ID_A), Gu = G_train_pop) + vsm(ism(ID_D), Gu = D_train_pop),
                       rcov = ~units, data = sommer_data_ad, naMethodY = "include", verbose = FALSE)
        pred_ad <- fitted(fit_ad)[test_indices_local]
        var_a <- fit_ad$theta[[1]][1]; if (length(var_a) == 0) var_a <- NA
        var_d <- fit_ad$theta[[2]][1]; if (length(var_d) == 0) var_d <- NA
        df_ad <- data.frame(Repetition = rep_id, Fold = i, Model = "AD-GBLUP", Cor = cor(pred_ad, phenoTest, use = "complete.obs"), varA = var_a, varD = var_d)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ad)
      }, error = function(e) { cat(paste("      - AD-GBLUP failed in Rep", rep_id, "Fold", i, "\n")) })
      
      # --- 4. ssGBLUP model (Corrected CV Logic) ---
      tryCatch({
        # CRITICAL: The H-inverse matrix MUST be recalculated for the specific set of individuals in each CV fold.
        # It is incorrect to simply subset the pre-calculated H-inverse of the full population.
        
        # 1. Define all individuals for this fold: current training set + all parents.
        current_train_ids <- names(phenoTrain)
        cv_ids_for_ssgblup <- unique(c(current_train_ids, parents))
        
        # 2. Extract corresponding A and G sub-matrices.
        A_cv <- A_full[cv_ids_for_ssgblup, cv_ids_for_ssgblup]
        g_cv_ids <- intersect(cv_ids_for_ssgblup, rownames(G_full))
        G_cv <- G_full[g_cv_ids, g_cv_ids]
        
        # 3. Recalculate H-inverse from scratch for this specific group.
        Hinv_cv <- doH_inverse(A_cv, G_cv)
        
        # 4. Prepare phenotype vector with NAs for unphenotyped parents.
        pheno_cv_ssgblup <- rep(NA, length(cv_ids_for_ssgblup))
        names(pheno_cv_ssgblup) <- cv_ids_for_ssgblup
        pheno_cv_ssgblup[current_train_ids] <- phenoTrain # Fill in known training phenotypes
        
        # 5. Prepare data for sommer and run the model.
        sommer_data_ss <- data.frame(ID = names(pheno_cv_ssgblup), y = pheno_cv_ssgblup)
        sommer_data_ss$ID <- factor(sommer_data_ss$ID, levels = rownames(Hinv_cv))
        
        fit_ss <- mmes(fixed = y~1, random = ~vsm(ism(ID), Gu = Hinv_cv), rcov = ~units,
                       data=sommer_data_ss, naMethodY = "include", verbose=F, henderson=T)
        
        # 6. Extract predictions for the test individuals by name.
        pred_ssgblup_all <- fitted(fit_ss)
        pred_ssgblup <- pred_ssgblup_all[fold_test_ids]
        df_ssgblup <- data.frame(Repetition = rep_id, Fold = i, Model = "ssGBLUP", Cor = cor(pred_ssgblup, phenoTest, use = "complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ssgblup)
        
      }, error = function(e) {
        cat(paste("      - ssGBLUP failed in Rep", rep_id, "Fold", i, "ERROR:", e$message, "\n"))
      })
      
      # --- 5. Deep Learning Models with Hyperparameter Tuning ---
      train_mean <- colMeans(genoTrain); train_sd <- apply(genoTrain, 2, sd); train_sd[train_sd == 0] <- 1
      genoTrain_scaled <- scale(genoTrain, center = train_mean, scale = train_sd)
      genoTest_scaled <- scale(genoTest, center = train_mean, scale = train_sd)
      
      tryCatch({ # MLP
        mlp_param_grid <- expand.grid(dropout_rate = c(0.3, 0.5), neurons = c(64, 128))
        best_val_loss <- Inf; best_mlp_params <- list(dropout_rate=NA, neurons=NA)
        for(j in 1:nrow(mlp_param_grid)) {
          params <- mlp_param_grid[j, ]
          model <- keras_model_sequential() %>%
            layer_dense(units = params$neurons, activation = "relu", input_shape = ncol(genoTrain)) %>%
            layer_dropout(rate = params$dropout_rate) %>% layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
          hist <- model %>% fit(genoTrain_scaled, phenoTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
          val_loss <- min(hist$metrics$val_loss, na.rm=T)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
        }
        final_model <- keras_model_sequential() %>%
          layer_dense(units = best_mlp_params$neurons, activation = "relu", input_shape = ncol(genoTrain)) %>%
          layer_dropout(rate = best_mlp_params$dropout_rate) %>% layer_dense(units = 1)
        final_model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
        final_model %>% fit(genoTrain_scaled, phenoTrain, epochs = 40, batch_size=32, verbose = 0)
        pred_mlp <- final_model %>% predict(genoTest_scaled, verbose=0)
        df_mlp <- data.frame(Repetition = rep_id, Fold = i, Model = "MLP", Cor = cor(pred_mlp[, 1], phenoTest, use = "complete.obs"), mlp_neurons = best_mlp_params$neurons, mlp_dropout = best_mlp_params$dropout_rate)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_mlp)
      }, error = function(e){ cat(paste("      - MLP failed in Rep", rep_id, "Fold", i, "\n")) })
      
      tryCatch({ # CNN
        xtrain_cnn <- array(genoTrain_scaled, dim = c(nrow(genoTrain_scaled), ncol(genoTrain_scaled), 1))
        xtest_cnn <- array(genoTest_scaled, dim = c(nrow(genoTest_scaled), ncol(genoTest_scaled), 1))
        cnn_param_grid <- expand.grid(filters = c(32, 64), kernel_size = c(5, 10))
        best_val_loss <- Inf; best_cnn_params <- list(filters=NA, kernel_size=NA)
        for(j in 1:nrow(cnn_param_grid)) {
          params <- cnn_param_grid[j, ]
          model <- keras_model_sequential() %>%
            layer_conv_1d(filters = params$filters, kernel_size = params$kernel_size, activation = "relu", input_shape = c(ncol(genoTrain), 1)) %>%
            layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>% layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = "rmsprop")
          hist <- model %>% fit(xtrain_cnn, phenoTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
          val_loss <- min(hist$metrics$val_loss, na.rm=T)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_cnn_params <- params }
        }
        final_model <- keras_model_sequential() %>%
          layer_conv_1d(filters = best_cnn_params$filters, kernel_size = best_cnn_params$kernel_size, activation = "relu", input_shape = c(ncol(genoTrain), 1)) %>%
          layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>% layer_dense(units = 1)
        final_model %>% compile(loss = "mse", optimizer = "rmsprop")
        final_model %>% fit(xtrain_cnn, phenoTrain, epochs = 40, batch_size=32, verbose = 0)
        pred_cnn <- final_model %>% predict(xtest_cnn, verbose=0)
        df_cnn <- data.frame(Repetition = rep_id, Fold = i, Model = "CNN", Cor = cor(pred_cnn[, 1], phenoTest, use = "complete.obs"), cnn_filters = best_cnn_params$filters, cnn_kernel_size = best_cnn_params$kernel_size)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_cnn)
      }, error = function(e){ cat(paste("      - CNN failed in Rep", rep_id, "Fold", i, "\n")) })
      
    }
    return(dplyr::bind_rows(rep_results_list))
  }
  
  # Execute the CV repetitions in parallel using the pre-configured cluster.
  cv_results_df <- future_map_dfr(.x = 1:cv_repeats, .f = run_one_repetition, .options = furrr_options(seed = TRUE))
  
  # --- Summarize CV results to find the best model ---
  cv_summary_stats <- cv_results_df %>%
    filter(!is.na(Cor)) %>%
    group_by(Model) %>%
    summarise(Mean_Cor = mean(Cor, na.rm = TRUE), SD_Cor = sd(Cor, na.rm = TRUE), .groups = 'drop') %>%
    arrange(desc(Mean_Cor))
  
  cat("\n[INFO] Cross-validation within training population finished.\n")
  cat("[INFO] Performance of all models in CV:\n")
  print(cv_summary_stats)
  
  best_model_name <- cv_summary_stats$Model[1]
  cat(paste0("\n[INFO] Best model identified: '", best_model_name, "' with mean CV accuracy of ", round(cv_summary_stats$Mean_Cor[1], 4), ".\n"))
  
  
  # --- Step 2: Train BEST model and perform SAMPLING VALIDATION on prediction population ---
  cat(paste0("\n[STEP 2] Training final '", best_model_name, "' model on all '", train_pop_name, "' data...\n"))
  final_train_geno <- genotypeMatrix[train_ids, ]
  final_train_pheno <- phenotypeVector[train_ids]
  
  prediction_accuracies <- numeric(pred_samples)
  
  cat(paste0("[INFO] Starting sampling validation on '", pred_pop_name, "' (", pred_samples, " samples of ", pred_sample_frac * 100, "%)...\n"))
  
  # This loop is sequential as each iteration is fast.
  for (iter in 1:pred_samples) {
    cat(paste0("  - Prediction Sample ", iter, "/", pred_samples, "...\n"))
    set.seed(42 + iter)
    sample_pred_ids <- sample(pred_ids, size = floor(length(pred_ids) * pred_sample_frac))
    
    final_pred_geno_sample <- genotypeMatrix[sample_pred_ids, ]
    final_pred_pheno_sample <- phenotypeVector[sample_pred_ids]
    
    predictions_sample <- NULL
    
    # --- PREDICTION LOGIC (UNIFIED) ---
    if (best_model_name %in% c("Ridge", "LASSO", "Elastic Net")) {
      alpha_val <- if (best_model_name == "Ridge") 0 else if (best_model_name == "LASSO") 1 else median(cv_results_df$alpha, na.rm = TRUE)
      final_model_trained <- cv.glmnet(final_train_geno, final_train_pheno, alpha = alpha_val, family = "gaussian")
      predictions_sample <- predict(final_model_trained, newx = final_pred_geno_sample, s = "lambda.min")[, 1]
      
    } else if (best_model_name %in% c("BRR", "BayesA", "BayesB", "BayesC", "Bayes G-BLUP")) {
      # Use BGLR's ability to predict on NAs for efficiency.
      ids_for_bglr <- c(train_ids, sample_pred_ids)
      y_bglr <- phenotypeVector[ids_for_bglr]; y_bglr[sample_pred_ids] <- NA
      
      model_name_bglr <- if(best_model_name == "Bayes G-BLUP") "RKHS" else best_model_name
      ETA_final <- if(model_name_bglr != "RKHS") {
        list(list(X = genotypeMatrix[ids_for_bglr,], model = model_name_bglr))
      } else {
        list(list(K = G_full[ids_for_bglr, ids_for_bglr], model = "RKHS"))
      }
      fit_bglr_final <- BGLR(y = y_bglr, ETA = ETA_final, nIter = 10000, burnIn = 2500, verbose = FALSE)
      predictions_sample <- fit_bglr_final$yHat[names(y_bglr) %in% sample_pred_ids]
      
    } else if (best_model_name %in% c("GBLUP", "AD-GBLUP", "ssGBLUP")) {
      # Use sommer's ability to predict on NAs.
      all_final_ids <- unique(c(train_ids, sample_pred_ids, parents))
      pheno_final <- rep(NA, length(all_final_ids)); names(pheno_final) <- all_final_ids
      pheno_final[train_ids] <- final_train_pheno
      
      sommer_data_final <- data.frame(ID=names(pheno_final), y=pheno_final)
      
      final_fit <- if(best_model_name == "GBLUP") {
        sommer_data_final$ID <- factor(sommer_data_final$ID, levels=rownames(G_full))
        mmes(fixed = y~1, random = ~vsm(ism(ID), Gu=G_full), rcov = ~units, data=sommer_data_final, naMethodY="include", verbose=F)
      } else if (best_model_name == "AD-GBLUP") {
        sommer_data_final$ID_A <- factor(sommer_data_final$ID, levels=rownames(G_full))
        sommer_data_final$ID_D <- factor(sommer_data_final$ID, levels=rownames(D_full))
        mmes(fixed = y~1, random=~vsm(ism(ID_A), Gu=G_full) + vsm(ism(ID_D), Gu=D_full), rcov=~units, data=sommer_data_final, naMethodY="include", verbose=F)
      } else { # ssGBLUP
        sommer_data_final$ID <- factor(sommer_data_final$ID, levels=rownames(Hinv_full))
        mmes(fixed = y~1, random = ~vsm(ism(ID), Gu=Hinv_full), rcov = ~units, data=sommer_data_final, naMethodY="include", verbose=F, henderson=T)
      }
      predictions_sample <- fitted(final_fit)[sample_pred_ids]
      
    } else if (best_model_name %in% c("MLP", "CNN")) {
      get_mode <- function(v) { v <- v[!is.na(v)]; uniqv <- unique(v); uniqv[which.max(tabulate(match(v, uniqv)))] }
      
      train_mean <- colMeans(final_train_geno); train_sd <- apply(final_train_geno, 2, sd); train_sd[train_sd == 0] <- 1
      final_train_geno_scaled <- scale(final_train_geno, center = train_mean, scale = train_sd)
      
      final_model_trained <- if(best_model_name == "MLP") {
        best_params <- cv_results_df %>% filter(Model=="MLP") %>% summarise(neurons=get_mode(mlp_neurons), dropout=get_mode(mlp_dropout))
        model <- keras_model_sequential() %>%
          layer_dense(units = best_params$neurons, activation = "relu", input_shape = numMarkers) %>%
          layer_dropout(rate = best_params$dropout) %>% layer_dense(units = 1)
        model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
        model %>% fit(final_train_geno_scaled, final_train_pheno, epochs = 50, batch_size=32, verbose = 0)
        model
      } else { # CNN
        best_params <- cv_results_df %>% filter(Model=="CNN") %>% summarise(filters=get_mode(cnn_filters), kernel_size=get_mode(cnn_kernel_size))
        train_cnn <- array(final_train_geno_scaled, dim = c(nrow(final_train_geno_scaled), numMarkers, 1))
        model <- keras_model_sequential() %>%
          layer_conv_1d(filters = best_params$filters, kernel_size = best_params$kernel_size, activation = "relu", input_shape = c(numMarkers, 1)) %>%
          layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>% layer_dense(units = 1)
        model %>% compile(loss = "mse", optimizer = "rmsprop")
        model %>% fit(train_cnn, final_train_pheno, epochs = 50, batch_size=32, verbose = 0)
        model
      }
      pred_geno_scaled <- scale(final_pred_geno_sample, center = train_mean, scale = train_sd)
      pred_input <- if(best_model_name == "MLP") { pred_geno_scaled } else { array(pred_geno_scaled, dim=c(nrow(pred_geno_scaled), numMarkers, 1)) }
      predictions_sample <- final_model_trained %>% predict(pred_input, verbose=0)
      predictions_sample <- predictions_sample[,1]
    }
    
    if (!is.null(predictions_sample)) {
      prediction_accuracies[iter] <- cor(predictions_sample, final_pred_pheno_sample, use = "complete.obs")
    } else {
      prediction_accuracies[iter] <- NA
      cat(paste0("    !-> Prediction failed for model '", best_model_name, "' in iteration ", iter, "\n"))
    }
  }
  
  avg_final_pred_accuracy <- mean(prediction_accuracies, na.rm = TRUE)
  sd_final_pred_accuracy <- sd(prediction_accuracies, na.rm = TRUE)
  
  cat(paste0("\n------------------------------------------------------------------\n"))
  cat(paste0(">>> Avg. Cross-Pop Prediction Accuracy: ", round(avg_final_pred_accuracy, 4), " ± ", round(sd_final_pred_accuracy, 4), "\n"))
  cat(paste0("    (Trained on ", train_pop_name, ", Predicted on ", pred_pop_name, ")\n"))
  cat(paste0("------------------------------------------------------------------\n"))
  
  # Return a summary tibble for this scenario
  return(tibble::tibble(
    Training_Population = train_pop_name,
    Prediction_Population = pred_pop_name,
    Best_Model_in_CV = best_model_name,
    CV_Accuracy_of_Best_Model = cv_summary_stats$Mean_Cor[1],
    Avg_Cross_Pop_Accuracy = avg_final_pred_accuracy,
    SD_Cross_Pop_Accuracy = sd_final_pred_accuracy,
    CV_Summary = list(cv_summary_stats)
  ))
}


# =====================================================================================
# Part 4: Execute Both Scenarios and Report Final Results
# =====================================================================================

# --- Run Scenario A: Train on NH/HN, Predict on HP ---
result_A <- perform_cross_population_prediction(
  train_ids = pop_NH_HN_ids,
  pred_ids = pop_HP_ids,
  train_pop_name = "NH_HN",
  pred_pop_name = "HP",
  cv_repeats = CV_REPETITIONS,
  pred_samples = PRED_SAMPLES,
  pred_sample_frac = PRED_SAMPLE_FRAC
)

# --- Run Scenario B: Train on HP, Predict on NH/HN ---
result_B <- perform_cross_population_prediction(
  train_ids = pop_HP_ids,
  pred_ids = pop_NH_HN_ids,
  train_pop_name = "HP",
  pred_pop_name = "NH_HN",
  cv_repeats = CV_REPETITIONS,
  pred_samples = PRED_SAMPLES,
  pred_sample_frac = PRED_SAMPLE_FRAC
)

# =====================================================================================
# Part 5: Final Summary and Cleanup
# =====================================================================================

# --- IMPORTANT: Stop the parallel cluster ---
cat("\n[PARALLEL] Stopping the parallel cluster...\n")
parallel::stopCluster(cl)
plan(sequential) # Return to sequential processing

# --- Final Summary Table ---
final_summary_printable <- bind_rows(
  result_A %>% select(-CV_Summary),
  result_B %>% select(-CV_Summary)
)

cat("\n\n====================================================================\n")
cat("                         FINAL CROSS-POPULATION PREDICTION SUMMARY\n")
cat("====================================================================\n\n")
print(final_summary_printable)

cat("\n\n--- Detailed CV Performance for Scenario A (Train: NH_HN) ---\n")
print(result_A$CV_Summary[[1]])
cat("\n--- Detailed CV Performance for Scenario B (Train: HP) ---\n")
print(result_B$CV_Summary[[1]])

# --- Save the final summary to a CSV file ---
output_filename <- paste0("GS_Cross_Population_Summary_", TRAIT_OF_INTEREST, ".csv")
write.csv(final_summary_printable, output_filename, row.names = FALSE)
cat(paste0("\n\n[COMPLETE] Analysis finished. Final summary saved to '", output_filename, "'\n"))
# =====================================================================================
#
#   Cross-Population Genomic Prediction Script (Revised and Synchronized)
#
# Description:
#   This script evaluates cross-population genomic prediction accuracy. Its model
#   implementations are synchronized with the 'full_pop.R' benchmark script.
#   The workflow is as follows:
#   1. Load and preprocess genotype and phenotype data.
#   2. Split individuals into two distinct populations (e.g., NH/HN and HP).
#   3. Perform a Principal Component Analysis (PCA) to visualize population structure.
#   4. For each scenario (e.g., Train on NH/HN, Predict on HP):
#      a. Identify the best-performing prediction model through parallelized, repeated
#         5-fold cross-validation *within the training population*. This involves
#         hyperparameter tuning for deep learning models.
#      b. Train the single best model on the *entire* training population.
#      c. Assess the final model's prediction accuracy on the unseen prediction
#         population using a repeated sampling strategy to ensure robust results.
#   5. Report the final cross-population prediction accuracies for all scenarios.
#
# Last Modified: Aug 04, 2025
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
# Ensure all necessary packages are installed before running.
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
# This step is crucial for ensuring parallel workers (especially for Keras/TensorFlow)
# can find and use the correct Python environment.
cat("Finding Python executable from the 'reseq' conda environment...\n")
use_condaenv("reseq", required = TRUE)

# Store the exact path to the python executable to be passed to each worker.
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
# Create the output directory for plots if it doesn't already exist.
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)


# =====================================================================================
# Part 0.5: Load, Clean, and Preprocess Data
# =====================================================================================
cat("\n[DATA] Loading and preprocessing data...\n")

# --- User Configuration ---
GENOTYPE_FILE     <- "genotype.dosages.tsv"
PHENOTYPE_FILE    <- "phenotype_BLUPs.csv"
TRAIT_OF_INTEREST <- "fruit_weight" # <-- !!! EDIT THIS TO YOUR TRAIT OF INTEREST !!!

# --- Load Genotype Data (aligned with full_pop.R method) ---
dosage_raw <- read.csv(GENOTYPE_FILE, check.names = FALSE, sep = "\t", header = TRUE)
marker_ids <- paste0(dosage_raw$CHROM, ":", dosage_raw$POS)

dosage_only <- dosage_raw[, 5:ncol(dosage_raw)]
dosage_only_matrix <- apply(dosage_only, 2, as.numeric)

genotypeMatrix_raw <- t(dosage_only_matrix)
rownames(genotypeMatrix_raw) <- colnames(dosage_only)
colnames(genotypeMatrix_raw) <- marker_ids
cat(paste0("[DATA] Loaded genotype data with ", nrow(genotypeMatrix_raw), " individuals and ", ncol(genotypeMatrix_raw), " markers.\n"))

# --- Load Phenotype Data ---
pheno_raw <- read.csv(PHENOTYPE_FILE, row.names = 1, check.names = FALSE, sep = ",", header = TRUE)
pheno_df <- as.data.frame(pheno_raw)
cat(paste0("[DATA] Loaded phenotype data for ", nrow(pheno_df), " individuals.\n"))

# --- Data Alignment and Cleaning ---
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
cat(paste0("[DATA] Found ", length(common_individuals), " individuals with both genotype and phenotype data.\n"))

genotypeMatrix <- genotypeMatrix_raw[common_individuals, ]

# Ensure the specified trait exists in the phenotype data
if (!TRAIT_OF_INTEREST %in% colnames(pheno_df)) {
  stop(paste("Error: The specified trait '", TRAIT_OF_INTEREST, "' was not found in the phenotype file. Please check spelling and case."))
}

# Select the trait of interest and create a named vector
phenotype_full_df <- pheno_df[common_individuals, , drop = FALSE] %>%
  dplyr::select(all_of(TRAIT_OF_INTEREST))

phenotypeVector <- phenotype_full_df[[TRAIT_OF_INTEREST]]
names(phenotypeVector) <- rownames(phenotype_full_df)

# --- Handle Missing Genotype Values (Mean Imputation) ---
cat("[DATA] Handling missing genotype values using mean imputation...\n")
impute_mean <- function(x) {
  mean_val <- mean(x, na.rm = TRUE)
  if (is.nan(mean_val)) mean_val <- 0 # Handle cases where a marker is all NA
  x[is.na(x)] <- round(mean_val)
  return(x)
}
genotypeMatrix <- apply(genotypeMatrix, 2, impute_mean)
if(sum(is.na(genotypeMatrix)) > 0) {
  warning("Missing values still exist after imputation. This can happen if a marker is missing for all individuals. Please check your data.")
}

# Store final dimensions
numIndividuals <- nrow(genotypeMatrix)
numMarkers <- ncol(genotypeMatrix)
cat(paste0("[DATA] Data preparation complete: ", numIndividuals, " individuals, ", numMarkers, " SNP markers.\n"))


# =====================================================================================
# Part 0.6: Split Data into Defined Populations
# =====================================================================================
cat("\n[DATA] Splitting data into specified populations based on ID prefixes...\n")
all_ids <- rownames(genotypeMatrix)
# Population 1: Individuals with IDs starting with "NH" or "HN"
pop_NH_HN_ids <- all_ids[startsWith(all_ids, "NH") | startsWith(all_ids, "HN")]
# Population 2: Individuals with IDs starting with "HP"
pop_HP_ids <- all_ids[startsWith(all_ids, "HP")]

cat(paste0("  - Population 1 (NH/HN): ", length(pop_NH_HN_ids), " individuals.\n"))
cat(paste0("  - Population 2 (HP): ", length(pop_HP_ids), " individuals.\n"))

# Stop if either population is empty
if(length(pop_NH_HN_ids) == 0 | length(pop_HP_ids) == 0) {
  stop("One or both defined populations have zero individuals. Please check your sample ID prefixes (e.g., 'NH', 'HP').")
}

# =====================================================================================
# Part 0.7: Population Structure Analysis via PCA
# =====================================================================================
cat("\n[PCA] Performing Principal Component Analysis to visualize population structure...\n")
# Perform PCA only on markers that have non-zero variance
non_zero_var_cols <- which(apply(genotypeMatrix, 2, var, na.rm = TRUE) > 1e-6)
pca_results <- prcomp(genotypeMatrix[, non_zero_var_cols], center = TRUE, scale. = TRUE)

# Create a data frame for plotting
pca_df <- as.data.frame(pca_results$x) %>%
  dplyr::mutate(ID = rownames(.),
                Population = case_when(
                  ID %in% pop_NH_HN_ids ~ "NH_HN",
                  ID %in% pop_HP_ids    ~ "HP",
                  TRUE                  ~ "Other" # Should not happen with current data
                ))

# Generate the PCA plot
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

# Save the plot to a PDF file
pca_filename <- paste0("plots/PCA_Plot_CrossPop_", TRAIT_OF_INTEREST, ".pdf")
ggsave(pca_filename, pca_plot, width = 8, height = 6, device = cairo_pdf)
cat(paste0("[PCA] PCA plot saved as '", pca_filename, "'.\n"))


# =====================================================================================
# Part 1: Helper Functions and Relationship Matrix Preparation
# =====================================================================================
cat("\n[PREP] Building pedigree and relationship matrices for the ENTIRE population...\n")
# These matrices are calculated once for the full dataset and then subsetted as needed
# to ensure consistency across all analyses.

# Source the function to create the dominance relationship matrix
source("get_DomRel_matrix.R")

# --- Build Pedigree Data Frame ---
parents <- c("HF1", "NZH2", "PJHH")
all_ped_ids <- unique(c(parents, all_ids))
ped_df <- data.frame(ID = all_ped_ids, Sire = 0, Dam = 0, stringsAsFactors = FALSE)
# Assign parents based on offspring ID prefixes
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) {
    ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) {
    ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "PJHH"
  }
}

# --- A-matrix (Pedigree-based Relationship Matrix) ---
A_full <- Amatrix(ped_df, ploidy = 4)
cat("[PREP] Full A-matrix (pedigree) built successfully.\n")

# --- G-matrix (Additive Genomic Relationship Matrix) ---
# Using VanRaden method for polyploids via AGHmatrix package
G_full <- Gmatrix(genotypeMatrix, method = "VanRaden", ploidy = 4)
G_full <- G_full + diag(nrow(G_full)) * 1e-4 # Regularize to ensure positive definiteness
cat("[PREP] Full G-matrix (additive) built successfully.\n")

# --- D-matrix (Dominance Genomic Relationship Matrix) ---
D_raw <- get_DomRel(genotypeMatrix, ploidy = 4)
# Center the D matrix
Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw))
D_centered <- Ic %*% D_raw %*% Ic
# Regularize the centered D matrix
D_full <- D_centered + diag(nrow(D_centered)) * 1e-4
rownames(D_full) <- rownames(genotypeMatrix)
colnames(D_full) <- rownames(genotypeMatrix)
cat("[PREP] Full D-matrix (dominance) built successfully.\n")

# --- H-inverse Matrix (for ssGBLUP) ---
# This helper function calculates the inverse of the H matrix, which combines
# pedigree and genomic information.
doH_inverse <- function(pedigreeRelationshipMatrix, grmForGenotyped) {
  genotypedIndicesInPedigree <- match(rownames(grmForGenotyped), rownames(pedigreeRelationshipMatrix))
  grmInverse <- solve(grmForGenotyped)
  A22 <- pedigreeRelationshipMatrix[genotypedIndicesInPedigree, genotypedIndicesInPedigree]
  pedigreeRelationshipInverseForGenotyped <- solve(A22)
  pedigreeRelationshipInverse <- solve(pedigreeRelationshipMatrix)
  
  hMatrixInverse <- pedigreeRelationshipInverse
  hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] <-
    hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] + grmInverse - pedigreeRelationshipInverseForGenotyped
  
  attr(hMatrixInverse, 'inverse') <- TRUE # Important for sommer package
  return(hMatrixInverse)
}

# Calculate H-inverse for the entire population.
Hinv_full <- doH_inverse(A_full, G_full)
cat("[PREP] Full H-inverse matrix (for ssGBLUP) built successfully.\n")

# =====================================================================================
# Part 2: Robust Parallel Backend Setup
# =====================================================================================
# --- User Configuration for Analysis ---
CV_REPETITIONS   <- 20  # Number of repeated CVs to find the best model
NUM_CORES_TO_USE <- 20  # <-- !!! SET THE NUMBER OF CPU CORES TO USE !!!
PRED_SAMPLES     <- 50  # Number of sampling iterations for final validation
PRED_SAMPLE_FRAC <- 0.8 # Fraction of prediction population to use in each sample

cat("\n[PARALLEL] Setting up robust parallel cluster with direct path injection...\n")
cl <- parallel::makeCluster(NUM_CORES_TO_USE)
# Export the Python path to each worker
parallel::clusterExport(cl, "python_exe_path")
# Initialize the correct python environment and load keras on each worker
parallel::clusterEvalQ(cl, {
  library(reticulate)
  use_python(python_exe_path, required = TRUE)
  library(keras)
})
# Set the parallel plan using 'future'
plan(cluster, workers = cl)
cat("[PARALLEL] Cluster is ready. Starting computations.\n")


# =====================================================================================
# Part 3: Main Analysis Function for a Single Cross-Population Scenario
# =====================================================================================

perform_cross_population_prediction <- function(train_ids,
                                                pred_ids,
                                                train_pop_name,
                                                pred_pop_name,
                                                cv_repeats = CV_REPETITIONS,
                                                pred_samples = PRED_SAMPLES,
                                                pred_sample_frac = PRED_SAMPLE_FRAC) {
  
  cat(paste0("\n\n======================================================================\n"))
  cat(paste0("  Scenario: Train on '", train_pop_name, "' -> Predict on '", pred_pop_name, "'\n"))
  cat(paste0("======================================================================\n"))
  
  # --- Step 1: Find the best model using repeated CV within the training population ---
  cat(paste0("\n[STEP 1] Performing ", cv_repeats, " repeats of 5-fold CV on '", train_pop_name,
             "' using ", NUM_CORES_TO_USE, " cores to find the best model...\n"))
  
  # Prepare data subsets for the training population
  geno_train_pop <- genotypeMatrix[train_ids, ]
  pheno_train_pop <- phenotypeVector[train_ids]
  n_train_pop <- length(pheno_train_pop)
  
  # Subset relationship matrices for the training population
  G_train_pop <- G_full[train_ids, train_ids]
  D_train_pop <- D_full[train_ids, train_ids]
  
  # For ssGBLUP, we need the pedigree of the training population PLUS founders
  ped_ids_train <- unique(c(train_ids, parents))
  A_train <- A_full[ped_ids_train, ped_ids_train]
  Hinv_train <- doH_inverse(A_train, G_train_pop)
  
  # This function runs ONE full repetition of 5-fold CV. It will be sent to the parallel workers.
  run_one_repetition <- function(rep_id) {
    
    rep_results_list <- list()
    
    # Load required packages on each parallel worker
    suppressPackageStartupMessages({
      library(tidyverse); library(BGLR); library(Matrix); library(glmnet);
      library(keras); library(caret); library(sommer)
    })
    
    # Standardize output dataframe to ensure all columns exist
    all_possible_columns <- c(
      "Repetition", "Fold", "Model", "Cor", "alpha", "varA", "varD",
      "mlp_neurons", "mlp_dropout", "cnn_filters", "cnn_kernel_size"
    )
    standardize_df <- function(df) {
      missing_cols <- setdiff(all_possible_columns, names(df))
      if (length(missing_cols) > 0) { df[missing_cols] <- NA }
      return(df[, all_possible_columns])
    }
    
    cat(paste0("  [CV] Starting Repetition ", rep_id, "/", cv_repeats, "...\n"))
    set.seed(42 + rep_id)
    folds <- createFolds(pheno_train_pop, k = 5, list = TRUE, returnTrain = FALSE)
    
    # Inner 5-Fold Cross-Validation Loop
    for (i in 1:length(folds)) {
      test_indices_local <- folds[[i]]
      train_indices_local <- setdiff(1:n_train_pop, test_indices_local)
      
      test_ids_local <- names(pheno_train_pop[test_indices_local])
      
      genoTrain <- geno_train_pop[train_indices_local, ]
      phenoTrain <- pheno_train_pop[train_indices_local]
      genoTest <- geno_train_pop[test_ids_local, ]
      phenoTest <- pheno_train_pop[test_ids_local]
      
      # Create a version of the phenotype vector with NAs for the test set
      pheno_with_NAs_cv <- pheno_train_pop
      pheno_with_NAs_cv[test_indices_local] <- NA
      
      # --- 1. glmnet Family (Ridge, LASSO, Elastic Net) ---
      tryCatch({
        cv_ridge <- cv.glmnet(genoTrain, phenoTrain, alpha = 0, family="gaussian")
        pred_ridge <- predict(cv_ridge, newx = genoTest, s = "lambda.min")[, 1]
        df_ridge <- data.frame(Repetition = rep_id, Fold = i, Model = "Ridge", Cor = cor(pred_ridge, phenoTest, use = "complete.obs"), alpha = 0)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ridge)
        
        cv_lasso <- cv.glmnet(genoTrain, phenoTrain, alpha = 1, family="gaussian")
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
      }, error = function(e) { cat(paste0("  ERROR in glmnet model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 2. GBLUP (using sommer) ---
      tryCatch({
        sommer_data_gblup <- data.frame(ID = names(pheno_train_pop), y = as.numeric(pheno_with_NAs_cv))
        sommer_data_gblup$ID <- factor(sommer_data_gblup$ID, levels = rownames(G_train_pop))
        
        fit_gblup <- mmes(
          fixed = y ~ 1, random = ~vsm(ism(ID), Gu = G_train_pop), rcov = ~units,
          data = sommer_data_gblup, naMethodY = "include", verbose = FALSE
        )
        
        predictions_table <- predict(fit_gblup, D = "ID")
        pred_gblup <- predictions_table$pvals[test_ids_local, "predicted.value"]
        
        df_gblup <- data.frame(Repetition = rep_id, Fold = i, Model = "GBLUP", Cor = cor(pred_gblup, phenoTest, use = "complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_gblup)
      }, error = function(e) { cat(paste0("  ERROR in GBLUP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 3. Bayesian Family (using BGLR) ---
      tryCatch({
        ETA_BRR <- list(list(X = geno_train_pop, model = "BRR"))
        ETA_BayesA <- list(list(X = geno_train_pop, model = "BayesA"))
        ETA_BayesB <- list(list(X = geno_train_pop, model = "BayesB"))
        ETA_BayesC <- list(list(X = geno_train_pop, model = "BayesC"))
        ETA_BGBLUP <- list(list(K = G_train_pop, model = "RKHS"))
        models_bglr <- list(BRR=ETA_BRR, BayesA=ETA_BayesA, BayesB=ETA_BayesB, BayesC=ETA_BayesC, `Bayes G-BLUP`=ETA_BGBLUP)
        for(m_name in names(models_bglr)){
          fit_bglr <- BGLR(y = pheno_with_NAs_cv, ETA = models_bglr[[m_name]], nIter = 10000, burnIn = 2500, verbose = FALSE, saveAt=paste0("rep_",rep_id,"_fold_",i,"_"))
          pred_bglr <- fit_bglr$yHat[test_indices_local]
          df_bglr <- data.frame(Repetition = rep_id, Fold = i, Model = m_name, Cor = cor(pred_bglr, phenoTest, use = "complete.obs"))
          rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_bglr)
        }
      }, error = function(e) { cat(paste0("  ERROR in BGLR model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 4. Single-Step GBLUP (ssGBLUP using sommer) ---
      tryCatch({
        # Create phenotype vector including founders for the ssGBLUP model
        pheno_vec_with_founders <- rep(NA, length(ped_ids_train))
        names(pheno_vec_with_founders) <- ped_ids_train
        pheno_vec_with_founders[names(pheno_train_pop)] <- pheno_train_pop
        
        # Mask test individuals for the current fold
        pheno_vec_with_founders[test_ids_local] <- NA
        
        sommer_data_ss <- data.frame(ID = names(pheno_vec_with_founders), y = pheno_vec_with_founders)
        sommer_data_ss$ID <- factor(sommer_data_ss$ID, levels = rownames(Hinv_train))
        
        fit_ssgblup <- mmes(
          fixed = y ~ 1, random = ~vsm(ism(ID), Gu = Hinv_train), rcov = ~units,
          data = sommer_data_ss, naMethodY = "include", verbose = FALSE, henderson = TRUE
        )
        
        predictions_table <- predict(fit_ssgblup, D = "ID")
        pred_ssgblup <- predictions_table$pvals[test_ids_local, "predicted.value"]
        
        df_ssgblup <- data.frame(Repetition = rep_id, Fold = i, Model = "ssGBLUP", Cor = cor(pred_ssgblup, phenoTest, use = "complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ssgblup)
      }, error = function(e) { cat(paste0("  ERROR in ssGBLUP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 5. Additive + Dominance GBLUP (AD-GBLUP) using sommer ---
      tryCatch({
        sommer_data_ad <- data.frame(ID = names(pheno_train_pop), y = as.numeric(pheno_with_NAs_cv))
        sommer_data_ad$ID_A <- factor(sommer_data_ad$ID, levels = rownames(G_train_pop))
        sommer_data_ad$ID_D <- factor(sommer_data_ad$ID, levels = rownames(D_train_pop))
        
        fit_ad_gblup <- mmes(
          fixed = y ~ 1, random = ~vsm(ism(ID_A), Gu = G_train_pop) + vsm(ism(ID_D), Gu = D_train_pop),
          rcov = ~ units, data = sommer_data_ad, naMethodY = "include", verbose = FALSE
        )
        
        predictions_table <- predict(fit_ad_gblup, D = "ID_A")
        pred_ad_gblup <- predictions_table$pvals[test_ids_local, "predicted.value"]
        
        var_a <- fit_ad_gblup$sigma[[1]]; var_d <- fit_ad_gblup$sigma[[2]]
        if (length(var_a) == 0) var_a <- NA
        if (length(var_d) == 0) var_d <- NA
        
        df_ad <- data.frame(Repetition=rep_id, Fold=i, Model="AD-GBLUP", Cor=cor(pred_ad_gblup, phenoTest, use="complete.obs"), varA=var_a, varD=var_d)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ad)
      }, error = function(e) { cat(paste0("  ERROR in AD-GBLUP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 6. Deep Learning Models (MLP & CNN) with Hyperparameter Tuning ---
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
            layer_dropout(rate = params$dropout_rate) %>%
            layer_dense(units = round(params$neurons/2), activation="relu") %>%
            layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
          hist <- model %>% fit(genoTrain_scaled, phenoTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
          val_loss <- min(hist$metrics$val_loss, na.rm=T)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
        }
        final_model <- keras_model_sequential(input_shape = c(ncol(genoTrain))) %>%
          layer_dense(units = best_mlp_params$neurons, activation = "relu") %>%
          layer_dropout(rate = best_mlp_params$dropout_rate) %>%
          layer_dense(units = round(best_mlp_params$neurons/2), activation="relu") %>%
          layer_dense(units = 1)
        final_model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
        final_model %>% fit(genoTrain_scaled, phenoTrain, epochs = 40, batch_size=32, verbose = 0)
        pred_mlp <- final_model %>% predict(genoTest_scaled, verbose = 0)
        df_mlp <- data.frame(Repetition=rep_id, Fold=i, Model="MLP", Cor=cor(pred_mlp[,1], phenoTest, use="complete.obs"), mlp_neurons=best_mlp_params$neurons, mlp_dropout=best_mlp_params$dropout_rate)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_mlp)
      }, error = function(e) { cat(paste0("  ERROR in MLP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      tryCatch({ # CNN
        xtrain_cnn <- array(genoTrain_scaled, dim = c(nrow(genoTrain_scaled), ncol(genoTrain_scaled), 1))
        xtest_cnn <- array(genoTest_scaled, dim = c(nrow(genoTest_scaled), ncol(genoTest_scaled), 1))
        cnn_param_grid <- expand.grid(filters = c(32, 64), kernel_size = c(5, 10))
        best_val_loss <- Inf; best_cnn_params <- list(filters=NA, kernel_size=NA)
        for(j in 1:nrow(cnn_param_grid)) {
          params <- cnn_param_grid[j, ]
          model <- keras_model_sequential(input_shape = c(ncol(genoTrain), 1)) %>%
            layer_conv_1d(filters = params$filters, kernel_size = params$kernel_size, activation = "relu") %>%
            layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
            layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = "rmsprop")
          hist <- model %>% fit(xtrain_cnn, phenoTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
          val_loss <- min(hist$metrics$val_loss, na.rm=T)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_cnn_params <- params }
        }
        final_model <- keras_model_sequential(input_shape = c(ncol(genoTrain), 1)) %>%
          layer_conv_1d(filters = best_cnn_params$filters, kernel_size = best_cnn_params$kernel_size, activation = "relu") %>%
          layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
          layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
        final_model %>% compile(loss = "mse", optimizer = "rmsprop")
        final_model %>% fit(xtrain_cnn, phenoTrain, epochs = 40, batch_size=32, verbose = 0)
        pred_cnn <- final_model %>% predict(xtest_cnn, verbose=0)
        df_cnn <- data.frame(Repetition=rep_id, Fold=i, Model="CNN", Cor=cor(pred_cnn[,1], phenoTest, use="complete.obs"), cnn_filters=best_cnn_params$filters, cnn_kernel_size=best_cnn_params$kernel_size)
        rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_cnn)
      }, error = function(e) { cat(paste0("  ERROR in CNN model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
    }
    return(dplyr::bind_rows(rep_results_list))
  }
  
  # Execute the CV repetitions in parallel using the pre-configured cluster.
  cv_results_df <- future_map_dfr(.x = 1:cv_repeats, .f = run_one_repetition, .options = furrr_options(seed = TRUE))
  
  # Clean up temporary BGLR files
  unlink(list.files(pattern = "rep_.*.dat"))
  
  # --- Summarize CV results to find the best model ---
  cv_summary_stats <- cv_results_df %>%
    filter(!is.na(Cor)) %>%
    group_by(Model) %>%
    summarise(Mean_Cor = mean(Cor, na.rm = TRUE), SD_Cor = sd(Cor, na.rm = TRUE), .groups = 'drop') %>%
    arrange(desc(Mean_Cor))
  
  cat("\n[INFO] Cross-validation within training population finished.\n")
  cat("[INFO] Performance of all models in the internal CV:\n")
  print(cv_summary_stats)
  
  best_model_name <- cv_summary_stats$Model[1]
  cat(paste0("\n[INFO] Best model identified for this scenario: '", best_model_name, "' with mean CV accuracy of ", round(cv_summary_stats$Mean_Cor[1], 4), ".\n"))
  
  
  # --- Step 2: Train BEST model and perform SAMPLING VALIDATION on prediction population ---
  cat(paste0("\n[STEP 2] Training final '", best_model_name, "' model on all '", train_pop_name, "' data...\n"))
  final_train_geno <- genotypeMatrix[train_ids, ]
  final_train_pheno <- phenotypeVector[train_ids]
  
  prediction_accuracies <- numeric(pred_samples)
  
  cat(paste0("[INFO] Starting sampling validation on '", pred_pop_name, "' (", pred_samples, " samples of ", pred_sample_frac * 100, "%)...\n"))
  
  # This loop is sequential as each iteration is fast.
  for (iter in 1:pred_samples) {
    if (iter %% 10 == 0) cat(paste0("  - Prediction Sample ", iter, "/", pred_samples, "...\n"))
    set.seed(42 + iter)
    # Randomly sample a fraction of the prediction population for this iteration
    sample_pred_ids <- sample(pred_ids, size = floor(length(pred_ids) * pred_sample_frac))
    
    final_pred_geno_sample <- genotypeMatrix[sample_pred_ids, ]
    final_pred_pheno_sample <- phenotypeVector[sample_pred_ids]
    
    predictions_sample <- NULL
    
    # --- PREDICTION LOGIC for the identified BEST model ---
    tryCatch({
      if (best_model_name %in% c("Ridge", "LASSO", "Elastic Net")) {
        get_mode <- function(v) { v <- v[!is.na(v)]; uniqv <- unique(v); uniqv[which.max(tabulate(match(v, uniqv)))] }
        alpha_val <- case_when(
          best_model_name == "Ridge" ~ 0,
          best_model_name == "LASSO" ~ 1,
          TRUE ~ get_mode(cv_results_df %>% filter(Model=="Elastic Net") %>% pull(alpha))
        )
        final_model_trained <- cv.glmnet(final_train_geno, final_train_pheno, alpha = alpha_val, family = "gaussian")
        predictions_sample <- predict(final_model_trained, newx = final_pred_geno_sample, s = "lambda.min")[, 1]
        
      } else if (best_model_name %in% c("BRR", "BayesA", "BayesB", "BayesC", "Bayes G-BLUP")) {
        # BGLR requires a single phenotype vector with NAs for the individuals to be predicted
        ids_for_bglr <- c(train_ids, sample_pred_ids)
        y_bglr <- phenotypeVector[ids_for_bglr]
        y_bglr[sample_pred_ids] <- NA
        
        model_name_bglr <- if(best_model_name == "Bayes G-BLUP") "RKHS" else best_model_name
        
        ETA_final <- if(model_name_bglr != "RKHS") {
          list(list(X = genotypeMatrix[ids_for_bglr,], model = model_name_bglr))
        } else {
          list(list(K = G_full[ids_for_bglr, ids_for_bglr], model = "RKHS"))
        }
        
        fit_bglr_final <- BGLR(y = y_bglr, ETA = ETA_final, nIter = 10000, burnIn = 2500, verbose = FALSE)
        # Extract predictions for the validation set individuals
        pred_indices_in_bglr <- match(sample_pred_ids, names(y_bglr))
        predictions_sample <- fit_bglr_final$yHat[pred_indices_in_bglr]
        
      } else if (best_model_name %in% c("GBLUP", "AD-GBLUP", "ssGBLUP")) {
        # sommer also uses a single phenotype vector with NAs for prediction
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
        
        pred_D_arg <- if(best_model_name == "AD-GBLUP") "ID_A" else "ID"
        final_pred_table <- predict(final_fit, D = pred_D_arg)
        predictions_sample <- final_pred_table$pvals[sample_pred_ids, "predicted.value"]
        
      } else if (best_model_name %in% c("MLP", "CNN")) {
        get_mode <- function(v) { v <- v[!is.na(v)]; if(length(v)==0) return(NA); uniqv <- unique(v); uniqv[which.max(tabulate(match(v, uniqv)))] }
        
        train_mean <- colMeans(final_train_geno); train_sd <- apply(final_train_geno, 2, sd); train_sd[train_sd == 0] <- 1
        final_train_geno_scaled <- scale(final_train_geno, center = train_mean, scale = train_sd)
        
        final_model_trained <- if(best_model_name == "MLP") {
          params <- cv_results_df %>% filter(Model=="MLP") %>% summarise(neurons=get_mode(mlp_neurons), dropout=get_mode(mlp_dropout))
          model <- keras_model_sequential(input_shape = c(numMarkers)) %>%
            layer_dense(units = params$neurons, activation = "relu") %>%
            layer_dropout(rate = params$dropout) %>%
            layer_dense(units = round(params$neurons/2), activation="relu") %>%
            layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
          model %>% fit(final_train_geno_scaled, final_train_pheno, epochs = 50, batch_size=32, verbose = 0)
          model
        } else { # CNN
          params <- cv_results_df %>% filter(Model=="CNN") %>% summarise(filters=get_mode(cnn_filters), kernel_size=get_mode(cnn_kernel_size))
          train_cnn <- array(final_train_geno_scaled, dim = c(nrow(final_train_geno_scaled), numMarkers, 1))
          model <- keras_model_sequential(input_shape = c(numMarkers, 1)) %>%
            layer_conv_1d(filters = params$filters, kernel_size = params$kernel_size, activation = "relu") %>%
            layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
            layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = "rmsprop")
          model %>% fit(train_cnn, final_train_pheno, epochs = 50, batch_size=32, verbose = 0)
          model
        }
        pred_geno_scaled <- scale(final_pred_geno_sample, center = train_mean, scale = train_sd)
        pred_input <- if(best_model_name == "MLP") pred_geno_scaled else array(pred_geno_scaled, dim=c(nrow(pred_geno_scaled), numMarkers, 1))
        predictions_sample <- final_model_trained %>% predict(pred_input, verbose=0)
        predictions_sample <- predictions_sample[,1]
      }
    }, error = function(e) {
      cat(paste0("   !-> Prediction failed for model '", best_model_name, "' in iteration ", iter, ": ", e$message, "\n"))
      predictions_sample <<- NULL # Ensure it is null on failure
    })
    
    if (!is.null(predictions_sample)) {
      prediction_accuracies[iter] <- cor(predictions_sample, final_pred_pheno_sample, use = "complete.obs")
    } else {
      prediction_accuracies[iter] <- NA
    }
  }
  
  avg_final_pred_accuracy <- mean(prediction_accuracies, na.rm = TRUE)
  sd_final_pred_accuracy <- sd(prediction_accuracies, na.rm = TRUE)
  
  cat(paste0("\n------------------------------------------------------------------\n"))
  cat(paste0(">>> Avg. Cross-Pop Accuracy: ", round(avg_final_pred_accuracy, 4), " \u00B1 ", round(sd_final_pred_accuracy, 4), "\n"))
  cat(paste0("    (Trained on ", train_pop_name, ", Predicted on ", pred_pop_name, ")\n"))
  cat(paste0("------------------------------------------------------------------\n"))
  
  # Return a summary tibble for this entire scenario
  return(tibble::tibble(
    Training_Population = train_pop_name,
    Prediction_Population = pred_pop_name,
    Best_Model_in_CV = best_model_name,
    CV_Accuracy_of_Best_Model = cv_summary_stats$Mean_Cor[1],
    Avg_Cross_Pop_Accuracy = avg_final_pred_accuracy,
    SD_Cross_Pop_Accuracy = sd_final_pred_accuracy,
    CV_Summary = list(cv_summary_stats) # Embed the detailed CV results
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
  pred_pop_name = "HP"
)

# --- Run Scenario B: Train on HP, Predict on NH/HN ---
result_B <- perform_cross_population_prediction(
  train_ids = pop_HP_ids,
  pred_ids = pop_NH_HN_ids,
  train_pop_name = "HP",
  pred_pop_name = "NH_HN"
)


# =====================================================================================
# Part 5: Final Summary and Cleanup
# =====================================================================================

# --- IMPORTANT: Stop the parallel cluster ---
cat("\n[PARALLEL] Stopping the parallel cluster...\n")
if(exists("cl")) parallel::stopCluster(cl)
plan(sequential) # Return to sequential processing

# --- Combine results into a final printable summary table ---
final_summary_printable <- bind_rows(
  result_A %>% select(-CV_Summary),
  result_B %>% select(-CV_Summary)
)

cat("\n\n====================================================================\n")
cat("                     FINAL CROSS-POPULATION PREDICTION SUMMARY\n")
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

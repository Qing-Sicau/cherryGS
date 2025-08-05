# =====================================================================================
#
#   Script 2: SNP Density Impact Analysis (Revised and Synchronized)
#
# Description:
#   This script evaluates how genomic prediction accuracy changes when using different
#   densities of SNP markers. It iterates through a list of specified densities,
#   and for each one, it runs a full, robust analysis pipeline using multiple
#   prediction models. All model implementations are synchronized with the
#   'full_pop.R' benchmark script.
#
# Last Modified: Aug 04, 2025
#
# =====================================================================================


# =====================================================================================
# Part 0: Environment Setup
# =====================================================================================
# Set environment variables to prevent over-threading issues with underlying math libraries
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)
Sys.setenv(OMP_NUM_THREADS = 1)

cat("[SETUP] Loading required R packages...\n")
required_packages <- c(
  "tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
  "tensorflow", "keras", "tfdatasets", "caret", "cowplot", "reticulate",
  "future", "furrr", "sommer", "Cairo"
)
suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})


# =====================================================================================
# Part 0.1: Find Python Path Before Parallelization (CRITICAL STEP)
# =====================================================================================
# This step finds the Python executable in the main R session. This exact path
# will then be sent to each parallel worker to ensure Keras/TensorFlow work correctly.
cat("[SETUP] Finding Python executable from the 'reseq' conda environment...\n")
use_condaenv("reseq", required = TRUE) 

python_exe_path <- reticulate::py_config()$python
cat(paste("[SETUP] Found python executable to be used by workers:", python_exe_path, "\n"))


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
cat("[SETUP] Environment setup complete.\n")


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

# --- Data Alignment and Cleaning ---
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
cat(paste("[DATA] Found ", length(common_individuals), " individuals with both genotype and phenotype data.\n"))

genotypeMatrix_full <- genotypeMatrix_raw[common_individuals, ]

# Ensure the specified trait exists
if (!TRAIT_OF_INTEREST %in% colnames(pheno_df)) {
  stop(paste("Error: The specified trait '", TRAIT_OF_INTEREST, "' was not found in the phenotype file."))
}
phenotype_full_df <- pheno_df[common_individuals, , drop = FALSE] %>%
  dplyr::select(all_of(TRAIT_OF_INTEREST))

phenotypeVector <- phenotype_full_df[[TRAIT_OF_INTEREST]]
names(phenotypeVector) <- rownames(phenotype_full_df)

# --- Handle Missing Genotype Values ---
cat("[DATA] Handling missing genotype values using mean imputation...\n")
impute_mean <- function(x) {
  mean_val <- mean(x, na.rm = TRUE)
  if (is.nan(mean_val)) mean_val <- 0 # Handle cases where a marker is all NA
  x[is.na(x)] <- round(mean_val)
  return(x)
}
genotypeMatrix_full <- apply(genotypeMatrix_full, 2, impute_mean)
if(sum(is.na(genotypeMatrix_full)) > 0) {
  warning("Missing values still exist after imputation. Please check your data.")
}

numIndividuals_full <- nrow(genotypeMatrix_full)
numMarkers_full <- ncol(genotypeMatrix_full)
cat(paste("[DATA] Full dataset prepared: ", numIndividuals_full, " individuals, ", numMarkers_full, " SNP markers.\n"))


# =====================================================================================
# Part 0.75: Define Helper Functions & Build Pedigree
# =====================================================================================
cat("\n[PREP] Defining helper functions and building pedigree...\n")

# This script is required for the dominance matrix calculation.
source("get_DomRel_matrix.R")

# Function to calculate H-inverse for ssGBLUP, synchronized with full_pop.R
doH_inverse <- function(pedigreeRelationshipMatrix, grmForGenotyped) {
  genotypedIndicesInPedigree <- match(rownames(grmForGenotyped), rownames(pedigreeRelationshipMatrix))
  grmInverse <- solve(grmForGenotyped)
  A22 <- pedigreeRelationshipMatrix[genotypedIndicesInPedigree, genotypedIndicesInPedigree]
  pedigreeRelationshipInverseForGenotyped <- solve(A22)
  pedigreeRelationshipInverse <- solve(pedigreeRelationshipMatrix)
  
  hMatrixInverse <- pedigreeRelationshipInverse
  hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] <-
    hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] + grmInverse - pedigreeRelationshipInverseForGenotyped
  
  attr(hMatrixInverse, 'inverse') <- TRUE # Critical for sommer package
  return(hMatrixInverse)
}

# --- Build Pedigree (A-matrix) - This is done only ONCE for the full population ---
parents <- c("HF1", "NZH2", "PJHH")
all_ped_ids <- unique(c(parents, common_individuals))
ped_df <- data.frame(ID = all_ped_ids, Sire = 0, Dam = 0, stringsAsFactors = FALSE)
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) {
    ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) {
    ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "PJHH"
  }
}
A_full <- Amatrix(ped_df, ploidy = 4)
cat("[PREP] A-matrix (pedigree relationship) built successfully.\n")


# =====================================================================================
# Part 1: Main Analysis Loop for SNP Density
# =====================================================================================
# --- User Configuration for the Analysis ---
NUM_REPEATS   <- 20
NUM_CORES     <- 20 # <-- !!! SET THE NUMBER OF CPU CORES TO USE !!!
SNP_DENSITIES <- c(1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01)

# --- Setup Robust Parallel Backend ---
cat(paste("\n[SETUP] Setting up robust parallel backend to use", NUM_CORES, "cores...\n"))
cl <- parallel::makeCluster(NUM_CORES)
parallel::clusterExport(cl, "python_exe_path")
parallel::clusterEvalQ(cl, {
  library(reticulate)
  use_python(python_exe_path, required = TRUE)
  library(keras)
})
plan(cluster, workers = cl)
cat("[SETUP] Parallel cluster is ready.\n")

set.seed(42) # Single seed for reproducibility of the entire analysis
all_results_list <- list()

# --- Outer "SNP Density" Loop ---
for (density in SNP_DENSITIES) {
  num_snps_to_sample <- round(numMarkers_full * density)
  cat(paste0("\n\n==================================================================\n"))
  cat(paste0("  Starting analysis for SNP Density: ", density * 100, "% (", num_snps_to_sample, " markers)\n"))
  cat(paste0("==================================================================\n"))
  
  # --- SNP Sub-sampling ---
  set.seed(123) # Use a fixed seed for sampling to ensure higher densities are supersets
  sampled_snp_indices <- sample(1:numMarkers_full, num_snps_to_sample)
  genotypeMatrix_sub <- genotypeMatrix_full[, sampled_snp_indices, drop=FALSE]
  
  # --- Prepare relationship matrices FOR THIS SNP SUBSET ---
  cat("[PREP] Calculating relationship matrices for the current SNP subset...\n")
  
  # Remove non-polymorphic markers from the subset *before* creating matrices
  sd_vals <- apply(genotypeMatrix_sub, 2, sd)
  poly_indices <- which(sd_vals > 1e-6)
  genotypeMatrix <- genotypeMatrix_sub[, poly_indices, drop=FALSE]
  numMarkers <- ncol(genotypeMatrix) # Update marker count for this density level
  
  if (numMarkers == 0) {
    cat("[WARNING] No polymorphic markers found for this density. Skipping.\n")
    next
  }
  
  # Additive Matrix (G)
  G <- Gmatrix(genotypeMatrix, method = "VanRaden", ploidy = 4)
  G <- G + diag(nrow(G)) * 1e-4
  
  # Dominance Matrix (D)
  D_raw <- get_DomRel(genotypeMatrix, ploidy = 4)
  Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw))
  D_centered <- Ic %*% D_raw %*% Ic
  D <- D_centered + diag(nrow(D_centered)) * 1e-4
  rownames(D) <- rownames(genotypeMatrix); colnames(D) <- rownames(genotypeMatrix)
  
  # H-inverse Matrix (for ssGBLUP)
  Hinv <- doH_inverse(A_full, G)
  
  # --- Define the function that runs ONE full repetition for the current density ---
  run_one_repetition <- function(rep_id) {
    rep_results_list <- list()
    
    # Load all required libraries on the worker node
    suppressPackageStartupMessages({
      library(tidyverse); library(BGLR); library(Matrix); library(glmnet);
      library(keras); library(caret); library(sommer)
    })
    
    # Helper function to standardize result data frames
    all_possible_columns <- c(
      "Repetition", "Fold", "Model", "Cor", "SNP_Density", "Num_SNPs",
      "alpha", "varA", "varD", "mlp_neurons", "mlp_dropout", "cnn_filters", "cnn_kernel_size"
    )
    standardize_df <- function(df) {
      missing_cols <- setdiff(all_possible_columns, names(df))
      if (length(missing_cols) > 0) df[missing_cols] <- NA
      return(df[, all_possible_columns])
    }
    
    cat(paste0("  [CV] Repetition ", rep_id, "/", NUM_REPEATS, "...\n"))
    folds <- createFolds(phenotypeVector, k = 5, list = TRUE, returnTrain = FALSE)
    
    for (i in 1:length(folds)) {
      test_indices <- folds[[i]]
      test_ids <- names(phenotypeVector[test_indices])
      train_indices <- setdiff(1:length(phenotypeVector), test_indices)
      
      phenoTrain <- phenotypeVector[train_indices]
      phenoTest <- phenotypeVector[test_ids]
      genoTrain <- genotypeMatrix[train_indices, ]
      genoTest <- genotypeMatrix[test_ids, ]
      
      pheno_with_NAs <- phenotypeVector; pheno_with_NAs[test_indices] <- NA
      
      # --- 1. glmnet Family ---
      tryCatch({
        cv_ridge <- cv.glmnet(genoTrain, phenoTrain, alpha = 0, family="gaussian")
        pred_ridge <- predict(cv_ridge, newx = genoTest, s = "lambda.min")[, 1]
        df_ridge <- data.frame(Repetition=rep_id, Fold=i, Model="Ridge", Cor=cor(pred_ridge, phenoTest, use="complete.obs"), alpha=0)
        rep_results_list[[length(rep_results_list) + 1]] <- df_ridge
        
        # ... (LASSO and Elastic Net logic identical to full_pop.R)
        cv_lasso <- cv.glmnet(genoTrain, phenoTrain, alpha = 1, family="gaussian")
        pred_lasso <- predict(cv_lasso, newx = genoTest, s = "lambda.min")[, 1]
        df_lasso <- data.frame(Repetition = rep_id, Fold = i, Model = "LASSO", Cor = cor(pred_lasso, phenoTest, use = "complete.obs"), alpha = 1)
        rep_results_list[[length(rep_results_list) + 1]] <- df_lasso
        
        best_alpha <- NA; best_lambda <- NA; best_mse <- Inf
        for (a in seq(0, 1, by = 0.2)) {
          cv_fit <- cv.glmnet(genoTrain, phenoTrain, alpha = a, family="gaussian")
          current_mse <- min(cv_fit$cvm, na.rm = TRUE)
          if (is.finite(current_mse) && current_mse < best_mse) { best_mse <- current_mse; best_alpha <- a; best_lambda <- cv_fit$lambda.min }
        }
        fit_en <- glmnet(genoTrain, phenoTrain, alpha = best_alpha, lambda = best_lambda, family="gaussian")
        pred_en <- predict(fit_en, newx = genoTest)[, 1]
        df_en <- data.frame(Repetition = rep_id, Fold = i, Model = "Elastic Net", Cor = cor(pred_en, phenoTest, use = "complete.obs"), alpha = best_alpha)
        rep_results_list[[length(rep_results_list) + 1]] <- df_en
        
      }, error = function(e){ cat(paste0("  - ERROR in glmnet, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 2. Bayesian Family (BGLR) - With parallel-safe file handling ---
      tryCatch({
        models_bglr <- list(
          BRR = list(list(X = genotypeMatrix, model = "BRR")),
          BayesA = list(list(X = genotypeMatrix, model = "BayesA")),
          BayesB = list(list(X = genotypeMatrix, model = "BayesB")),
          BayesC = list(list(X = genotypeMatrix, model = "BayesC")),
          `Bayes G-BLUP` = list(list(K = G, model = "RKHS"))
        )
        for(m_name in names(models_bglr)){
          fit_bglr <- BGLR(y = pheno_with_NAs, ETA = models_bglr[[m_name]], nIter = 10000, burnIn = 2500, verbose = FALSE, saveAt=paste0("density_",density,"_rep_",rep_id,"_fold_",i,"_"))
          pred_bglr <- fit_bglr$yHat[test_indices]
          df_bglr <- data.frame(Repetition = rep_id, Fold = i, Model = m_name, Cor = cor(pred_bglr, phenoTest, use = "complete.obs"))
          rep_results_list[[length(rep_results_list) + 1]] <- df_bglr
        }
      }, error = function(e){ cat(paste0("  - ERROR in BGLR, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 3. sommer Family (GBLUP, AD-GBLUP, ssGBLUP) - Using robust predict() method ---
      tryCatch({ # GBLUP
        data_sommer <- data.frame(ID = names(phenotypeVector), y = pheno_with_NAs)
        data_sommer$ID <- factor(data_sommer$ID, levels = rownames(G))
        fit_gblup <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=G), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
        pred_table <- predict(fit_gblup, D = "ID")
        pred_gblup <- pred_table$pvals[test_ids, "predicted.value"]
        df_gblup <- data.frame(Repetition=rep_id, Fold=i, Model="GBLUP", Cor=cor(pred_gblup, phenoTest, use="complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- df_gblup
      }, error = function(e){ cat(paste0("  - ERROR in GBLUP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      tryCatch({ # AD-GBLUP
        data_sommer <- data.frame(ID = names(phenotypeVector), y = pheno_with_NAs)
        data_sommer$ID_A <- factor(data_sommer$ID, levels = rownames(G))
        data_sommer$ID_D <- factor(data_sommer$ID, levels = rownames(D))
        fit_ad <- mmes(fixed=y~1, random=~vsm(ism(ID_A), Gu=G) + vsm(ism(ID_D), Gu=D), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
        pred_table <- predict(fit_ad, D = "ID_A")
        pred_ad <- pred_table$pvals[test_ids, "predicted.value"]
        var_a <- fit_ad$sigma[[1]]; if (length(var_a) == 0) var_a <- NA
        var_d <- fit_ad$sigma[[2]]; if (length(var_d) == 0) var_d <- NA
        df_ad <- data.frame(Repetition=rep_id, Fold=i, Model="AD-GBLUP", Cor=cor(pred_ad, phenoTest, use="complete.obs"), varA=var_a, varD=var_d)
        rep_results_list[[length(rep_results_list) + 1]] <- df_ad
      }, error = function(e){ cat(paste0("  - ERROR in AD-GBLUP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      tryCatch({ # ssGBLUP
        pheno_ssgblup <- rep(NA, nrow(A_full)); names(pheno_ssgblup) <- rownames(A_full)
        pheno_ssgblup[names(pheno_with_NAs)] <- pheno_with_NAs
        data_sommer <- data.frame(ID = names(pheno_ssgblup), y = pheno_ssgblup)
        data_sommer$ID <- factor(data_sommer$ID, levels = rownames(Hinv))
        fit_ss <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=Hinv), rcov=~units, data=data_sommer, naMethodY="include", verbose=F, henderson=T)
        pred_table <- predict(fit_ss, D = "ID")
        pred_ss <- pred_table$pvals[test_ids, "predicted.value"]
        df_ss <- data.frame(Repetition=rep_id, Fold=i, Model="ssGBLUP", Cor=cor(pred_ss, phenoTest, use="complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- df_ss
      }, error = function(e){ cat(paste0("  - ERROR in ssGBLUP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      # --- 4. Deep Learning Models - Synchronized Architectures ---
      train_mean <- colMeans(genoTrain); train_sd <- apply(genoTrain, 2, sd); train_sd[train_sd == 0] <- 1
      genoTrain_scaled <- scale(genoTrain, center = train_mean, scale = train_sd)
      genoTest_scaled <- scale(genoTest, center = train_mean, scale = train_sd)
      
      tryCatch({ # MLP - Corrected Architecture
        mlp_param_grid <- expand.grid(dropout_rate = c(0.3, 0.5), neurons = c(64, 128))
        best_val_loss <- Inf; best_mlp_params <- list(dropout_rate=NA, neurons=NA)
        for(j in 1:nrow(mlp_param_grid)) {
          params <- mlp_param_grid[j, ]
          model <- keras_model_sequential() %>%
            layer_dense(units = params$neurons, activation = "relu", input_shape = numMarkers) %>%
            layer_dropout(rate = params$dropout_rate) %>%
            layer_dense(units = round(params$neurons/2), activation="relu") %>%
            layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
          hist <- model %>% fit(genoTrain_scaled, phenoTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
          val_loss <- min(hist$metrics$val_loss, na.rm=T)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
        }
        final_model <- keras_model_sequential(input_shape = c(numMarkers)) %>%
          layer_dense(units = best_mlp_params$neurons, activation = "relu") %>%
          layer_dropout(rate = best_mlp_params$dropout_rate) %>%
          layer_dense(units = round(best_mlp_params$neurons/2), activation="relu") %>%
          layer_dense(units = 1)
        final_model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
        final_model %>% fit(genoTrain_scaled, phenoTrain, epochs = 40, batch_size=32, verbose = 0)
        pred_mlp <- final_model %>% predict(genoTest_scaled, verbose=0)
        df_mlp <- data.frame(Repetition=rep_id, Fold=i, Model="MLP", Cor=cor(pred_mlp[,1], phenoTest, use="complete.obs"), mlp_neurons = best_mlp_params$neurons, mlp_dropout = best_mlp_params$dropout_rate)
        rep_results_list[[length(rep_results_list) + 1]] <- df_mlp
      }, error = function(e){ cat(paste0("  - ERROR in MLP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
      
      tryCatch({ # CNN - Corrected Architecture
        xtrain_cnn <- array(genoTrain_scaled, dim = c(nrow(genoTrain_scaled), numMarkers, 1))
        xtest_cnn <- array(genoTest_scaled, dim = c(nrow(genoTest_scaled), numMarkers, 1))
        cnn_param_grid <- expand.grid(filters = c(32, 64), kernel_size = c(5, 10))
        best_val_loss <- Inf; best_cnn_params <- list(filters=NA, kernel_size=NA)
        for(j in 1:nrow(cnn_param_grid)) {
          params <- cnn_param_grid[j, ]
          model <- keras_model_sequential() %>%
            layer_conv_1d(filters = params$filters, kernel_size = params$kernel_size, activation = "relu", input_shape = c(numMarkers, 1)) %>%
            layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
            layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
          model %>% compile(loss = "mse", optimizer = "rmsprop")
          hist <- model %>% fit(xtrain_cnn, phenoTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
          val_loss <- min(hist$metrics$val_loss, na.rm=T)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_cnn_params <- params }
        }
        final_model <- keras_model_sequential(input_shape = c(numMarkers, 1)) %>%
          layer_conv_1d(filters = best_cnn_params$filters, kernel_size = best_cnn_params$kernel_size, activation = "relu") %>%
          layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
          layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
        final_model %>% compile(loss = "mse", optimizer = "rmsprop")
        final_model %>% fit(xtrain_cnn, phenoTrain, epochs = 40, batch_size=32, verbose = 0)
        pred_cnn <- final_model %>% predict(xtest_cnn, verbose=0)
        df_cnn <- data.frame(Repetition=rep_id, Fold=i, Model="CNN", Cor=cor(pred_cnn[,1], phenoTest, use="complete.obs"), cnn_filters=best_cnn_params$filters, cnn_kernel_size=best_cnn_params$kernel_size)
        rep_results_list[[length(rep_results_list) + 1]] <- df_cnn
      }, error = function(e){ cat(paste0("  - ERROR in CNN, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
    }
    # Add density information and standardize columns before returning
    results_for_rep <- dplyr::bind_rows(rep_results_list)
    results_for_rep$SNP_Density <- density
    results_for_rep$Num_SNPs <- num_snps_to_sample
    return(standardize_df(results_for_rep))
  }
  
  # --- Run the repetitions in parallel for the current density ---
  density_results_df <- future_map_dfr(
    .x = 1:NUM_REPEATS,
    .f = run_one_repetition,
    .options = furrr_options(seed = TRUE)
  )
  
  all_results_list[[as.character(density)]] <- density_results_df
}
# --- End of outer "SNP Density" Loop ---


# =====================================================================================
# Part 2: Final Cleanup, Result Summarization, and Visualization
# =====================================================================================

# --- IMPORTANT: Stop the parallel cluster to free up resources ---
cat("\n[CLEANUP] Stopping the parallel cluster...\n")
parallel::stopCluster(cl)
plan(sequential) # Return to sequential processing

# Clean up BGLR files from all runs
unlink(list.files(pattern = "density_.*.dat"))

results_df <- dplyr::bind_rows(all_results_list)
cat("\n[COMPLETE] Full analysis across all SNP densities finished!\n")

# --- Calculate Summary Statistics ---
cat("\n[RESULTS] Summarizing and plotting results...\n")
summary_stats <- results_df %>%
  dplyr::filter(!is.na(Cor)) %>%
  dplyr::group_by(Model, SNP_Density, Num_SNPs) %>%
  dplyr::summarise(
    Mean_Cor = mean(Cor, na.rm=TRUE),
    SD_Cor = sd(Cor, na.rm=TRUE),
    .groups = 'drop'
  ) %>%
  dplyr::arrange(desc(SNP_Density), desc(Mean_Cor))

print("Summary of prediction accuracies across different SNP densities:")
print(as.data.frame(summary_stats))

# --- Plot: Model Performance vs. SNP Density (Corrected Logic) ---
# Calculate the actual number of markers sampled for each density level from the summary table
# This ensures the breaks are accurate even if some markers were removed (e.g., non-polymorphic)
snp_break_points <- summary_stats %>%
  dplyr::select(SNP_Density, Num_SNPs) %>%
  dplyr::distinct() %>%
  dplyr::arrange(Num_SNPs)

plot_density_decay <- ggplot(summary_stats, aes(x = Num_SNPs, y = Mean_Cor, group = Model, color = Model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5, alpha = 0.8) +
  # Corrected geom_errorbar: width is now smaller and absolute, preventing overly large bars on the log scale
  geom_errorbar(aes(ymin = Mean_Cor - SD_Cor, ymax = Mean_Cor + SD_Cor), width = 0, alpha = 0.6) + # Set width to 0 to make it a vertical line
  geom_point(size = 2.5, alpha = 0.8) + # Re-plot points to be on top of error bars
  
  # Corrected scale_x_log10:
  # 1. Let ggplot handle the minor breaks automatically for a clean log scale.
  # 2. Set the major breaks to be exactly at our tested SNP numbers for clarity.
  # 3. Use the density percentages as labels for better intuition.
  scale_x_log10(
    breaks = snp_break_points$Num_SNPs,
    labels = paste0(snp_break_points$SNP_Density * 100, "%")
  ) +
  
  # Add annotation for number of SNPs below the percentage for full context
  annotate("text", x = snp_break_points$Num_SNPs, y = min(summary_stats$Mean_Cor - summary_stats$SD_Cor, na.rm = TRUE) * 0.95, 
           label = paste0("(", scales::comma(snp_break_points$Num_SNPs), ")"), size = 2.8, angle = 45, hjust = 1) +
  
  labs(
    title = paste("Prediction Accuracy vs. SNP Marker Density for", TRAIT_OF_INTEREST),
    subtitle = paste0("Average of ", NUM_REPEATS, " Repeats (5-Fold Cross-Validation)"),
    x = "SNP Density (% of Total Markers)",
    y = "Mean Prediction Accuracy (Pearson's r)",
    color = "Prediction Model"
  ) +
  theme_publication() +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1) # Rotate labels for better fit
  ) +
  coord_cartesian(clip = "off") # Allows annotations outside the main plot area


# --- Create output directory and Save Results ---
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)
print(plot_density_decay)
ggsave(paste0("plots/GS_SNPDensity_Comparison_", TRAIT_OF_INTEREST, ".png"), plot = plot_density_decay, width = 10, height = 8, dpi = 300, device = "png", type = "cairo")
write.csv(results_df, paste0("GS_All_Density_Results_", TRAIT_OF_INTEREST, ".csv"), row.names = FALSE)
write.csv(summary_stats, paste0("GS_Density_Summary_", TRAIT_OF_INTEREST, ".csv"), row.names = FALSE)
cat(paste0("\n[COMPLETE] Analysis finished. SNP density plot and summary data have been saved.\n"))

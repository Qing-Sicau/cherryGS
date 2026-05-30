# =====================================================================================
# Part 0: Environment Setup
# =====================================================================================

# --- Define a logging function to prepend timestamps ---
log_message <- function(message) {
  cat(paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", message, "\n"))
}

log_message("Starting analysis pipeline.")

# Set environment variables to prevent over-threading issues
log_message("[SETUP] Setting environment variables for thread management (OPENBLAS, MKL, OMP).")
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)
Sys.setenv(OMP_NUM_THREADS = 1)

log_message("[SETUP] Loading required R packages...")
required_packages <- c(
  "tidyverse", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
  "tensorflow", "keras3", "tfdatasets", "caret", "cowplot", "reticulate",
  "future", "furrr", "sommer", "Cairo"
)
suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})
log_message("[SETUP] All required packages loaded successfully.")


# =====================================================================================
# Part 2: Find Python Path Before Parallelization (CRITICAL STEP)
# =====================================================================================
log_message("[SETUP] Finding Python executable from the 'poly_map' conda environment...")
use_condaenv("poly_map", required = TRUE)
python_exe_path <- reticulate::py_config()$python
log_message(paste("[SETUP] Found python executable to be used by workers:", python_exe_path))


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
log_message("[SETUP] Creating output directories './plots' and './results' if they don't exist.")
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)
if (!dir.exists("./results")) dir.create("./results", recursive = TRUE)
log_message("[SETUP] Environment setup complete.")


# =====================================================================================
# Part 3: Load, Clean, and Preprocess Data
# =====================================================================================
log_message("\n[DATA] Starting data loading and preprocessing...")

# --- User Configuration ---
GENOTYPE_FILE  <- "genotype.dosages.tsv"
PHENOTYPE_FILE <- "phenotype_BLUPs.csv"

# --- Load Genotype Data ---
log_message(paste("[DATA] Loading genotype data from", GENOTYPE_FILE, "..."))
dosage_raw <- read.csv(GENOTYPE_FILE, check.names = FALSE, sep = "\t", header = TRUE)
marker_ids <- paste0(dosage_raw$CHROM, ":", dosage_raw$POS)
dosage_only <- dosage_raw[, 5:ncol(dosage_raw)]
# OPTIMIZATION: Use as.matrix for potential speed up over apply
dosage_only_matrix <- as.matrix(dosage_only)
storage.mode(dosage_only_matrix) <- "numeric"
genotypeMatrix_raw <- t(dosage_only_matrix)
rownames(genotypeMatrix_raw) <- colnames(dosage_only)
colnames(genotypeMatrix_raw) <- marker_ids
log_message(paste0("[DATA] Loaded genotype data with ", nrow(genotypeMatrix_raw), " individuals and ", ncol(genotypeMatrix_raw), " markers."))

# --- Load Phenotype Data ---
log_message(paste("[DATA] Loading phenotype data from", PHENOTYPE_FILE, "..."))
pheno_raw <- read.csv(PHENOTYPE_FILE, row.names = 1, check.names = FALSE, sep = ",", header = TRUE)
pheno_df <- as.data.frame(pheno_raw)
ALL_TRAITS_IN_FILE <- colnames(pheno_df)
log_message(paste0("[DATA] Found ", length(ALL_TRAITS_IN_FILE), " traits in phenotype file: ", paste(ALL_TRAITS_IN_FILE, collapse=", ")))

# --- Data Alignment and Cleaning ---
log_message("[DATA] Aligning genotype and phenotype data by common individuals...")
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
log_message(paste("[DATA] Found ", length(common_individuals), " individuals with both genotype and phenotype data."))

genotypeMatrix <- genotypeMatrix_raw[common_individuals, ]
pheno_df_aligned <- pheno_df[common_individuals, , drop = FALSE]
log_message("[DATA] Data alignment complete.")

# --- Handle Missing Genotype Values (Mean Imputation) ---
log_message("[DATA] Handling missing genotype values using mean imputation...")
col_means <- colMeans(genotypeMatrix, na.rm = TRUE)
col_means[is.nan(col_means)] <- 0 # Handle columns with all NAs
missing_indices <- which(is.na(genotypeMatrix), arr.ind = TRUE)
genotypeMatrix[missing_indices] <- round(col_means[missing_indices[, 2]])

if(sum(is.na(genotypeMatrix)) > 0) {
  warning("Missing values still exist after imputation. Please check your data.")
}
numIndividuals <- nrow(genotypeMatrix)
numMarkers <- ncol(genotypeMatrix)
log_message(paste("[DATA] Data preparation complete: ", numIndividuals, " individuals, ", numMarkers, " SNP markers."))


# =====================================================================================
# Part 4: Pre-calculate Relationship Matrices and Pedigree (更正后)
# =====================================================================================
log_message("\n[PREP] Pre-calculating relationship matrices and pedigree...")

source("get_DomRel_matrix.R")

# --- Additive (G) and Dominance (D) Matrices ---
G <- Gmatrix(genotypeMatrix, method = "VanRaden", ploidy = 4)
G <- G + diag(nrow(G)) * 1e-4
log_message("[PREP] G-matrix (additive) built successfully.")

D_raw <- get_DomRel(genotypeMatrix, ploidy = 4)
Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw))
D_centered <- Ic %*% D_raw %*% Ic
D <- D_centered + diag(nrow(D_centered)) * 1e-4
rownames(D) <- rownames(genotypeMatrix); colnames(D) <- rownames(genotypeMatrix)
log_message("[PREP] D-matrix (dominance) built successfully.")

# --- Pedigree (A) and H-inverse Matrices for ssGBLUP ---
# This is the robust custom function to calculate H-inverse directly
doH_1_inverse <- function(pedigreeRelationshipMatrix, grmForGenotyped) {
    genotypedIndicesInPedigree <- match(rownames(grmForGenotyped), rownames(pedigreeRelationshipMatrix))
    grmInverse <- solve(grmForGenotyped)
    A22 <- pedigreeRelationshipMatrix[genotypedIndicesInPedigree, genotypedIndicesInPedigree]
    pedigreeRelationshipInverseForGenotyped <- solve(A22)
    pedigreeRelationshipInverse <- solve(pedigreeRelationshipMatrix)
    hMatrixInverse <- pedigreeRelationshipInverse
    hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] <-
        hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] + grmInverse - pedigreeRelationshipInverseForGenotyped
    attr(hMatrixInverse, 'inverse') <- TRUE
    return(hMatrixInverse)
}

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
log_message("[PREP] A-matrix (pedigree) built successfully.")

hMatrixInverse <- doH_1_inverse(A_full, G)
log_message("[PREP] H-inverse matrix (for ssGBLUP) built successfully.")

# =====================================================================================
# Part 5: Main Analysis Loop for All Traits
# =====================================================================================
# --- User Configuration for Analysis ---
NUM_REPEATS <- 20
NUM_CORES   <- 20

# --- Setup Robust Parallel Backend ---
log_message(paste("\n[SETUP] Setting up robust parallel backend to use", NUM_CORES, "cores..."))
cl <- parallel::makeCluster(NUM_CORES)
log_message("[SETUP] Exporting python executable path to workers...")
parallel::clusterExport(cl, "python_exe_path")
log_message("[SETUP] Initializing worker environments (setting env vars, loading libraries)...")
parallel::clusterEvalQ(cl, {
  # Set all environment variables BEFORE loading any libraries
  Sys.setenv(OPENBLAS_NUM_THREADS = 1)
  Sys.setenv(MKL_NUM_THREADS = 1)
  Sys.setenv(OMP_NUM_THREADS = 1)
  
  # Use environment variables to configure TensorFlow threading PREEMPTIVELY
  Sys.setenv(TF_NUM_INTEROP_THREADS = 1)
  Sys.setenv(TF_NUM_INTRAOP_THREADS = 1)
  
  # Now, load the libraries. TensorFlow will respect the env vars upon initialization.
  library(reticulate)
  use_python(python_exe_path, required = TRUE)
  library(keras3)
  tf <- tensorflow::tf
  tensorflow::set_random_seed(1234, disable_gpu = TRUE) 
})

plan(cluster, workers = cl)
log_message("[SETUP] Parallel cluster is ready.")


# Start of the main loop to iterate through each trait
for (TRAIT_OF_INTEREST in ALL_TRAITS_IN_FILE) {

  log_message(paste0("\n\n######################################################################"))
  log_message(paste0("###   STARTING ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###"))
  log_message(paste0("######################################################################"))

  # --- Select phenotype data for the current trait ---
  phenotypeVector <- pheno_df_aligned[[TRAIT_OF_INTEREST]]
  names(phenotypeVector) <- rownames(pheno_df_aligned)

  # =====================================================================================
  # ## Perform DL hyperparameter tuning ONCE per trait.
  # =====================================================================================
  log_message(paste0("[PREP] Performing one-time hyperparameter tuning for DL models for trait: ", TRAIT_OF_INTEREST))

  # Use a random 80/20 split of the full dataset for tuning
  set.seed(123) # for reproducibility
  tuning_indices <- createDataPartition(phenotypeVector, p = 0.8, list = FALSE)
  tuning_genotype <- genotypeMatrix[tuning_indices, ]
  validation_genotype <- genotypeMatrix[-tuning_indices, ]
  tuning_phenotype <- phenotypeVector[tuning_indices]
  validation_phenotype <- phenotypeVector[-tuning_indices]

  # Scale data for tuning
  train_mean_tune <- colMeans(tuning_genotype)
  train_sd_tune <- apply(tuning_genotype, 2, sd)
  train_sd_tune[train_sd_tune == 0] <- 1
  tuning_genotype_scaled <- scale(tuning_genotype, center = train_mean_tune, scale = train_sd_tune)
  validation_genotype_scaled <- scale(validation_genotype, center = train_mean_tune, scale = train_sd_tune)

  TRAINING_EPOCHS <- 100
  callbacks_list <- list(
      callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)
  )

  # --- MLP Tuning ---
  log_message("[PREP] Starting MLP hyperparameter tuning...")
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
    log_message(paste("[PREP] MLP tuning complete. Best params: neurons=", best_mlp_params$neurons, ", dropout=", best_mlp_params$dropout_rate, ", lr=", best_mlp_params$learning_rate))
  }, error=function(e){log_message(paste("  - WARNING: MLP tuning failed with error:", e$message, ". Using defaults."))})

  # --- CNN Tuning ---
  log_message("[PREP] Starting CNN hyperparameter tuning...")
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
    log_message(paste("[PREP] CNN tuning complete. Best params: filters=", best_cnn_params$filters, ", kernel_size=", best_cnn_params$kernel_size, ", lr=", best_cnn_params$learning_rate))
  }, error=function(e){log_message(paste("  - WARNING: CNN tuning failed with error:", e$message, ". Using defaults."))})


  # --- Define the function that runs ONE full repetition for the current trait ---
  # Pass tuned DL parameters into the function.
  run_one_repetition <- function(rep_id, tuned_mlp_params, tuned_cnn_params) {
    rep_results_list <- list()
    suppressPackageStartupMessages({
      library(tidyverse); library(BGLR); library(Matrix); library(glmnet);
      library(keras3); library(caret); library(sommer)
    })

    all_possible_columns <- c("Repetition", "Fold", "Model", "Cor", "alpha", "varA", "varD", "mlp_neurons", "mlp_dropout", "cnn_filters", "cnn_kernel_size")
    standardize_df <- function(df) {
      missing_cols <- setdiff(all_possible_columns, names(df))
      if (length(missing_cols) > 0) df[missing_cols] <- NA
      return(df[, all_possible_columns])
    }

    cat(paste0("  [CV] Trait: ", TRAIT_OF_INTEREST, " - Starting Repetition ", rep_id, "/", NUM_REPEATS, "...\n"))
    set.seed(42 + rep_id)
    folds <- createFolds(phenotypeVector, k = 5, list = TRUE, returnTrain = FALSE)

    for (i in 1:length(folds)) {
      test_indices <- folds[[i]]
      test_ids <- names(phenotypeVector[test_indices])
      train_indices <- setdiff(1:numIndividuals, test_indices)

      genotypeTrain <- genotypeMatrix[train_indices, ]
      phenotypeTrain <- phenotypeVector[train_indices]
      genotypeTest <- genotypeMatrix[test_ids, ]
      phenotypeTest <- phenotypeVector[test_ids]

      phenotypeWithNAs <- phenotypeVector; phenotypeWithNAs[test_indices] <- NA

      # --- 1. glmnet Family ---
      tryCatch({
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running glmnet models (Ridge, LASSO, Elastic Net)...\n"))
        cv_ridge <- cv.glmnet(genotypeTrain, phenotypeTrain, alpha = 0, family="gaussian")
        pred_ridge <- predict(cv_ridge, newx = genotypeTest, s = "lambda.min")[, 1]
        df_ridge <- data.frame(Repetition=rep_id, Fold=i, Model="Ridge", Cor=cor(pred_ridge, phenotypeTest, use="complete.obs"), alpha=0)
        rep_results_list[[length(rep_results_list) + 1]] <- df_ridge

        cv_lasso <- cv.glmnet(genotypeTrain, phenotypeTrain, alpha = 1, family="gaussian")
        pred_lasso <- predict(cv_lasso, newx = genotypeTest, s = "lambda.min")[, 1]
        df_lasso <- data.frame(Repetition = rep_id, Fold = i, Model = "LASSO", Cor = cor(pred_lasso, phenotypeTest, use = "complete.obs"), alpha = 1)
        rep_results_list[[length(rep_results_list) + 1]] <- df_lasso

        best_alpha <- NA; best_lambda <- NA; best_mse <- Inf
        for (a in seq(0, 1, by = 0.2)) {
          cv_fit <- cv.glmnet(genotypeTrain, phenotypeTrain, alpha = a, family="gaussian")
          current_mse <- min(cv_fit$cvm, na.rm = TRUE)
          if (is.finite(current_mse) && current_mse < best_mse) { best_mse <- current_mse; best_alpha <- a; best_lambda <- cv_fit$lambda.min }
        }
        fit_en <- glmnet(genotypeTrain, phenotypeTrain, alpha = best_alpha, lambda = best_lambda, family="gaussian")
        pred_en <- predict(fit_en, newx = genotypeTest)[, 1]
        df_en <- data.frame(Repetition = rep_id, Fold = i, Model = "Elastic Net", Cor = cor(pred_en, phenotypeTest, use = "complete.obs"), alpha = best_alpha)
        rep_results_list[[length(rep_results_list) + 1]] <- df_en
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": glmnet models complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in glmnet, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })

      # --- 2. Bayesian Family (BGLR) ---
      tryCatch({
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running BGLR models...\n"))
        models_bglr <- list(
          BRR = list(list(X = genotypeMatrix, model = "BRR")),
          BayesA = list(list(X = genotypeMatrix, model = "BayesA")),
          BayesB = list(list(X = genotypeMatrix, model = "BayesB")),
          BayesC = list(list(X = genotypeMatrix, model = "BayesC")),
          `Bayes G-BLUP` = list(list(K = G, model = "RKHS"))
        )
        for(m_name in names(models_bglr)){
          cat(paste0("      - BGLR Model: ", m_name, "\n"))
          fit_bglr <- BGLR(y = phenotypeWithNAs, ETA = models_bglr[[m_name]], nIter = 10000, burnIn = 2500, verbose = FALSE, saveAt=paste0("trait_",TRAIT_OF_INTEREST,"_rep_",rep_id,"_fold_",i,"_"))
          pred_bglr <- fit_bglr$yHat[test_indices]
          df_bglr <- data.frame(Repetition = rep_id, Fold = i, Model = m_name, Cor = cor(pred_bglr, phenotypeTest, use = "complete.obs"))
          rep_results_list[[length(rep_results_list) + 1]] <- df_bglr
        }
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": BGLR models complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in BGLR, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })

      # --- 3. sommer Family (GBLUP, AD-GBLUP, ssGBLUP) ---
      # GBLUP (Corrected Workflow)
      tryCatch({
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running GBLUP...\n"))
        data_sommer <- data.frame(ID = names(phenotypeVector), y = phenotypeWithNAs)
        data_sommer$ID <- factor(data_sommer$ID, levels = rownames(G))
        
        fit_gblup <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=G), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
        
        pred_table <- predict(fit_gblup, D = "ID")
        pred_gblup <- pred_table$pvals[test_ids, "predicted.value"]
        
        df_gblup <- data.frame(Repetition=rep_id, Fold=i, Model="GBLUP", Cor=cor(pred_gblup, phenotypeTest, use="complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- df_gblup
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": GBLUP complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in GBLUP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })

      # AD-GBLUP (Corrected Workflow)
      tryCatch({
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running AD-GBLUP...\n"))
        data_sommer <- data.frame(ID = names(phenotypeVector), y = phenotypeWithNAs)
        data_sommer$ID_A <- factor(data_sommer$ID, levels = rownames(G))
        data_sommer$ID_D <- factor(data_sommer$ID, levels = rownames(D))
        
        fit_ad <- mmes(fixed=y~1, random=~vsm(ism(ID_A), Gu=G) + vsm(ism(ID_D), Gu=D), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
        # pred_table <- predict(fit_ad, D = "ID_A")
        # pred_ad <- pred_table$pvals[test_ids, "predicted.value"]
        intercept <- fit_ad$b[1, 1]
        u_A <- fit_ad$uList[[1]][test_ids, , drop=FALSE] 
        u_D <- fit_ad$uList[[2]][test_ids, , drop=FALSE]
        pred_ad <- intercept + u_A + u_D

        var_a <- fit_ad$sigma[[1]]; if (length(var_a) == 0) var_a <- NA
        var_d <- fit_ad$sigma[[2]]; if (length(var_d) == 0) var_d <- NA
        
        df_ad <- data.frame(Repetition=rep_id, Fold=i, Model="AD-GBLUP", Cor=cor(pred_ad, phenotypeTest, use="complete.obs"), varA=var_a, varD=var_d)
        rep_results_list[[length(rep_results_list) + 1]] <- df_ad
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": AD-GBLUP complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in AD-GBLUP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
        
      # ssGBLUP (Corrected Workflow)
      tryCatch({
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running ssGBLUP...\n"))
        pheno_ssgblup <- rep(NA, nrow(A_full)); names(pheno_ssgblup) <- rownames(A_full)
        pheno_ssgblup[names(phenotypeWithNAs)] <- phenotypeWithNAs
        data_sommer <- data.frame(ID = names(pheno_ssgblup), y = pheno_ssgblup)
        data_sommer$ID <- factor(data_sommer$ID, levels = rownames(hMatrixInverse))
        
        fit_ss <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=hMatrixInverse), rcov=~units, data=data_sommer, naMethodY="include", verbose=F, henderson=T)
        pred_table <- predict(fit_ss, D = "ID")
        pred_ss <- pred_table$pvals[test_ids, "predicted.value"]
        
        df_ss <- data.frame(Repetition=rep_id, Fold=i, Model="ssGBLUP", Cor=cor(pred_ss, phenotypeTest, use="complete.obs"))
        rep_results_list[[length(rep_results_list) + 1]] <- df_ss
        cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": ssGBLUP complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in ssGBLUP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })

      # 老是报错，放弃，需要检查软件版本参数更新情况
      # # ssGBLUP (Corrected Workflow with simulated incomplete-information scenario)
      # tryCatch({
      #   cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running ssGBLUP (with simulated scenario)...\n"))

      #   set.seed(123 + i) # set seeds to be able to repeat sampling
      #   train_gp_indices <- sample(train_indices, size = floor(0.5 * length(train_indices)))
      #   train_gp_ids <- names(phenotypeVector[train_gp_indices])

      #   G_subset <- G[train_gp_ids, train_gp_ids]
      #   hMatrixInverse_fold <- doH_1_inverse(A_full, G_subset)

      #   # new hMatrixInverse_fold to run the model
      #   pheno_ssgblup <- rep(NA, nrow(A_full)); names(pheno_ssgblup) <- rownames(A_full)
      #   pheno_ssgblup[names(phenotypeWithNAs)] <- phenotypeWithNAs
      #   data_sommer <- data.frame(ID = names(pheno_ssgblup), y = pheno_ssgblup)
        
      #   # use the new hMatrixInverse_fold
      #   data_sommer$ID <- factor(data_sommer$ID, levels = rownames(hMatrixInverse_fold))
        
      #   fit_ss <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=hMatrixInverse_fold), rcov=~units, data=data_sommer, naMethodY="include", verbose=F, henderson=T)
      #   pred_table <- predict(fit_ss, D = "ID")
      #   pred_ss <- pred_table$pvals[test_ids, "predicted.value"]
        
      #   df_ss_sampled <- data.frame(Repetition=rep_id, Fold=i, Model="ssGBLUP_50_Sampled", Cor=cor(pred_ss, phenotypeTest, use="complete.obs"))
      #   rep_results_list[[length(rep_results_list) + 1]] <- df_ss_sampled
      #   cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": ssGBLUP (with simulated scenario) complete.\n"))
        
      # }, error = function(e){ cat(paste0("  - ERROR in ssGBLUP with simulated scenariao, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })

      # --- 4. Deep Learning Models ---
      cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Preparing data for DL models...\n"))
      train_mean <- colMeans(genotypeTrain); train_sd <- apply(genotypeTrain, 2, sd); train_sd[train_sd == 0] <- 1
      genotypeTrain_scaled <- scale(genotypeTrain, center = train_mean, scale = train_sd)
      genotypeTest_scaled <- scale(genotypeTest, center = train_mean, scale = train_sd)

      # --- MLP with optimized parameters ---
      tryCatch({
          cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running MLP...\n"))
          final_model <- keras_model_sequential() %>%
            layer_dense(units=tuned_mlp_params$neurons, input_shape=numMarkers, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=tuned_mlp_params$dropout_rate) %>%
            layer_dense(units=round(tuned_mlp_params$neurons/2), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          final_model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=tuned_mlp_params$learning_rate))
          final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
          final_model %>% fit(genotypeTrain_scaled, phenotypeTrain, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
          pred_mlp <- final_model %>% predict(genotypeTest_scaled, verbose=0)
          df_mlp <- data.frame(Repetition=rep_id, Fold=i, Model="MLP", Cor=cor(pred_mlp[,1], phenotypeTest, use="complete.obs"), mlp_neurons=tuned_mlp_params$neurons, mlp_dropout=tuned_mlp_params$dropout_rate)
          rep_results_list[[length(rep_results_list) + 1]] <- df_mlp
          cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": MLP complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in MLP, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })

      # --- CNN with optimized parameters ---
      tryCatch({
          cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": Running CNN...\n"))
          xtrain_cnn <- array(genotypeTrain_scaled, dim=c(nrow(genotypeTrain_scaled), numMarkers, 1))
          xtest_cnn <- array(genotypeTest_scaled, dim=c(nrow(genotypeTest_scaled), numMarkers, 1))
          final_model <- keras_model_sequential() %>%
            layer_conv_1d(filters=tuned_cnn_params$filters, kernel_size=tuned_cnn_params$kernel_size, input_shape=c(numMarkers,1), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
            layer_dense(units=64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          final_model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=tuned_cnn_params$learning_rate))
          final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
          final_model %>% fit(xtrain_cnn, phenotypeTrain, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
          pred_cnn <- final_model %>% predict(xtest_cnn, verbose=0)
          df_cnn <- data.frame(Repetition=rep_id, Fold=i, Model="CNN", Cor=cor(pred_cnn[,1], phenotypeTest, use="complete.obs"), cnn_filters=tuned_cnn_params$filters, cnn_kernel_size=tuned_cnn_params$kernel_size)
          rep_results_list[[length(rep_results_list) + 1]] <- df_cnn
          cat(paste0("    [CV] Rep ", rep_id, " Fold ", i, ": CNN complete.\n"))
      }, error = function(e){ cat(paste0("  - ERROR in CNN, Rep ", rep_id, " Fold ", i, ": ", e$message, "\n")) })
    }
    cat(paste0("  [CV] Trait: ", TRAIT_OF_INTEREST, " - Finished Repetition ", rep_id, "/", NUM_REPEATS, ".\n"))
    return(lapply(rep_results_list, standardize_df) %>% dplyr::bind_rows())
  }


log_message(paste("[SETUP] Exporting required objects to parallel workers for trait:", TRAIT_OF_INTEREST, "..."))
required_globals <- c(
    "run_one_repetition",
    "phenotypeVector", "genotypeMatrix", "G", "D", "hMatrixInverse",
    "A_full", "numIndividuals", "numMarkers", "TRAIT_OF_INTEREST", "NUM_REPEATS"
)
parallel::clusterExport(cl, varlist = required_globals, envir = environment())
log_message("[SETUP] Objects successfully exported to all workers.")

  # --- Run repetitions in parallel for the current trait ---
  log_message(paste("[EXECUTION] Starting parallel cross-validation for trait:", TRAIT_OF_INTEREST, "..."))
  results_df <- future_map_dfr(
    .x = 1:NUM_REPEATS,
    # ## Pass the one-time tuned parameters to each repetition
    .f = ~run_one_repetition(.x, tuned_mlp_params = best_mlp_params, tuned_cnn_params = best_cnn_params),
    .options = furrr_options(seed = TRUE)
  )
  log_message(paste("\n[COMPLETE] Cross-validation finished for trait: ", TRAIT_OF_INTEREST, "!"))

  # =====================================================================================
  # Part 6: Result Summarization and Visualization for the Current Trait
  # =====================================================================================
  # (This part remains the same, it was already correct)
  log_message(paste("\n[RESULTS] Summarizing and plotting results for trait:", TRAIT_OF_INTEREST, "..."))

  if (nrow(results_df) > 0) {
      log_message("[RESULTS] Calculating summary statistics (Mean, SD)...")
      summary_stats <- results_df %>%
        filter(!is.na(Cor)) %>%
        group_by(Model) %>%
        summarise(Mean_Cor = mean(Cor), SD_Cor = sd(Cor), .groups = 'drop') %>%
        arrange(desc(Mean_Cor))
      print(summary_stats)

      results_for_test <- results_df %>% filter(!is.na(Cor))
      if(length(unique(results_for_test$Model)) > 1){
          log_message("[RESULTS] Performing ANOVA and Tukey's HSD tests...")
          anova_results <- aov(Cor ~ Model, data = results_for_test)
          tukey_results <- TukeyHSD(anova_results)
          cat("\n--- ANOVA Summary ---\n"); print(summary(anova_results))
          cat("\n--- Tukey's HSD Paired Comparisons ---\n"); print(tukey_results)
          tukey_df <- as.data.frame(tukey_results$Model)
          tukey_df$comparison <- rownames(tukey_df)
          log_message("[RESULTS] Saving Tukey HSD results to CSV...")
          write.csv(tukey_df, paste0("results/GS_TukeyHSD_", TRAIT_OF_INTEREST, ".csv"), row.names = FALSE)
          log_message("[RESULTS] Tukey HSD results saved.")
      }

      log_message("[RESULTS] Generating boxplot of model performance...")
      plot_main <- ggplot(results_df %>% filter(!is.na(Cor)), aes(x = reorder(Model, Cor, FUN = median), y = Cor, fill = Model)) +
        geom_boxplot(alpha = 0.8, show.legend = FALSE) +
        stat_summary(fun=mean, geom="point", shape=23, size=3, fill="white", show.legend = FALSE) +
        coord_flip() +
        labs(
          title = "Comparison of Genomic Prediction Models",
          subtitle = paste0("Trait: ", TRAIT_OF_INTEREST, " (", NUM_REPEATS, " Repeats of 5-Fold CV)"),
          x = "Model", y = "Prediction Accuracy (Pearson's r)"
        ) +
        theme_publication() +
        theme(axis.text.y = element_text(face = "bold"))

      plot_filename <- paste0("plots/GS_Model_Comparison_", TRAIT_OF_INTEREST, ".pdf")
      log_message(paste("[RESULTS] Saving plot to", plot_filename, "..."))
      ggsave(plot_filename, plot = plot_main, width = 10, height = 8, device = cairo_pdf)
      
      results_filename <- paste0("results/GS_All_Results_", TRAIT_OF_INTEREST, ".csv")
      summary_filename <- paste0("results/GS_Summary_", TRAIT_OF_INTEREST, ".csv")
      log_message(paste("[RESULTS] Saving full results to", results_filename, "..."))
      write.csv(results_df, results_filename, row.names=FALSE)
      log_message(paste("[RESULTS] Saving summary results to", summary_filename, "..."))
      write.csv(summary_stats, summary_filename, row.names = FALSE)

      log_message(paste("\n[SAVED] Analysis complete for ", TRAIT_OF_INTEREST, ". Results and plots are saved."))
  } else {
      log_message(paste("\n[WARNING] No results were generated for trait: ", TRAIT_OF_INTEREST, ". Please check for errors in the logs."))
  }

  # Clean up BGLR files for the current trait before starting the next one
  log_message(paste("[CLEANUP] Removing temporary BGLR files for trait:", TRAIT_OF_INTEREST, "..."))
  unlink(list.files(pattern = paste0("trait_", TRAIT_OF_INTEREST, ".*.dat")))

} # End of the main loop for traits


# =====================================================================================
# Part 7: Final Cleanup
# =====================================================================================
log_message("\n[CLEANUP] All traits analyzed. Stopping the parallel cluster...")
parallel::stopCluster(cl)
plan(sequential)
log_message("[CLEANUP] Parallel cluster stopped.")
log_message("\n[--- FINISHED ---] Analysis pipeline complete for all traits.")
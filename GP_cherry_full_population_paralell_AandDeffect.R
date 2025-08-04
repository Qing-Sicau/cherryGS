# =====================================================================================
# Part 0: Environment Setup (Load all necessary packages)
# =====================================================================================
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
# Part 0.1: Find Python Path Before Parallelization
# =====================================================================================
# First, configure reticulate in the main session to find the python path.
cat("Finding Python executable from the 'reseq' conda environment...\n")
use_condaenv("reseq", required = TRUE)

# Get the exact path to the python binary that we will send to the workers.
# This is the most important variable.
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

# Create output directory for plots if it doesn't exist
output_dir <- getwd()
if (!dir.exists(file.path(output_dir, "plots"))) dir.create(file.path(output_dir, "plots"), recursive = TRUE)

my_save_plot <- function(plot, filename, width = 12, height = 8) {
  ggsave(file.path(output_dir, "plots", filename),
         plot,
         width = width,
         height = height,
         device = cairo_pdf,
         dpi = 300)
  cat("Saved plot:", filename, "\n")
}

# =====================================================================================
# Part 0.5: Load, Clean, and Preprocess Your Real Data
# =====================================================================================
cat("Loading and preprocessing your real data...\n")

# --- User Configuration ---
GENOTYPE_FILE <- "genotype.dosages.tsv"
PHENOTYPE_FILE <- "phenotype_BLUPs.csv"
# Please specify the trait you want to analyze here.
# It must exactly match a row name in your phenotype.csv file.

TRAIT_OF_INTEREST <- "fruit_weight" # <-- !!! EDIT THIS !!!

# --- Load Genotype Data ---
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
cat(paste("Found", length(common_individuals), "individuals with both genotype and phenotype data.\n"))

genotypeMatrix <- genotypeMatrix_raw[common_individuals, ]

phenotype_full <- pheno_df %>%
  dplyr::mutate(my_row = rownames(pheno_df)) %>%
  dplyr::filter(my_row %in% common_individuals) %>%
  dplyr::select(TRAIT_OF_INTEREST) %>%
  as.data.frame()

if (!TRAIT_OF_INTEREST %in% colnames(phenotype_full)) {
  stop(paste("Error: The specified trait '", TRAIT_OF_INTEREST, "' was not found in the phenotype file. Please check spelling and case."))
}

pheno_values <- phenotype_full[[TRAIT_OF_INTEREST]]
phenotypeVector <- as.numeric(pheno_values)
names(phenotypeVector) <- common_individuals

# --- Handle Missing Genotype Values (Mean Imputation) ---
cat("Handling missing values in genotypes (using mean imputation)...\n")
impute_mean <- function(x) {
  mean_val <- mean(x, na.rm = TRUE)
  if (is.nan(mean_val)) {
    mean_val <- 0
  }
  x[is.na(x)] <- round(mean_val)
  return(x)
}


genotypeMatrix <- apply(genotypeMatrix, 2, impute_mean)
if(sum(is.na(genotypeMatrix)) > 0) {
  warning("Missing values still exist in the genotype matrix after imputation. This can happen if a marker is missing for all individuals. Please check your data.")
}

# --- Update Global Variables ---
numIndividuals <- nrow(genotypeMatrix)
numMarkers <- ncol(genotypeMatrix)
cat(paste("Data preparation complete:", numIndividuals, "individuals,", numMarkers, "SNP markers.\n"))

# =====================================================================================
# Part 1: Helper Functions and Pedigree Preparation for ssGBLUP
# =====================================================================================

# Source the function to create the dominance matrix.
cat("Loading dominance relationship matrix function...\n")
source("get_DomRel_matrix.R")


# --- G-Matrix Function for Polyploid Data ---
create_g_matrix_polyploid <- function(M, ploidy_level) {
  cat("Calculating Additive G-matrix for polyploid data using AGHmatrix...\n")
  G <- Gmatrix(M, method = "VanRaden", ploidy = ploidy_level)
  G_reg <- G + diag(nrow(M)) * 1e-4
  return(G_reg)
}

# Function to create the dominance relationship matrix with names preserved
create_d_matrix_polyploid <- function(M, ploidy_level) {
  cat("Calculating Dominance D-matrix for polyploid data using get_DomRel...\n")
  D <- get_DomRel(M, ploidy = ploidy_level)
  
  # Center the matrix
  Ic <- diag(nrow(D)) - (1/nrow(D)) * matrix(1, nrow(D), nrow(D))
  D_centered <- Ic %*% D %*% Ic
  
  # Regularize the matrix
  D_reg <- D_centered + diag(nrow(M)) * 1e-4
  
  rownames(D_reg) <- rownames(M)
  colnames(D_reg) <- rownames(M)
  
  return(D_reg)
}

# --- Helper for ssGBLUP H-inverse matrix still needed
doH_1_inverse <- function(pedigreeRelationshipMatrix, grmForGenotyped, genotypedIndicesInPedigree) {
  grmInverse <- solve(grmForGenotyped)
  A22 <- pedigreeRelationshipMatrix[genotypedIndicesInPedigree, genotypedIndicesInPedigree]
  pedigreeRelationshipInverseForGenotyped <- solve(A22)
  pedigreeRelationshipInverse <- solve(pedigreeRelationshipMatrix)
  hMatrixInverse <- pedigreeRelationshipInverse
  hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] <-
    hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] + grmInverse - pedigreeRelationshipInverseForGenotyped
  return(hMatrixInverse)
}


# --- Build Real Pedigree for ssGBLUP ---
cat("Building real pedigree based on your provided relationship information...\n")
parents <- c("HF1", "NZH2", "PJHH")
offspring_ids <- rownames(genotypeMatrix)
all_ped_ids <- unique(c(parents, offspring_ids))
ped_df <- data.frame(
  ID = all_ped_ids,
  Sire = 0,
  Dam = 0,
  stringsAsFactors = FALSE
)
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) {
    ped_df$Sire[i] <- "HF1"
    ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) {
    ped_df$Sire[i] <- "HF1"
    ped_df$Dam[i] <- "PJHH"
  }
}

pedigreeRelationshipMatrix <- Amatrix(ped_df, ploidy = 4)
cat("A-matrix (pedigree relationship) built successfully.\n")

genotypeMatrixForG <- genotypeMatrix[, which(apply(genotypeMatrix, 2, sd) != 0)]
grmForGenotyped <- create_g_matrix_polyploid(genotypeMatrixForG, ploidy_level = 4)
genotypedIndicesInPedigree <- match(rownames(genotypeMatrix), rownames(pedigreeRelationshipMatrix))

hMatrixInverse <- doH_1_inverse(pedigreeRelationshipMatrix, grmForGenotyped, genotypedIndicesInPedigree)
cat("H-inverse matrix (for ssGBLUP) built successfully.\n")


# =====================================================================================
# Part 2: Main Loop for Repeated 5-Fold Cross-Validation (PARALLELIZED)
# =====================================================================================
# --- User Configuration ---
NUM_REPEATS <- 20 # <-- Set the number of repetitions here.
NUM_CORES <- 20   # <-- !!! SET THE NUMBER OF CPU CORES TO USE !!!

# --- Setup Parallel Backend (Definitive Cluster Method) ---
cat("Setting up robust parallel cluster with direct path injection...\n")

cl <- parallel::makeCluster(NUM_CORES)
parallel::clusterExport(cl, "python_exe_path")
parallel::clusterEvalQ(cl, {
  library(reticulate)
  use_python(python_exe_path, required = TRUE)
  library(keras)
})
plan(cluster, workers = cl)
cat("Cluster is ready. Starting computations.\n")


set.seed(42)

# --- Hyperparameter Grids ---
alpha_grid <- seq(0, 1, by = 0.2)

# --- Pre-calculate Relationship Matrices ---
G <- create_g_matrix_polyploid(genotypeMatrix, ploidy_level = 4)
D <- create_d_matrix_polyploid(genotypeMatrix, ploidy_level = 4)

# --- Define the function that runs ONE full repetition ---
run_one_repetition <- function(rep_id) {
  
  rep_results_list <- list()
  
  # Ensure packages are available on each parallel worker
  library(tidyverse)
  library(BGLR)
  library(Matrix)
  library(glmnet)
  library(keras)
  library(caret)
  library(sommer)
  
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
  
  cat(paste0(">>> Processing Repetition ", rep_id, "/", NUM_REPEATS, " <<<\n"))
  
  folds <- createFolds(phenotypeVector, k = 5, list = TRUE, returnTrain = FALSE)
  
  # Inner 5-Fold Cross-Validation Loop
  for (i in 1:length(folds)) {
    cat(paste0("  --- [Repetition ", rep_id, "] Fold ", i, "/5 ---\n"))
    
    test_indices <- folds[[i]]
    train_indices <- setdiff(1:numIndividuals, test_indices)
    
    # Use names for robust subsetting
    test_ids <- names(phenotypeVector[test_indices])
    
    genotypeTrain <- genotypeMatrix[train_indices, ]
    phenotypeTrain <- phenotypeVector[train_indices]
    genotypeTest <- genotypeMatrix[test_ids, ]
    phenotypeTest <- phenotypeVector[test_ids]
    
    phenotypeWithNAs <- phenotypeVector
    phenotypeWithNAs[test_indices] <- NA
    
    # --- 1. glmnet Family ---
    cv_ridge <- cv.glmnet(genotypeTrain, phenotypeTrain, alpha = 0, family="gaussian")
    pred_ridge <- predict(cv_ridge, newx = genotypeTest, s = "lambda.min")[, 1]
    df_ridge <- data.frame(Repetition = rep_id, Fold = i, Model = "Ridge", Cor = cor(pred_ridge, phenotypeTest, use = "complete.obs"), alpha = 0)
    rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ridge)
    
    cv_lasso <- cv.glmnet(genotypeTrain, phenotypeTrain, alpha = 1, family="gaussian")
    pred_lasso <- predict(cv_lasso, newx = genotypeTest, s = "lambda.min")[, 1]
    df_lasso <- data.frame(Repetition = rep_id, Fold = i, Model = "LASSO", Cor = cor(pred_lasso, phenotypeTest, use = "complete.obs"), alpha = 1)
    rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_lasso)
    
    best_alpha <- NA; best_lambda <- NA; best_mse <- Inf
    for (a in alpha_grid) {
      cv_fit <- cv.glmnet(genotypeTrain, phenotypeTrain, alpha = a, family="gaussian")
      current_mse <- min(cv_fit$cvm, na.rm = TRUE)
      if (is.finite(current_mse) && current_mse < best_mse) {
        best_mse <- current_mse; best_alpha <- a; best_lambda <- cv_fit$lambda.min
      }
    }
    fit_en <- glmnet(genotypeTrain, phenotypeTrain, alpha = best_alpha, lambda = best_lambda, family="gaussian")
    pred_en <- predict(fit_en, newx = genotypeTest)[, 1]
    df_en <- data.frame(Repetition = rep_id, Fold = i, Model = "Elastic Net", Cor = cor(pred_en, phenotypeTest, use = "complete.obs"), alpha = best_alpha)
    rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_en)
    
    # --- 2. GBLUP (using sommer) ---
    tryCatch({
      sommer_data_gblup <- data.frame(ID = names(phenotypeVector), y = as.numeric(phenotypeWithNAs))
      sommer_data_gblup$ID <- factor(sommer_data_gblup$ID, levels = rownames(G))
      
      fit_gblup <- mmes(
        fixed = y ~ 1,
        random = ~vsm(ism(ID), Gu = G),
        rcov = ~units,
        data = sommer_data_gblup,
        naMethodY = "include",
        verbose = FALSE
      )
      
      pred_gblup <- fitted(fit_gblup)[test_ids]
      df_gblup <- data.frame(Repetition = rep_id, Fold = i, Model = "GBLUP", Cor = cor(pred_gblup, phenotypeTest, use = "complete.obs"))
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_gblup)
    }, error = function(e) {
      cat(paste0(" ERROR in GBLUP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n"))
      df_error <- data.frame(Repetition = rep_id, Fold = i, Model = "GBLUP", Cor = NA)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_error)
    })
    
    # --- 3. Bayesian Family ---
    ETA_BRR <- list(list(X = genotypeMatrix, model = "BRR"))
    ETA_BayesA <- list(list(X = genotypeMatrix, model = "BayesA"))
    ETA_BayesB <- list(list(X = genotypeMatrix, model = "BayesB"))
    ETA_BayesC <- list(list(X = genotypeMatrix, model = "BayesC"))
    ETA_BGBLUP <- list(list(K = G, model = "RKHS"))
    models_bglr <- list(BRR=ETA_BRR, BayesA=ETA_BayesA, BayesB=ETA_BayesB, BayesC=ETA_BayesC, `Bayes G-BLUP`=ETA_BGBLUP)
    for(m_name in names(models_bglr)){
      fit_bglr <- BGLR(y = phenotypeWithNAs, ETA = models_bglr[[m_name]], nIter = 10000, burnIn = 2500, verbose = FALSE, saveAt=paste0("rep_",rep_id,"_fold_",i,"_"))
      pred_bglr <- fit_bglr$yHat[test_indices]
      df_bglr <- data.frame(Repetition = rep_id, Fold = i, Model = m_name, Cor = cor(pred_bglr, phenotypeTest, use = "complete.obs"))
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_bglr)
    }
    
    # --- 4. Single-Step GBLUP (ssGBLUP using sommer) ---
    tryCatch({
      phenotypeVectorWithFounders <- rep(NA, nrow(pedigreeRelationshipMatrix))
      names(phenotypeVectorWithFounders) <- rownames(pedigreeRelationshipMatrix)
      phenotypeVectorWithFounders[names(phenotypeVector)] <- phenotypeVector
      
      phenotypeWithNAsAndFounders <- phenotypeVectorWithFounders
      phenotypeWithNAsAndFounders[test_ids] <- NA
      
      sommer_data_ss <- data.frame(
        ID = names(phenotypeWithNAsAndFounders),
        y = phenotypeWithNAsAndFounders,
        stringsAsFactors = TRUE
      )
      sommer_data_ss$ID <- factor(sommer_data_ss$ID, levels = rownames(hMatrixInverse))
      attr(hMatrixInverse, 'inverse') <- TRUE
      
      fit_ssgblup <- mmes(
        fixed = y ~ 1,
        random = ~vsm(ism(ID), Gu = hMatrixInverse),
        rcov = ~units,
        data = sommer_data_ss,
        naMethodY = "include",
        verbose = FALSE,
        nIters = 150,
        henderson = TRUE
      )
      
      pred_ssgblup <- fitted(fit_ssgblup)[test_ids]
      df_ssgblup <- data.frame(Repetition = rep_id, Fold = i, Model = "ssGBLUP", Cor = cor(pred_ssgblup, phenotypeTest, use = "complete.obs"))
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ssgblup)
    }, error = function(e) {
      cat(paste0("  ERROR in ssGBLUP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n"))
      df_error <- data.frame(Repetition = rep_id, Fold = i, Model = "ssGBLUP", Cor = NA)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_error)
    })
    
    # --- 5. Additive + Dominance GBLUP (AD-GBLUP) using sommer ---
    tryCatch({
      sommer_data <- data.frame(
        ID = names(phenotypeVector),
        y = as.numeric(phenotypeWithNAs),
        stringsAsFactors = TRUE
      )
      sommer_data$ID_A <- factor(sommer_data$ID, levels = rownames(G))
      sommer_data$ID_D <- factor(sommer_data$ID, levels = rownames(D))
      
      fit_ad_gblup <- mmes(
        fixed = y ~ 1,
        random = ~vsm(ism(ID_A), Gu = G) + vsm(ism(ID_D), Gu = D),
        rcov = ~ units,
        data = sommer_data,
        naMethodY = "include",
        verbose = FALSE
      )
      
      pred_ad_gblup <- fitted(fit_ad_gblup)[test_ids]
      cor_ad <- cor(pred_ad_gblup, phenotypeTest, use = "complete.obs")
      
      var_a <- fit_ad_gblup$theta[[1]][1]
      var_d <- fit_ad_gblup$theta[[2]][1]
      if (length(var_a) == 0) var_a <- NA
      if (length(var_d) == 0) var_d <- NA
      
      df_ad <- data.frame(
        Repetition = rep_id, Fold = i, Model = "AD-GBLUP",
        Cor = cor_ad, varA = var_a, varD = var_d
      )
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_ad)
    }, error = function(e) {
      cat(paste0("  ERROR in AD-GBLUP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n"))
      df_error <- data.frame(Repetition = rep_id, Fold = i, Model = "AD-GBLUP", Cor = NA)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_error)
    })
    
    # --- 6. Deep Learning Models ---
    train_mean <- colMeans(genotypeTrain); train_sd <- apply(genotypeTrain, 2, sd); train_sd[train_sd == 0] <- 1
    genotypeTrain_scaled <- scale(genotypeTrain, center = train_mean, scale = train_sd)
    genotypeTest_scaled <- scale(genotypeTest, center = train_mean, scale = train_sd)
    
    tryCatch({ # MLP
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
        hist <- model %>% fit(genotypeTrain_scaled, phenotypeTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
        val_loss <- min(hist$metrics$val_loss, na.rm=T)
        if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
      }
      final_model <- keras_model_sequential() %>%
        layer_dense(units = best_mlp_params$neurons, activation = "relu", input_shape = numMarkers) %>%
        layer_dropout(rate = best_mlp_params$dropout_rate) %>%
        layer_dense(units = round(best_mlp_params$neurons/2), activation="relu") %>%
        layer_dense(units = 1)
      final_model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))
      final_model %>% fit(genotypeTrain_scaled, phenotypeTrain, epochs = 40, batch_size=32, verbose = 0)
      pred_mlp <- final_model %>% predict(genotypeTest_scaled)
      df_mlp <- data.frame(Repetition = rep_id, Fold = i, Model = "MLP", Cor = cor(pred_mlp[, 1], phenotypeTest, use = "complete.obs"), mlp_neurons = best_mlp_params$neurons, mlp_dropout = best_mlp_params$dropout_rate)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_mlp)
    }, error = function(e){
      cat(paste0("  ERROR in MLP model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n"))
      df_error <- data.frame(Repetition = rep_id, Fold = i, Model = "MLP", Cor = NA)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_error)
    })
    
    tryCatch({ # CNN
      xtrain_cnn <- array(genotypeTrain_scaled, dim = c(nrow(genotypeTrain_scaled), ncol(genotypeTrain_scaled), 1))
      xtest_cnn <- array(genotypeTest_scaled, dim = c(nrow(genotypeTest_scaled), ncol(genotypeTest_scaled), 1))
      cnn_param_grid <- expand.grid(filters = c(32, 64), kernel_size = c(5, 10))
      best_val_loss <- Inf; best_cnn_params <- list(filters=NA, kernel_size=NA)
      for(j in 1:nrow(cnn_param_grid)) {
        params <- cnn_param_grid[j, ]
        model <- keras_model_sequential() %>%
          layer_conv_1d(filters = params$filters, kernel_size = params$kernel_size, activation = "relu", input_shape = c(numMarkers, 1)) %>%
          layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
          layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
        model %>% compile(loss = "mse", optimizer = "rmsprop")
        hist <- model %>% fit(xtrain_cnn, phenotypeTrain, epochs = 30, batch_size=32, validation_split = 0.2, verbose = 0, callbacks=list(callback_early_stopping(patience=5)))
        val_loss <- min(hist$metrics$val_loss, na.rm=T)
        if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_cnn_params <- params }
      }
      final_model <- keras_model_sequential() %>%
        layer_conv_1d(filters = best_cnn_params$filters, kernel_size = best_cnn_params$kernel_size, activation = "relu", input_shape = c(numMarkers, 1)) %>%
        layer_max_pooling_1d(pool_size = 4) %>% layer_flatten() %>%
        layer_dense(units = 64, activation="relu") %>% layer_dense(units = 1)
      final_model %>% compile(loss = "mse", optimizer = "rmsprop")
      final_model %>% fit(xtrain_cnn, phenotypeTrain, epochs = 40, batch_size=32, verbose = 0)
      pred_cnn <- final_model %>% predict(xtest_cnn)
      df_cnn <- data.frame(Repetition = rep_id, Fold = i, Model = "CNN", Cor = cor(pred_cnn[, 1], phenotypeTest, use = "complete.obs"), cnn_filters = best_cnn_params$filters, cnn_kernel_size = best_cnn_params$kernel_size)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_cnn)
    }, error = function(e){
      cat(paste0("  ERROR in CNN model for Rep ", rep_id, " Fold ", i, ": ", e$message, "\n"))
      df_error <- data.frame(Repetition = rep_id, Fold = i, Model = "CNN", Cor = NA)
      rep_results_list[[length(rep_results_list) + 1]] <- standardize_df(df_error)
    })
    
  } # End of inner 5-fold loop
  
  return(dplyr::bind_rows(rep_results_list))
}


# --- Run the repetitions in parallel ---
cat(paste0("\n!!! Starting ", NUM_REPEATS, " repeats of 5-fold cross-validation in parallel on ", NUM_CORES, " cores !!!\n"))
cat("This will be a time-consuming process. Please be patient.\n")

results_df <- future_map_dfr(
  .x = 1:NUM_REPEATS,
  .f = run_one_repetition,
  .options = furrr_options(seed = TRUE)
)

# --- Clean Up BGLR files and Finalize ---
unlink(list.files(pattern = "rep_.*.dat"))
cat("\nCross-validation finished!\n")

# --- IMPORTANT: Stop the parallel cluster ---
cat("Stopping the parallel cluster...\n")
parallel::stopCluster(cl)


# =====================================================================================
# Part 3: Result Summarization and Visualization
# =====================================================================================
# --- Calculate Summary Statistics ---
summary_stats <- results_df %>%
  filter(!is.na(Cor)) %>%
  group_by(Model) %>%
  summarise(
    Mean_Cor = mean(Cor),
    SD_Cor = sd(Cor),
    .groups = 'drop'
  ) %>%
  arrange(desc(Mean_Cor))

print("Summary of prediction accuracies from repeated cross-validation:")
print(summary_stats)

# --- Main Plot: Model Performance Comparison ---
plot_main <- ggplot(results_df %>% filter(!is.na(Cor)), aes(x = reorder(Model, Cor, FUN = median), y = Cor, fill = Model)) +
  geom_boxplot(alpha = 0.8, show.legend = FALSE) +
  stat_summary(fun=mean, geom="point", shape=23, size=3, fill="white", show.legend = FALSE) +
  coord_flip() +
  labs(
    title = "Comparison of Different Genomic Prediction Models",
    subtitle = paste0("Trait: ", TRAIT_OF_INTEREST, " (", NUM_REPEATS, " Repeats of 5-Fold CV)"),
    x = "Model",
    y = "Prediction Accuracy (r)"
  ) +
  theme_publication() +
  theme(axis.text.y = element_text(face = "bold"))

# --- Print and Save Results ---
print(plot_main)
ggsave(paste0("GS_Model_Comparison_", TRAIT_OF_INTEREST, ".png"), plot = plot_main, width = 8, height = 7, dpi = 300)
write.csv(results_df, paste0("GS_All_Results_", TRAIT_OF_INTEREST, ".csv"), row.names=FALSE)
write.csv(summary_stats, paste0("GS_Summary_", TRAIT_OF_INTEREST, ".csv"), row.names = FALSE)
cat("\nAnalysis pipeline finished. Result plots and summary data have been saved to your working directory.\n")


# =====================================================================================
# Part 3.5: Statistical Significance Testing (NEW)
# =====================================================================================

cat("Part 3.5: Statistical Significance Testing\n")

# Filter for valid correlation values
results_for_test <- results_df %>% filter(!is.na(Cor))

# Perform ANOVA to test for any significant difference among models
cat("--- Performing ANOVA on Model Performance ---\n")
anova_results <- aov(Cor ~ Model, data = results_for_test)
print(summary(anova_results))

# Perform Tukey's HSD test for pairwise model comparisons
cat("\n--- Performing Tukey's HSD for Pairwise Comparisons ---\n")
tukey_results <- TukeyHSD(anova_results)
print(tukey_results)
cat("\nNOTE: In the Tukey HSD results, a 'p adj' value < 0.05 indicates a significant difference between that pair of models.\n")


# =====================================================================================
# Part 4: Analysis of the Best Model and its Hyperparameters
# =====================================================================================
cat("\n\n=================================================\n")
cat("Part 4: Best Model and Hyperparameter Analysis\n")
cat("=================================================\n\n")

# 1. Identify the best performing model based on mean accuracy.
best_model_name <- summary_stats$Model[1]
cat(paste("Analysis shows that the model with the highest average prediction accuracy is:", best_model_name, "\n\n"))

# 2. Filter the results for this best model.
best_model_results <- results_df %>%
  filter(Model == best_model_name, !is.na(Cor))

# 3. Analyze and report the most frequently used hyperparameters for this model.
cat("In", nrow(best_model_results), "successful cross-validation runs, the distribution of this model's optimal hyperparameters was as follows:\n")

# Helper function to find the mode (most frequent value).
get_mode <- function(v) {
  v <- v[!is.na(v)]
  if(length(v) == 0) return(NA)
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

if (best_model_name %in% c("Ridge", "LASSO", "Elastic Net")) {
  if ("alpha" %in% names(best_model_results)) {
    param_table <- table(best_model_results$alpha)
    best_param <- get_mode(best_model_results$alpha)
    cat("  - Frequency of 'alpha' parameter usage:\n")
    print(param_table)
    cat(paste0("\n>>> Recommended parameter for the final model: alpha = ", best_param, "\n"))
    cat("    You can use this alpha value with cv.glmnet() on your full dataset to train the final model. It will automatically find the best lambda for you.\n")
  }
} else if (best_model_name %in% c("GBLUP", "ssGBLUP")) {
  cat("The GBLUP and ssGBLUP models were fitted using REML in the 'sommer' package.\n")
  cat("Variance components were estimated automatically in each cross-validation fold, so there are no external hyperparameters to report.\n")
} else if (best_model_name == "MLP") {
  if (all(c("mlp_neurons", "mlp_dropout") %in% names(best_model_results))) {
    neurons_table <- table(best_model_results$mlp_neurons)
    best_neurons <- get_mode(best_model_results$mlp_neurons)
    cat("  - Frequency of 'mlp_neurons' parameter usage:\n")
    print(neurons_table)
    
    dropout_table <- table(best_model_results$mlp_dropout)
    best_dropout <- get_mode(best_model_results$mlp_dropout)
    cat("\n  - Frequency of 'mlp_dropout' parameter usage:\n")
    print(dropout_table)
    
    cat(paste0("\n>>> Recommended structure for the final MLP model: neurons = ", best_neurons, ", dropout_rate = ", best_dropout, "\n"))
  }
} else if (best_model_name == "CNN") {
  if (all(c("cnn_filters", "cnn_kernel_size") %in% names(best_model_results))) {
    filters_table <- table(best_model_results$cnn_filters)
    best_filters <- get_mode(best_model_results$cnn_filters)
    cat("  - Frequency of 'cnn_filters' parameter usage:\n")
    print(filters_table)
    
    kernel_size_table <- table(best_model_results$cnn_kernel_size)
    best_kernel_size <- get_mode(best_model_results$cnn_kernel_size)
    cat("\n  - Frequency of 'cnn_kernel_size' parameter usage:\n")
    print(kernel_size_table)
    
    cat(paste0("\n>>> Recommended structure for the final CNN model: filters = ", best_filters, ", kernel_size = ", best_kernel_size, "\n"))
  }
} else if (best_model_name == "AD-GBLUP") {
  cat("The AD-GBLUP model automatically estimates additive and dominance variances in each run using REML.\n")
  cat("Here are the average variance components across all successful runs:\n")
  avg_var_a <- mean(best_model_results$varA, na.rm=TRUE)
  avg_var_d <- mean(best_model_results$varD, na.rm=TRUE)
  cat(paste0("  - Average Additive Variance (Va): ", round(avg_var_a, 4), "\n"))
  cat(paste0("  - Average Dominance Variance (Vd): ", round(avg_var_d, 4), "\n"))
  cat("\nTo train a final model, run the 'sommer' model on your full dataset. It will provide the final variance component estimates and GEBVs.\n")
} else {
  cat("This model does not have external hyperparameters to tune in this pipeline.\n")
}
cat("\nAnalysis complete. You now have both a model performance evaluation and recommended parameters for final model training.\n")
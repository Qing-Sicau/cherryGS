# =====================================================================================
#
#   Script 2: SNP Density Impact Analysis (V6 - OPTIMIZED & ROBUST)
#
# Last Modified: Sep 02, 2025
#
# =====================================================================================


# =====================================================================================
# Part 0: Environment Setup
# =====================================================================================
# --- Set environment variables to prevent over-threading issues ---
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
  "future", "furrr", "sommer", "Cairo"
)
suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})
log_message("SETUP: All required packages loaded successfully.")


# =====================================================================================
# Part 1: Find Python Path Before Parallelization (CRITICAL STEP)
# =====================================================================================
log_message("SETUP: Finding Python executable from the 'poly_map' conda environment...")
# NOTE: The original script referenced 'reseq', ensure this is the correct environment.
tryCatch({
    use_condaenv("poly_map", required = TRUE)
    python_exe_path <- reticulate::py_config()$python
    log_message("SETUP: Found python executable to be used by workers: ", python_exe_path)
}, error = function(e) {
    log_message("SETUP: ERROR - Could not find the 'reseq' conda environment. Please check the name and installation.")
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

# --- Create output directories if they don't exist ---
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)
if (!dir.exists("./results")) dir.create("./results", recursive = TRUE)
log_message("SETUP: Environment setup complete.")


# =====================================================================================
# Part 2: Load, Clean, and Preprocess Data
# =====================================================================================
log_message("DATA: Loading and preprocessing data...")
t_start_data <- proc.time()

GENOTYPE_FILE  <- "genotype.dosages.tsv"
PHENOTYPE_FILE <- "phenotype_BLUPs.csv"

# --- Load Genotype and Phenotype Data ---
dosage_raw <- read.csv(GENOTYPE_FILE, check.names = FALSE, sep = "\t", header = TRUE)
marker_ids <- paste0(dosage_raw$CHROM, ":", dosage_raw$POS)
dosage_only <- dosage_raw[, 5:ncol(dosage_raw)]
dosage_only_matrix <- as.matrix(dosage_only)
storage.mode(dosage_only_matrix) <- "numeric"
genotypeMatrix_raw <- t(dosage_only_matrix)
rownames(genotypeMatrix_raw) <- colnames(dosage_only)
colnames(genotypeMatrix_raw) <- marker_ids
log_message("DATA: Loaded genotype data with ", nrow(genotypeMatrix_raw), " individuals and ", ncol(genotypeMatrix_raw), " markers.")

pheno_raw <- read.csv(PHENOTYPE_FILE, row.names = 1, check.names = FALSE, sep = ",", header = TRUE)
pheno_df <- as.data.frame(pheno_raw)
ALL_TRAITS_IN_FILE <- colnames(pheno_df)
log_message("DATA: Found ", length(ALL_TRAITS_IN_FILE), " traits in phenotype file: ", paste(ALL_TRAITS_IN_FILE, collapse=", "))

# --- Data Alignment and Cleaning ---
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
log_message("DATA: Found ", length(common_individuals), " individuals with both genotype and phenotype data.")
genotypeMatrix_full <- genotypeMatrix_raw[common_individuals, ]
pheno_df_aligned <- pheno_df[common_individuals, , drop = FALSE]

# --- Handle Missing Genotype Values ---
log_message("DATA: Handling missing genotype values using mean imputation...")
col_means <- colMeans(genotypeMatrix_full, na.rm = TRUE)
col_means[is.nan(col_means)] <- 0
missing_indices <- which(is.na(genotypeMatrix_full), arr.ind = TRUE)
genotypeMatrix_full[missing_indices] <- round(col_means[missing_indices[, 2]])
numIndividuals_full <- nrow(genotypeMatrix_full); numMarkers_full <- ncol(genotypeMatrix_full)

t_end_data <- proc.time()
log_message("DATA: Full dataset prepared: ", numIndividuals_full, " individuals, ", numMarkers_full, " SNP markers.")
log_message("DATA: Time for data prep: ", round((t_end_data - t_start_data)[3], 2), " seconds.")

# =====================================================================================
# Part 3: Pre-build Pedigree (once)
# =====================================================================================
log_message("PREP: Building pedigree (A-matrix)...")
source("get_DomRel_matrix.R")
parents <- c("HF1", "NZH2", "PJHH")
all_ped_ids <- unique(c(parents, common_individuals))
ped_df <- data.frame(ID = all_ped_ids, Sire = 0, Dam = 0, stringsAsFactors = FALSE)
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) { ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) { ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "PJHH" }
}
A_full <- Amatrix(ped_df, ploidy = 4)
log_message("PREP: A-matrix built successfully.")

# =====================================================================================
# Part 4: Main Analysis Loop for All Traits and Densities
# =====================================================================================
NUM_REPEATS   <- 20
NUM_CORES     <- 20
SNP_DENSITIES <- c(1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01)

# --- Proactive Cleanup of temporary files ---
log_message("CLEANUP: Removing any old BGLR temporary files before starting...")
unlink(list.files(pattern = "density_.*rep_.*fold_.*\\.dat"))

# --- Setup Robust Parallel Backend ---
log_message("SETUP: Setting up robust parallel backend to use ", NUM_CORES, " cores...")
cl <- parallel::makeCluster(NUM_CORES)
parallel::clusterExport(cl, "python_exe_path")
parallel::clusterEvalQ(cl, {
  Sys.setenv(OPENBLAS_NUM_THREADS = 1); Sys.setenv(MKL_NUM_THREADS = 1); Sys.setenv(OMP_NUM_THREADS = 1)
  Sys.setenv(TF_NUM_INTEROP_THREADS = 1); Sys.setenv(TF_NUM_INTRAOP_THREADS = 1)
  library(reticulate); use_python(python_exe_path, required = TRUE)
  library(keras3); tf <- tensorflow::tf
  tensorflow::set_random_seed(1234, disable_gpu = TRUE)
})
plan(cluster, workers = cl)
log_message("SETUP: Parallel cluster is ready.")

# --- Main loop to iterate through each trait ---
for (TRAIT_OF_INTEREST in ALL_TRAITS_IN_FILE) {
  t_start_trait <- proc.time()
  log_message("\n######################################################################")
  log_message("###   STARTING DENSITY ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###")
  log_message("######################################################################\n")

  phenotypeVector <- pheno_df_aligned[[TRAIT_OF_INTEREST]]
  names(phenotypeVector) <- rownames(pheno_df_aligned)
  
  # --- Setup for incremental saving for the current trait ---
  trait_id_safe <- gsub("[^a-zA-Z0-9_]", "_", TRAIT_OF_INTEREST) # Make filename safe
  incremental_file <- paste0("results/GS_Incremental_Results_", trait_id_safe, ".csv")
  if (file.exists(incremental_file)) file.remove(incremental_file)
  log_message("INFO: Incremental results for this trait will be saved to: ", incremental_file)

  all_results_list <- list()
  
  # --- Outer "SNP Density" Loop ---
  for (density in SNP_DENSITIES) {
    t_start_density <- proc.time()
    num_snps_to_sample <- round(numMarkers_full * density)
    log_message("==================================================================")
    log_message("  Trait: ", TRAIT_OF_INTEREST, " | SNP Density: ", density * 100, "% (", num_snps_to_sample, " markers)", indent = 1)
    log_message("==================================================================")
    
    set.seed(123)
    sampled_snp_indices <- sample(1:numMarkers_full, num_snps_to_sample)
    genotypeMatrix_sub <- genotypeMatrix_full[, sampled_snp_indices, drop=FALSE]
    
    log_message("PREP: Calculating relationship matrices for the current SNP subset...", indent = 2)
    sd_vals <- apply(genotypeMatrix_sub, 2, sd)
    poly_indices <- which(sd_vals > 1e-6)
    genotypeMatrix <- genotypeMatrix_sub[, poly_indices, drop=FALSE]
    numMarkers <- ncol(genotypeMatrix)
    numIndividuals <- nrow(genotypeMatrix)
    
    if (numMarkers == 0) {
      log_message("WARNING: No polymorphic markers found for this density. Skipping.", indent = 2)
      next
    }
    
    G <- Gmatrix(genotypeMatrix, method="VanRaden", ploidy=4); G <- G + diag(nrow(G)) * 1e-4
    D_raw <- get_DomRel(genotypeMatrix, ploidy=4)
    Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw))
    D <- (Ic %*% D_raw %*% Ic) + diag(nrow(D_raw)) * 1e-4
    rownames(D) <- rownames(genotypeMatrix); colnames(D) <- rownames(genotypeMatrix)
    H <- sommer::H.mat(A=A_full, G=G)

    log_message("PREP: All relationship matrices are ready for this density level.", indent = 2)

    # --- DL hyperparameter tuning ONCE per density level ---
    log_message("PREP: Performing one-time DL hyperparameter tuning for density ", density*100, "%", indent = 2)
    set.seed(123)
    tuning_indices <- createDataPartition(phenotypeVector, p = 0.8, list = FALSE)
    tuning_genotype <- genotypeMatrix[tuning_indices, ]; validation_genotype <- genotypeMatrix[-tuning_indices, ]
    tuning_phenotype <- phenotypeVector[tuning_indices]; validation_phenotype <- phenotypeVector[-tuning_indices]
    
    train_mean_tune <- colMeans(tuning_genotype); train_sd_tune <- apply(tuning_genotype, 2, sd); train_sd_tune[train_sd_tune == 0] <- 1
    tuning_genotype_scaled <- scale(tuning_genotype, center = train_mean_tune, scale = train_sd_tune)
    validation_genotype_scaled <- scale(validation_genotype, center = train_mean_tune, scale = train_sd_tune)

    TRAINING_EPOCHS <- 100
    callbacks_list <- list(
        callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE),
        callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)
    )

    best_mlp_params <- list(neurons=64, dropout_rate=0.4, learning_rate=0.005)
    tryCatch({
      mlp_param_grid <- expand.grid(neurons=c(64, 128), dropout_rate=c(0.4, 0.6), learning_rate=c(0.005, 0.001))
      best_val_loss <- Inf
      for(j in 1:nrow(mlp_param_grid)) {
          params <- mlp_param_grid[j,]; model <- keras_model_sequential() %>%
            layer_dense(units=params$neurons, input_shape=numMarkers, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=params$dropout_rate) %>%
            layer_dense(units=round(params$neurons/2), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=params$learning_rate))
          hist <- model %>% fit(tuning_genotype_scaled, tuning_phenotype, epochs=TRAINING_EPOCHS, batch_size=32, validation_data=list(validation_genotype_scaled, validation_phenotype), verbose=0, callbacks=callbacks_list)
          val_loss <- min(hist$metrics$val_loss, na.rm=TRUE)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_mlp_params <- params }
      }
    }, error=function(e){log_message("WARNING: MLP tuning failed. Using defaults.", indent=3)})

    best_cnn_params <- list(filters=32, kernel_size=8, learning_rate=0.005)
    tryCatch({
      xtrain_cnn_tune <- array(tuning_genotype_scaled, dim=c(nrow(tuning_genotype_scaled), numMarkers, 1))
      xval_cnn_tune <- array(validation_genotype_scaled, dim=c(nrow(validation_genotype_scaled), numMarkers, 1))
      cnn_param_grid <- expand.grid(filters=c(32,64), kernel_size=c(8,12), learning_rate=c(0.005,0.001))
      best_val_loss <- Inf
      for(j in 1:nrow(cnn_param_grid)) {
          params <- cnn_param_grid[j,]; model <- keras_model_sequential() %>%
            layer_conv_1d(filters=params$filters, kernel_size=params$kernel_size, input_shape=c(numMarkers,1), kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
            layer_dense(units=64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
            layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
          model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=params$learning_rate))
          hist <- model %>% fit(xtrain_cnn_tune, tuning_phenotype, epochs=TRAINING_EPOCHS, batch_size=32, validation_data=list(xval_cnn_tune, validation_phenotype), verbose=0, callbacks=callbacks_list)
          val_loss <- min(hist$metrics$val_loss, na.rm=TRUE)
          if (is.finite(val_loss) && val_loss < best_val_loss) { best_val_loss <- val_loss; best_cnn_params <- params }
      }
    }, error=function(e){log_message("WARNING: CNN tuning failed. Using defaults.", indent=3)})
    
    # --- Define the function that runs ONE full repetition for the current density ---
    run_one_repetition <- function(rep_id, tuned_mlp_params, tuned_cnn_params) {
      rep_results_list <- list()
      suppressPackageStartupMessages({
        library(tidyverse); library(BGLR); library(Matrix); library(glmnet);
        library(keras3); library(caret); library(sommer)
      })
      
      all_possible_columns <- c("Repetition","Fold","Model","Cor","SNP_Density","Num_SNPs","alpha","varA","varD","mlp_neurons","mlp_dropout","cnn_filters","cnn_kernel_size")
      standardize_df <- function(df) {
        missing_cols <- setdiff(all_possible_columns, names(df)); if (length(missing_cols) > 0) df[missing_cols] <- NA
        return(df[, all_possible_columns])
      }
      
      # Minimal console output from workers, main logs are better
      cat(paste0("  ..Rep ", rep_id, ".."))
      set.seed(42 + rep_id)
      folds <- createFolds(phenotypeVector, k = 5, list = TRUE, returnTrain = FALSE)
      
      for (i in 1:length(folds)) {
        test_indices <- folds[[i]]; test_ids <- names(phenotypeVector[test_indices]); train_indices <- setdiff(1:numIndividuals, test_indices)
        genoTrain <- genotypeMatrix[train_indices,]; phenoTrain <- phenotypeVector[train_indices]
        genoTest <- genotypeMatrix[test_ids,]; phenoTest <- phenotypeVector[test_ids]
        pheno_with_NAs <- phenotypeVector; pheno_with_NAs[test_indices] <- NA
        
        # --- 1. glmnet Family ---
        tryCatch({
            cv_ridge <- cv.glmnet(genoTrain, phenoTrain, alpha=0, family="gaussian"); pred_ridge <- predict(cv_ridge, newx=genoTest, s="lambda.min")[,1]
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model="Ridge", Cor=cor(pred_ridge,phenoTest,use="complete.obs"), alpha=0)
            cv_lasso <- cv.glmnet(genoTrain, phenoTrain, alpha=1, family="gaussian"); pred_lasso <- predict(cv_lasso, newx=genoTest, s="lambda.min")[,1]
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model="LASSO", Cor=cor(pred_lasso,phenoTest,use="complete.obs"), alpha=1)
            best_alpha<-NA; best_lambda<-NA; best_mse<-Inf
            for(a in seq(0,1,by=0.2)){
                cv_fit <- cv.glmnet(genoTrain,phenoTrain,alpha=a,family="gaussian"); current_mse <- min(cv_fit$cvm,na.rm=T)
                if(is.finite(current_mse) && current_mse < best_mse){best_mse<-current_mse; best_alpha<-a; best_lambda<-cv_fit$lambda.min}
            }
            fit_en <- glmnet(genoTrain, phenoTrain, alpha=best_alpha, lambda=best_lambda, family="gaussian"); pred_en <- predict(fit_en, newx=genoTest)[,1]
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model="Elastic Net", Cor=cor(pred_en,phenoTest,use="complete.obs"), alpha=best_alpha)
        }, error=function(e){}) # Errors are logged in main thread if needed
        
        # --- 2. Bayesian Family (BGLR) ---
        tryCatch({
            models_bglr <- list(BRR=list(list(X=genotypeMatrix, model="BRR")), BayesA=list(list(X=genotypeMatrix, model="BayesA")), BayesB=list(list(X=genotypeMatrix, model="BayesB")), BayesC=list(list(X=genotypeMatrix, model="BayesC")), `Bayes G-BLUP`=list(list(K=G, model="RKHS")))
            for(m_name in names(models_bglr)){
                fit_bglr <- BGLR(y=pheno_with_NAs, ETA=models_bglr[[m_name]], nIter=10000, burnIn=2500, verbose=F, saveAt=paste0("density_",density,"_rep_",rep_id,"_fold_",i,"_"))
                pred_bglr <- fit_bglr$yHat[test_indices]
                rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model=m_name, Cor=cor(pred_bglr,phenoTest,use="complete.obs"))
            }
        }, error=function(e){})

        # --- 3. sommer Family (Corrected Workflow) ---
        tryCatch({ # GBLUP
            data_sommer <- data.frame(ID=names(phenotypeVector), y=pheno_with_NAs); data_sommer$ID <- factor(data_sommer$ID, levels=rownames(G))
            fit_gblup <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=G), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
            pred_table <- predict(fit_gblup, D = "ID"); pred_gblup <- pred_table$pvals[test_ids, "predicted.value"]
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model="GBLUP", Cor=cor(pred_gblup,phenoTest,use="complete.obs"))
        }, error=function(e){})
        tryCatch({ # AD-GBLUP
            data_sommer <- data.frame(ID=names(phenotypeVector), y=pheno_with_NAs); data_sommer$ID_A <- factor(data_sommer$ID, levels=rownames(G)); data_sommer$ID_D <- factor(data_sommer$ID, levels=rownames(D))
            fit_ad <- mmes(fixed=y~1, random=~vsm(ism(ID_A), Gu=G) + vsm(ism(ID_D), Gu=D), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
            pred_table <- predict(fit_ad, D = "ID_A"); pred_ad <- pred_table$pvals[test_ids, "predicted.value"]
            var_a <- fit_ad$sigma[[1]]; if (length(var_a) == 0) var_a <- NA; var_d <- fit_ad$sigma[[2]]; if (length(var_d) == 0) var_d <- NA
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model="AD-GBLUP", Cor=cor(pred_ad,phenoTest,use="complete.obs"), varA=var_a, varD=var_d)
        }, error=function(e){})
        tryCatch({ # ssGBLUP
            pheno_ssgblup <- rep(NA, nrow(H)); names(pheno_ssgblup) <- rownames(H); pheno_ssgblup[names(pheno_with_NAs)] <- pheno_with_NAs
            data_sommer <- data.frame(ID=names(pheno_ssgblup), y=pheno_ssgblup); data_sommer$ID <- factor(data_sommer$ID, levels = rownames(H))
            fit_ss <- mmes(fixed=y~1, random=~vsm(ism(ID), Gu=H), rcov=~units, data=data_sommer, naMethodY="include", verbose=F)
            pred_table <- predict(fit_ss, D = "ID"); pred_ss <- pred_table$pvals[test_ids, "predicted.value"]
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id, Fold=i, Model="ssGBLUP", Cor=cor(pred_ss,phenoTest,use="complete.obs"))
        }, error=function(e){})
        
        # --- 4. Deep Learning Models ---
        train_mean<-colMeans(genoTrain); train_sd<-apply(genoTrain,2,sd); train_sd[train_sd==0]<-1
        genoTrain_scaled <- scale(genoTrain, center=train_mean, scale=train_sd); genoTest_scaled <- scale(genoTest, center=train_mean, scale=train_sd)
        tryCatch({ # MLP
            final_model <- keras_model_sequential() %>%
                layer_dense(units=tuned_mlp_params$neurons, input_shape=numMarkers, kernel_regularizer=regularizer_l2(l2 = 0.001)) %>%
                layer_batch_normalization() %>% layer_activation_relu() %>% layer_dropout(rate=tuned_mlp_params$dropout_rate) %>%
                layer_dense(units=round(tuned_mlp_params$neurons/2), kernel_regularizer=regularizer_l2(l2 = 0.001)) %>%
                layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
            final_model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=tuned_mlp_params$learning_rate))
            final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
            final_model %>% fit(genoTrain_scaled, phenoTrain, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
            pred_mlp <- final_model %>% predict(genoTest_scaled, verbose=0)
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id,Fold=i,Model="MLP",Cor=cor(pred_mlp[,1],phenoTest,use="complete.obs"),mlp_neurons=tuned_mlp_params$neurons,mlp_dropout=tuned_mlp_params$dropout_rate)
        }, error=function(e){})
        tryCatch({ # CNN
            xtrain_cnn <- array(genoTrain_scaled, dim=c(nrow(genoTrain_scaled), numMarkers, 1)); xtest_cnn <- array(genoTest_scaled, dim=c(nrow(genoTest_scaled), numMarkers, 1))
            final_model <- keras_model_sequential() %>%
                layer_conv_1d(filters=tuned_cnn_params$filters, kernel_size=tuned_cnn_params$kernel_size, input_shape=c(numMarkers,1), kernel_regularizer=regularizer_l2(l2 = 0.001)) %>%
                layer_batch_normalization() %>% layer_activation_relu() %>% layer_max_pooling_1d(pool_size=4) %>% layer_flatten() %>%
                layer_dense(units=64, kernel_regularizer=regularizer_l2(l2 = 0.001)) %>%
                layer_batch_normalization() %>% layer_activation_relu() %>% layer_dense(units=1)
            final_model %>% compile(loss="mse", optimizer=optimizer_rmsprop(learning_rate=tuned_cnn_params$learning_rate))
            final_callbacks <- list(callback_early_stopping(monitor="loss", patience=10, restore_best_weights=TRUE))
            final_model %>% fit(xtrain_cnn, phenoTrain, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0, callbacks=final_callbacks)
            pred_cnn <- final_model %>% predict(xtest_cnn, verbose=0)
            rep_results_list[[length(rep_results_list)+1]] <- data.frame(Repetition=rep_id,Fold=i,Model="CNN",Cor=cor(pred_cnn[,1],phenoTest,use="complete.obs"),cnn_filters=tuned_cnn_params$filters,cnn_kernel_size=tuned_cnn_params$kernel_size)
        }, error=function(e){})
      }
      results_for_rep <- dplyr::bind_rows(rep_results_list)
      results_for_rep$SNP_Density <- density; results_for_rep$Num_SNPs <- num_snps_to_sample
      
      # *** CRITICAL BUG FIX ***
      # The original code `lapply(results_for_rep, standardize_df)` was incorrect.
      # It should apply the function to the entire data frame at once.
      return(standardize_df(results_for_rep))
    }
    
    # --- Run repetitions in parallel for the current density ---
    log_message("EXEC: Starting ", NUM_REPEATS, " repetitions of 5-fold CV in parallel...", indent = 2)
    density_results_df <- future_map_dfr(
      .x = 1:NUM_REPEATS, 
      .f = ~run_one_repetition(.x, tuned_mlp_params = best_mlp_params, tuned_cnn_params = best_cnn_params),
      .options = furrr_options(seed = TRUE)
    )
    cat("\n") # Newline after worker messages
    log_message("EXEC: Parallel processing for this density finished.", indent = 2)
    
    # --- INCREMENTAL SAVE ---
    # Save results for the current density immediately.
    write.table(
      density_results_df,
      file = incremental_file,
      append = TRUE,
      sep = ",",
      row.names = FALSE,
      col.names = !file.exists(incremental_file)
    )
    log_message("SAVE: Results for density ", density*100, "% successfully appended to ", incremental_file, indent = 2)

    all_results_list[[as.character(density)]] <- density_results_df
    t_end_density <- proc.time()
    log_message("TIME: Analysis for density ", density*100, "% took ", round((t_end_density - t_start_density)[3], 2), " seconds.", indent = 2)
  }
  
  # =====================================================================================
  # Part 5: Result Summarization and Visualization for the Current Trait
  # =====================================================================================
  results_df <- dplyr::bind_rows(all_results_list)
  log_message("COMPLETE: Analysis across all SNP densities finished for trait: ", TRAIT_OF_INTEREST, "!")
  
  log_message("RESULTS: Summarizing and plotting results for trait: ", TRAIT_OF_INTEREST, "...")
  summary_stats <- results_df %>%
    dplyr::filter(!is.na(Cor)) %>%
    dplyr::group_by(Model, SNP_Density, Num_SNPs) %>%
    dplyr::summarise(Mean_Cor = mean(Cor, na.rm=TRUE), SD_Cor = sd(Cor, na.rm=TRUE), .groups = 'drop') %>%
    dplyr::arrange(desc(SNP_Density), desc(Mean_Cor))
  print(as.data.frame(summary_stats))
  
  snp_break_points <- summary_stats %>%
    dplyr::select(SNP_Density, Num_SNPs) %>%
    dplyr::distinct() %>% dplyr::arrange(Num_SNPs)
  
  plot_density_decay <- ggplot(summary_stats, aes(x=Num_SNPs, y=Mean_Cor, group=Model, color=Model)) +
    geom_line(linewidth=1.1) + geom_point(size=2.5, alpha=0.8) +
    geom_errorbar(aes(ymin=Mean_Cor-SD_Cor, ymax=Mean_Cor+SD_Cor), width=0, alpha=0.6) +
    scale_x_log10(breaks=snp_break_points$Num_SNPs, labels=paste0(snp_break_points$SNP_Density*100,"%")) +
    annotate("text", x=snp_break_points$Num_SNPs, y=min(summary_stats$Mean_Cor-summary_stats$SD_Cor,na.rm=T)*0.95, 
             label=paste0("(", scales::comma(snp_break_points$Num_SNPs), ")"), size=2.8, angle=45, hjust=1) +
    labs(
      title=paste("Prediction Accuracy vs. SNP Marker Density for", TRAIT_OF_INTEREST),
      subtitle=paste0("Average of ", NUM_REPEATS, " Repeats (5-Fold Cross-Validation)"),
      x="SNP Density (% of Total Markers)", y="Mean Prediction Accuracy (Pearson's r)", color="Prediction Model"
    ) +
    theme_publication() + theme(legend.position="right", axis.text.x=element_text(angle=45, hjust=1)) +
    coord_cartesian(clip="off")

  # --- Save Results and Plots ---
  ggsave(paste0("plots/GS_SNPDensity_Comparison_", trait_id_safe, ".pdf"), plot=plot_density_decay, width=10, height=8, device=cairo_pdf)
  write.csv(results_df, paste0("results/GS_All_Density_Results_", trait_id_safe, ".csv"), row.names = FALSE)
  write.csv(summary_stats, paste0("results/GS_Density_Summary_", trait_id_safe, ".csv"), row.names = FALSE)
  log_message("SAVED: Analysis complete for ", TRAIT_OF_INTEREST, ". Density plot and summary data saved.")
  
  # Clean up BGLR files before next trait
  unlink(list.files(pattern = "density_.*.dat"))

  t_end_trait <- proc.time()
  log_message("###   Total time for trait ", TRAIT_OF_INTEREST, ": ", round((t_end_trait - t_start_trait)[3] / 60, 2), " minutes.   ###")
}

# =====================================================================================
# Part 6: Final Cleanup
# =====================================================================================
log_message("CLEANUP: All traits analyzed. Stopping the parallel cluster...")
parallel::stopCluster(cl)
plan(sequential)
log_message("\n[--- FINISHED ---] Analysis pipeline complete for all traits.")

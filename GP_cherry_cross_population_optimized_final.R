# =====================================================================================
#
#   Script 3: Cross-Population Genomic Prediction (V2 - Automated Multi-Trait)
#
# Description:
#   This script evaluates cross-population genomic prediction accuracy. It has been
#   upgraded to automatically iterate through all traits in the phenotype file.
#   For each trait, it performs both cross-population scenarios, summarizes the
#   results, and saves outputs systematically named by the trait.
#
# Last Modified: Aug 22, 2025
#
# =====================================================================================


# =====================================================================================
# Part 0: Environment Setup
# =====================================================================================
Sys.setenv(OPENBLAS_NUM_THREADS = 1); Sys.setenv(MKL_NUM_THREADS = 1); Sys.setenv(OMP_NUM_THREADS = 1)

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
# Part 1: Find Python Path Before Parallelization (CRITICAL STEP)
# =====================================================================================
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

# Create output directories if they don't exist
if (!dir.exists("./plots")) dir.create("./plots", recursive = TRUE)
if (!dir.exists("./results")) dir.create("./results", recursive = TRUE)
cat("[SETUP] Environment setup complete.\n")


# =====================================================================================
# Part 2: Load, Clean, and Preprocess Data
# =====================================================================================
cat("\n[DATA] Loading and preprocessing data...\n")

# --- User Configuration ---
GENOTYPE_FILE  <- "genotype.dosages.tsv"
PHENOTYPE_FILE <- "phenotype_BLUPs.csv"

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
ALL_TRAITS_IN_FILE <- colnames(pheno_df) # UPGRADE: Identify all available traits
cat(paste0("[DATA] Found ", length(ALL_TRAITS_IN_FILE), " traits in phenotype file: ", paste(ALL_TRAITS_IN_FILE, collapse=", "), "\n"))

# --- Data Alignment and Cleaning ---
common_individuals <- intersect(rownames(genotypeMatrix_raw), rownames(pheno_df))
cat(paste0("[DATA] Found ", length(common_individuals), " individuals with both genotype and phenotype data.\n"))
genotypeMatrix <- genotypeMatrix_raw[common_individuals, ]
pheno_df_aligned <- pheno_df[common_individuals, , drop = FALSE]

# --- Handle Missing Genotype Values (Mean Imputation) ---
cat("[DATA] Handling missing genotype values using mean imputation...\n")
impute_mean <- function(x) {
  mean_val <- mean(x, na.rm = TRUE)
  if (is.nan(mean_val)) mean_val <- 0
  x[is.na(x)] <- round(mean_val)
  return(x)
}
genotypeMatrix <- apply(genotypeMatrix, 2, impute_mean)
numIndividuals <- nrow(genotypeMatrix); numMarkers <- ncol(genotypeMatrix)
cat(paste0("[DATA] Data preparation complete: ", numIndividuals, " individuals, ", numMarkers, " SNP markers.\n"))

# =====================================================================================
# Part 3: Split Data into Defined Populations
# =====================================================================================
cat("\n[DATA] Splitting data into specified populations based on ID prefixes...\n")
all_ids <- rownames(genotypeMatrix)
pop_NH_HN_ids <- all_ids[startsWith(all_ids, "NH") | startsWith(all_ids, "HN")]
pop_HP_ids <- all_ids[startsWith(all_ids, "HP")]
cat(paste0("  - Population 1 (NH/HN): ", length(pop_NH_HN_ids), " individuals.\n"))
cat(paste0("  - Population 2 (HP): ", length(pop_HP_ids), " individuals.\n"))
if(length(pop_NH_HN_ids) == 0 | length(pop_HP_ids) == 0) {
  stop("One or both defined populations have zero individuals.")
}

# =====================================================================================
# Part 4: Population Structure Analysis via PCA (Trait-Independent)
# =====================================================================================
cat("\n[PCA] Performing Principal Component Analysis to visualize population structure...\n")
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
# UPGRADE: Save the PCA plot (it's the same for all traits)
ggsave("plots/PCA_Plot_CrossPop_Structure.pdf", pca_plot, width = 8, height = 6, device = "cairo_pdf")
cat("[PCA] PCA plot saved as 'plots/PCA_Plot_CrossPop_Structure.pdf'.\n")

# =====================================================================================
# Part 5: Pre-calculate Relationship Matrices and Pedigree
# =====================================================================================
cat("\n[PREP] Building relationship matrices for the ENTIRE population...\n")
source("get_DomRel_matrix.R")
parents <- c("HF1", "NZH2", "PJHH")
all_ped_ids <- unique(c(parents, all_ids))
ped_df <- data.frame(ID = all_ped_ids, Sire = 0, Dam = 0, stringsAsFactors = FALSE)
for (i in 1:nrow(ped_df)) {
  id <- ped_df$ID[i]
  if (startsWith(id, "NH") || startsWith(id, "HN")) { ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "NZH2"
  } else if (startsWith(id, "HP")) { ped_df$Sire[i] <- "HF1"; ped_df$Dam[i] <- "PJHH" }
}
A_full <- Amatrix(ped_df, ploidy = 4)
cat("[PREP] Full A-matrix (pedigree) built.\n")
G_full <- Gmatrix(genotypeMatrix, method = "VanRaden", ploidy = 4)
G_full <- G_full + diag(nrow(G_full)) * 1e-4
cat("[PREP] Full G-matrix (additive) built.\n")
D_raw <- get_DomRel(genotypeMatrix, ploidy = 4)
Ic <- diag(nrow(D_raw)) - (1/nrow(D_raw)) * matrix(1, nrow(D_raw), nrow(D_raw))
D_full <- (Ic %*% D_raw %*% Ic) + diag(nrow(D_raw)) * 1e-4
rownames(D_full) <- rownames(genotypeMatrix); colnames(D_full) <- rownames(genotypeMatrix)
cat("[PREP] Full D-matrix (dominance) built.\n")
doH_inverse <- function(pedigreeRelationshipMatrix, grmForGenotyped) {
  genotypedIndicesInPedigree <- match(rownames(grmForGenotyped), rownames(pedigreeRelationshipMatrix))
  grmInverse <- solve(grmForGenotyped); A22 <- pedigreeRelationshipMatrix[genotypedIndicesInPedigree, genotypedIndicesInPedigree]
  pedigreeRelationshipInverseForGenotyped <- solve(A22); pedigreeRelationshipInverse <- solve(pedigreeRelationshipMatrix)
  hMatrixInverse <- pedigreeRelationshipInverse
  hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] <- hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] + grmInverse - pedigreeRelationshipInverseForGenotyped
  attr(hMatrixInverse, 'inverse') <- TRUE; return(hMatrixInverse)
}
Hinv_full <- doH_inverse(A_full, G_full)
cat("[PREP] Full H-inverse matrix built.\n")

# =====================================================================================
# Part 6: Robust Parallel Backend Setup
# =====================================================================================
CV_REPETITIONS   <- 3; NUM_CORES_TO_USE <- 3; PRED_SAMPLES <- 3; PRED_SAMPLE_FRAC <- 0.8
cat(paste("\n[PARALLEL] Setting up robust parallel cluster with", NUM_CORES_TO_USE, "cores...\n"))
cl <- parallel::makeCluster(NUM_CORES_TO_USE)
parallel::clusterExport(cl, "python_exe_path")
parallel::clusterEvalQ(cl, {
  Sys.setenv(OPENBLAS_NUM_THREADS=1); Sys.setenv(MKL_NUM_THREADS=1); Sys.setenv(OMP_NUM_THREADS=1)
  library(reticulate); use_python(python_exe_path, required=TRUE); library(keras); tf <- tensorflow::tf
  tf$config$threading$set_inter_op_parallelism_threads(1L); tf$config$threading$set_intra_op_parallelism_threads(1L)
})
plan(cluster, workers = cl)
cat("[PARALLEL] Cluster is ready.\n")

# =====================================================================================
# Part 7: Main Analysis Function for a Single Cross-Population Scenario
# =====================================================================================
# UPGRADE: Function now accepts `phenotypeVector` as an argument
perform_cross_population_prediction <- function(train_ids, pred_ids, train_pop_name, pred_pop_name,
                                                phenotypeVector, # <-- New argument
                                                cv_repeats = CV_REPETITIONS,
                                                pred_samples = PRED_SAMPLES,
                                                pred_sample_frac = PRED_SAMPLE_FRAC) {
  
  cat(paste0("\n----------------------------------------------------------------------\n"))
  cat(paste0("  Scenario: Train on '", train_pop_name, "' -> Predict on '", pred_pop_name, "'\n"))
  cat(paste0("----------------------------------------------------------------------\n"))
  
  # --- Step 1: Find best model via CV within the training population ---
  cat(paste0("[STEP 1] Performing ", cv_repeats, "x 5-fold CV on '", train_pop_name, "'...\n"))
  
  geno_train_pop <- genotypeMatrix[train_ids, ]; pheno_train_pop <- phenotypeVector[train_ids]; n_train_pop <- length(pheno_train_pop)
  G_train_pop <- G_full[train_ids, train_ids]; D_train_pop <- D_full[train_ids, train_ids]
  ped_ids_train <- unique(c(train_ids, parents)); A_train <- A_full[ped_ids_train, ped_ids_train]; Hinv_train <- doH_inverse(A_train, G_train_pop)
  
  run_one_repetition <- function(rep_id) {
    rep_results_list <- list()
    suppressPackageStartupMessages({ library(tidyverse); library(BGLR); library(Matrix); library(glmnet); library(keras); library(caret); library(sommer) })
    all_possible_columns <- c("Repetition","Fold","Model","Cor","alpha","varA","varD","mlp_neurons","mlp_dropout","cnn_filters","cnn_kernel_size")
    standardize_df <- function(df) { missing_cols<-setdiff(all_possible_columns,names(df)); if(length(missing_cols)>0){df[missing_cols]<-NA}; return(df[,all_possible_columns])}
    cat(paste0("  [CV] Repetition ", rep_id, "/", cv_repeats, "...\n")); set.seed(42 + rep_id)
    folds <- createFolds(pheno_train_pop, k=5, list=T, returnTrain=F)
    for(i in 1:length(folds)){
      test_indices_local<-folds[[i]]; train_indices_local<-setdiff(1:n_train_pop,test_indices_local); test_ids_local<-names(pheno_train_pop[test_indices_local])
      genoTrain<-geno_train_pop[train_indices_local,]; phenoTrain<-pheno_train_pop[train_indices_local]; genoTest<-geno_train_pop[test_ids_local,]; phenoTest<-pheno_train_pop[test_ids_local]
      pheno_with_NAs_cv<-pheno_train_pop; pheno_with_NAs_cv[test_indices_local]<-NA
      # Models 1-6 (glmnet, GBLUP, BGLR, ssGBLUP, AD-GBLUP, Deep Learning) - CORE LOGIC UNCHANGED
      tryCatch({cv_ridge<-cv.glmnet(genoTrain,phenoTrain,alpha=0,family="gaussian"); pred_ridge<-predict(cv_ridge,newx=genoTest,s="lambda.min")[,1]; df_ridge<-data.frame(Repetition=rep_id,Fold=i,Model="Ridge",Cor=cor(pred_ridge,phenoTest,use="complete.obs"),alpha=0); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_ridge); cv_lasso<-cv.glmnet(genoTrain,phenoTrain,alpha=1,family="gaussian"); pred_lasso<-predict(cv_lasso,newx=genoTest,s="lambda.min")[,1]; df_lasso<-data.frame(Repetition=rep_id,Fold=i,Model="LASSO",Cor=cor(pred_lasso,phenoTest,use="complete.obs"),alpha=1); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_lasso); best_alpha<-NA;best_lambda<-NA;best_mse<-Inf; for(a in seq(0,1,by=0.2)){cv_fit<-cv.glmnet(genoTrain,phenoTrain,alpha=a,family="gaussian"); current_mse<-min(cv_fit$cvm,na.rm=T); if(is.finite(current_mse)&&current_mse<best_mse){best_mse<-current_mse;best_alpha<-a;best_lambda<-cv_fit$lambda.min}}; fit_en<-glmnet(genoTrain,phenoTrain,alpha=best_alpha,lambda=best_lambda,family="gaussian"); pred_en<-predict(fit_en,newx=genoTest)[,1]; df_en<-data.frame(Repetition=rep_id,Fold=i,Model="Elastic Net",Cor=cor(pred_en,phenoTest,use="complete.obs"),alpha=best_alpha); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_en)},error=function(e){cat(paste0(" ERROR glmnet Rep ",rep_id," F ",i,": ",e$message,"\n"))})
      tryCatch({sommer_data_gblup<-data.frame(ID=names(pheno_train_pop),y=as.numeric(pheno_with_NAs_cv)); sommer_data_gblup$ID<-factor(sommer_data_gblup$ID,levels=rownames(G_train_pop)); fit_gblup<-mmes(fixed=y~1,random=~vsm(ism(ID),Gu=G_train_pop),rcov=~units,data=sommer_data_gblup,naMethodY="include",verbose=F); pred_gblup<-predict(fit_gblup,D="ID")$pvals[test_ids_local,"predicted.value"]; df_gblup<-data.frame(Repetition=rep_id,Fold=i,Model="GBLUP",Cor=cor(pred_gblup,phenoTest,use="complete.obs")); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_gblup)},error=function(e){cat(paste0(" ERROR GBLUP Rep ",rep_id," F ",i,": ",e$message,"\n"))})
      tryCatch({models_bglr<-list(BRR=list(list(X=geno_train_pop,model="BRR")),BayesA=list(list(X=geno_train_pop,model="BayesA")),BayesB=list(list(X=geno_train_pop,model="BayesB")),BayesC=list(list(X=geno_train_pop,model="BayesC")),`Bayes G-BLUP`=list(list(K=G_train_pop,model="RKHS"))); for(m_name in names(models_bglr)){fit_bglr<-BGLR(y=pheno_with_NAs_cv,ETA=models_bglr[[m_name]],nIter=10000,burnIn=2500,verbose=F,saveAt=paste0("rep_",rep_id,"_fold_",i,"_")); pred_bglr<-fit_bglr$yHat[test_indices_local]; df_bglr<-data.frame(Repetition=rep_id,Fold=i,Model=m_name,Cor=cor(pred_bglr,phenoTest,use="complete.obs")); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_bglr)}},error=function(e){cat(paste0(" ERROR BGLR Rep ",rep_id," F ",i,": ",e$message,"\n"))})
      tryCatch({pheno_vec_with_founders<-rep(NA,length(ped_ids_train)); names(pheno_vec_with_founders)<-ped_ids_train; pheno_vec_with_founders[names(pheno_train_pop)]<-pheno_train_pop; pheno_vec_with_founders[test_ids_local]<-NA; sommer_data_ss<-data.frame(ID=names(pheno_vec_with_founders),y=pheno_vec_with_founders); sommer_data_ss$ID<-factor(sommer_data_ss$ID,levels=rownames(Hinv_train)); fit_ssgblup<-mmes(fixed=y~1,random=~vsm(ism(ID),Gu=Hinv_train),rcov=~units,data=sommer_data_ss,naMethodY="include",verbose=F,henderson=T); pred_ssgblup<-predict(fit_ssgblup,D="ID")$pvals[test_ids_local,"predicted.value"]; df_ssgblup<-data.frame(Repetition=rep_id,Fold=i,Model="ssGBLUP",Cor=cor(pred_ssgblup,phenoTest,use="complete.obs")); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_ssgblup)},error=function(e){cat(paste0(" ERROR ssGBLUP Rep ",rep_id," F ",i,": ",e$message,"\n"))})
      tryCatch({sommer_data_ad<-data.frame(ID=names(pheno_train_pop),y=as.numeric(pheno_with_NAs_cv)); sommer_data_ad$ID_A<-factor(sommer_data_ad$ID,levels=rownames(G_train_pop)); sommer_data_ad$ID_D<-factor(sommer_data_ad$ID,levels=rownames(D_train_pop)); fit_ad_gblup<-mmes(fixed=y~1,random=~vsm(ism(ID_A),Gu=G_train_pop)+vsm(ism(ID_D),Gu=D_train_pop),rcov=~units,data=sommer_data_ad,naMethodY="include",verbose=F); pred_ad_gblup<-predict(fit_ad_gblup,D="ID_A")$pvals[test_ids_local,"predicted.value"]; var_a<-fit_ad_gblup$sigma[[1]];var_d<-fit_ad_gblup$sigma[[2]]; if(length(var_a)==0)var_a<-NA; if(length(var_d)==0)var_d<-NA; df_ad<-data.frame(Repetition=rep_id,Fold=i,Model="AD-GBLUP",Cor=cor(pred_ad_gblup,phenoTest,use="complete.obs"),varA=var_a,varD=var_d); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_ad)},error=function(e){cat(paste0(" ERROR AD-GBLUP Rep ",rep_id," F ",i,": ",e$message,"\n"))})
      train_mean<-colMeans(genoTrain);train_sd<-apply(genoTrain,2,sd);train_sd[train_sd==0]<-1; genoTrain_scaled<-scale(genoTrain,center=train_mean,scale=train_sd); genoTest_scaled<-scale(genoTest,center=train_mean,scale=train_sd); TRAINING_EPOCHS<-100; callbacks_list<-list(callback_early_stopping(monitor="val_loss",patience=10,restore_best_weights=T),callback_reduce_lr_on_plateau(monitor="val_loss",factor=0.2,patience=5))
      tryCatch({mlp_param_grid<-expand.grid(neurons=c(64,128),dropout_rate=c(0.4,0.6),learning_rate=c(0.005,0.001)); best_val_loss<-Inf; best_mlp_params<-list(neurons=NA,dropout_rate=NA,learning_rate=NA); for(j in 1:nrow(mlp_param_grid)){params<-mlp_param_grid[j,]; model<-keras_model_sequential()%>%layer_dense(units=params$neurons,input_shape=ncol(genoTrain),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dropout(rate=params$dropout_rate)%>%layer_dense(units=round(params$neurons/2),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dense(units=1); model%>%compile(loss="mse",optimizer=optimizer_adam(learning_rate=params$learning_rate)); hist<-model%>%fit(genoTrain_scaled,phenoTrain,epochs=TRAINING_EPOCHS,batch_size=32,validation_split=0.2,verbose=0,callbacks=callbacks_list); val_loss<-min(hist$metrics$val_loss,na.rm=T); if(is.finite(val_loss)&&val_loss<best_val_loss){best_val_loss<-val_loss;best_mlp_params<-params}}; final_model<-keras_model_sequential()%>%layer_dense(units=best_mlp_params$neurons,input_shape=ncol(genoTrain),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dropout(rate=best_mlp_params$dropout_rate)%>%layer_dense(units=round(best_mlp_params$neurons/2),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dense(units=1); final_model%>%compile(loss="mse",optimizer=optimizer_adam(learning_rate=best_mlp_params$learning_rate)); final_callbacks<-list(callback_early_stopping(monitor="loss",patience=10)); final_model%>%fit(genoTrain_scaled,phenoTrain,epochs=TRAINING_EPOCHS,batch_size=32,verbose=0,callbacks=final_callbacks); pred_mlp<-final_model%>%predict(genoTest_scaled); df_mlp<-data.frame(Repetition=rep_id,Fold=i,Model="MLP",Cor=cor(pred_mlp[,1],phenoTest,use="complete.obs"),mlp_neurons=best_mlp_params$neurons,mlp_dropout=best_mlp_params$dropout_rate); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_mlp)},error=function(e){cat(paste0(" ERROR MLP Rep ",rep_id," F ",i,": ",e$message,"\n"))})
      tryCatch({xtrain_cnn<-array(genoTrain_scaled,dim=c(nrow(genoTrain_scaled),ncol(genoTrain_scaled),1)); xtest_cnn<-array(genoTest_scaled,dim=c(nrow(genoTest_scaled),ncol(genoTest_scaled),1)); cnn_param_grid<-expand.grid(filters=c(32,64),kernel_size=c(8,12),learning_rate=c(0.005,0.001)); best_val_loss<-Inf; best_cnn_params<-list(filters=NA,kernel_size=NA,learning_rate=NA); for(j in 1:nrow(cnn_param_grid)){params<-cnn_param_grid[j,]; model<-keras_model_sequential()%>%layer_conv_1d(filters=params$filters,kernel_size=params$kernel_size,input_shape=c(ncol(genoTrain),1),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_max_pooling_1d(pool_size=4)%>%layer_flatten()%>%layer_dense(units=64,kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dense(units=1); model%>%compile(loss="mse",optimizer=optimizer_rmsprop(learning_rate=params$learning_rate)); hist<-model%>%fit(xtrain_cnn,phenoTrain,epochs=TRAINING_EPOCHS,batch_size=32,validation_split=0.2,verbose=0,callbacks=callbacks_list); val_loss<-min(hist$metrics$val_loss,na.rm=T); if(is.finite(val_loss)&&val_loss<best_val_loss){best_val_loss<-val_loss;best_cnn_params<-params}}; final_model<-keras_model_sequential()%>%layer_conv_1d(filters=best_cnn_params$filters,kernel_size=best_cnn_params$kernel_size,input_shape=c(ncol(genoTrain),1),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_max_pooling_1d(pool_size=4)%>%layer_flatten()%>%layer_dense(units=64,kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dense(units=1); final_model%>%compile(loss="mse",optimizer=optimizer_rmsprop(learning_rate=best_cnn_params$learning_rate)); final_callbacks<-list(callback_early_stopping(monitor="loss",patience=10)); final_model%>%fit(xtrain_cnn,phenoTrain,epochs=TRAINING_EPOCHS,batch_size=32,verbose=0,callbacks=final_callbacks); pred_cnn<-final_model%>%predict(xtest_cnn); df_cnn<-data.frame(Repetition=rep_id,Fold=i,Model="CNN",Cor=cor(pred_cnn[,1],phenoTest,use="complete.obs"),cnn_filters=best_cnn_params$filters,cnn_kernel_size=best_cnn_params$kernel_size); rep_results_list[[length(rep_results_list)+1]]<-standardize_df(df_cnn)},error=function(e){cat(paste0(" ERROR CNN Rep ",rep_id," F ",i,": ",e$message,"\n"))})
    }
    return(dplyr::bind_rows(rep_results_list))
  }
  
  cv_results_df <- future_map_dfr(.x=1:cv_repeats, .f=run_one_repetition, .options=furrr_options(seed=T))
  unlink(list.files(pattern="rep_.*.dat"))
  
  cv_summary_stats <- cv_results_df %>% filter(!is.na(Cor)) %>% group_by(Model) %>% summarise(Mean_Cor=mean(Cor,na.rm=T),SD_Cor=sd(Cor,na.rm=T),.groups='drop') %>% arrange(desc(Mean_Cor))
  cat("\n[INFO] Internal CV performance:\n"); print(cv_summary_stats)
  best_model_name <- cv_summary_stats$Model[1]
  cat(paste0("\n[INFO] Best model for this scenario: '", best_model_name, "' (Mean CV Acc: ", round(cv_summary_stats$Mean_Cor[1], 4), ").\n"))
  
  # --- Step 2: Train BEST model and perform SAMPLING VALIDATION on prediction population ---
  cat(paste0("[STEP 2] Training '", best_model_name, "' on all '", train_pop_name, "' and validating on '", pred_pop_name, "'...\n"))
  final_train_geno <- genotypeMatrix[train_ids, ]; final_train_pheno <- phenotypeVector[train_ids]; prediction_accuracies <- numeric(pred_samples)
  
  for (iter in 1:pred_samples) {
    if (iter %% 10 == 0) cat(paste0("  - Pred Sample ", iter, "/", pred_samples, "...\n")); set.seed(42 + iter)
    sample_pred_ids <- sample(pred_ids, size = floor(length(pred_ids) * pred_sample_frac))
    final_pred_geno_sample <- genotypeMatrix[sample_pred_ids, ]; final_pred_pheno_sample <- phenotypeVector[sample_pred_ids]
    predictions_sample <- NULL
    
    # --- PREDICTION LOGIC for the identified BEST model - CORE LOGIC UNCHANGED ---
    tryCatch({if(best_model_name %in% c("Ridge","LASSO","Elastic Net")){get_mode<-function(v){v<-v[!is.na(v)];uniqv<-unique(v);uniqv[which.max(tabulate(match(v,uniqv)))]};alpha_val<-case_when(best_model_name=="Ridge"~0,best_model_name=="LASSO"~1,T~get_mode(cv_results_df%>%filter(Model=="Elastic Net")%>%pull(alpha)));final_model_trained<-cv.glmnet(final_train_geno,final_train_pheno,alpha=alpha_val,family="gaussian");predictions_sample<-predict(final_model_trained,newx=final_pred_geno_sample,s="lambda.min")[,1]}else if(best_model_name %in% c("BRR","BayesA","BayesB","BayesC","Bayes G-BLUP")){ids_for_bglr<-c(train_ids,sample_pred_ids);y_bglr<-phenotypeVector[ids_for_bglr];y_bglr[sample_pred_ids]<-NA;model_name_bglr<-if(best_model_name=="Bayes G-BLUP")"RKHS" else best_model_name;ETA_final<-if(model_name_bglr!="RKHS"){list(list(X=genotypeMatrix[ids_for_bglr,],model=model_name_bglr))}else{list(list(K=G_full[ids_for_bglr,ids_for_bglr],model="RKHS"))};fit_bglr_final<-BGLR(y=y_bglr,ETA=ETA_final,nIter=10000,burnIn=2500,verbose=F);pred_indices_in_bglr<-match(sample_pred_ids,names(y_bglr));predictions_sample<-fit_bglr_final$yHat[pred_indices_in_bglr]}else if(best_model_name %in% c("GBLUP","AD-GBLUP","ssGBLUP")){all_final_ids<-unique(c(train_ids,sample_pred_ids,parents));pheno_final<-rep(NA,length(all_final_ids));names(pheno_final)<-all_final_ids;pheno_final[train_ids]<-final_train_pheno;sommer_data_final<-data.frame(ID=names(pheno_final),y=pheno_final);final_fit<-if(best_model_name=="GBLUP"){sommer_data_final$ID<-factor(sommer_data_final$ID,levels=rownames(G_full));mmes(fixed=y~1,random=~vsm(ism(ID),Gu=G_full),rcov=~units,data=sommer_data_final,naMethodY="include",verbose=F)}else if(best_model_name=="AD-GBLUP"){sommer_data_final$ID_A<-factor(sommer_data_final$ID,levels=rownames(G_full));sommer_data_final$ID_D<-factor(sommer_data_final$ID,levels=rownames(D_full));mmes(fixed=y~1,random=~vsm(ism(ID_A),Gu=G_full)+vsm(ism(ID_D),Gu=D_full),rcov=~units,data=sommer_data_final,naMethodY="include",verbose=F)}else{sommer_data_final$ID<-factor(sommer_data_final$ID,levels=rownames(Hinv_full));mmes(fixed=y~1,random=~vsm(ism(ID),Gu=Hinv_full),rcov=~units,data=sommer_data_final,naMethodY="include",verbose=F,henderson=T)};pred_D_arg<-if(best_model_name=="AD-GBLUP")"ID_A" else "ID";final_pred_table<-predict(final_fit,D=pred_D_arg);predictions_sample<-final_pred_table$pvals[sample_pred_ids,"predicted.value"]}else if(best_model_name %in% c("MLP","CNN")){get_mode<-function(v){v<-v[!is.na(v)];if(length(v)==0)return(NA);uniqv<-unique(v);uniqv[which.max(tabulate(match(v,uniqv)))]};train_mean<-colMeans(final_train_geno);train_sd<-apply(final_train_geno,2,sd);train_sd[train_sd==0]<-1;final_train_geno_scaled<-scale(final_train_geno,center=train_mean,scale=train_sd);TRAINING_EPOCHS<-100;final_model_trained<-if(best_model_name=="MLP"){params<-cv_results_df%>%filter(Model=="MLP")%>%summarise(neurons=get_mode(mlp_neurons),dropout=get_mode(mlp_dropout),learning_rate=0.001);model<-keras_model_sequential()%>%layer_dense(units=params$neurons,input_shape=numMarkers,kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dropout(rate=params$dropout)%>%layer_dense(units=round(params$neurons/2),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dense(units=1);model%>%compile(loss="mse",optimizer=optimizer_adam(learning_rate=params$learning_rate));final_callbacks<-list(callback_early_stopping(monitor="loss",patience=10));model%>%fit(final_train_geno_scaled,final_train_pheno,epochs=TRAINING_EPOCHS,batch_size=32,verbose=0,callbacks=final_callbacks);model}else{params<-cv_results_df%>%filter(Model=="CNN")%>%summarise(filters=get_mode(cnn_filters),kernel_size=get_mode(cnn_kernel_size),learning_rate=0.001);train_cnn<-array(final_train_geno_scaled,dim=c(nrow(final_train_geno_scaled),numMarkers,1));model<-keras_model_sequential()%>%layer_conv_1d(filters=params$filters,kernel_size=params$kernel_size,input_shape=c(numMarkers,1),kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_max_pooling_1d(pool_size=4)%>%layer_flatten()%>%layer_dense(units=64,kernel_regularizer=regularizer_l2(l=0.001))%>%layer_batch_normalization()%>%layer_activation_relu()%>%layer_dense(units=1);model%>%compile(loss="mse",optimizer=optimizer_rmsprop(learning_rate=params$learning_rate));final_callbacks<-list(callback_early_stopping(monitor="loss",patience=10));model%>%fit(train_cnn,final_train_pheno,epochs=TRAINING_EPOCHS,batch_size=32,verbose=0,callbacks=final_callbacks);model};pred_geno_scaled<-scale(final_pred_geno_sample,center=train_mean,scale=train_sd);pred_input<-if(best_model_name=="MLP")pred_geno_scaled else array(pred_geno_scaled,dim=c(nrow(pred_geno_scaled),numMarkers,1));predictions_sample<-final_model_trained%>%predict(pred_input,verbose=0);predictions_sample<-predictions_sample[,1]}},error=function(e){cat(paste0(" !-> Pred failed for ",best_model_name," iter ",iter,": ",e$message,"\n"));predictions_sample<<-NULL})
    
    if(!is.null(predictions_sample)){prediction_accuracies[iter]<-cor(predictions_sample,final_pred_pheno_sample,use="complete.obs")}else{prediction_accuracies[iter]<-NA}
  }
  
  avg_final_pred_accuracy <- mean(prediction_accuracies, na.rm = TRUE)
  sd_final_pred_accuracy <- sd(prediction_accuracies, na.rm = TRUE)
  cat(paste0(">>> Avg. Cross-Pop Accuracy: ", round(avg_final_pred_accuracy, 4), " ± ", round(sd_final_pred_accuracy, 4), "\n"))
  
  return(tibble::tibble(
    Training_Population = train_pop_name, Prediction_Population = pred_pop_name,
    Best_Model_in_CV = best_model_name, CV_Accuracy_of_Best_Model = cv_summary_stats$Mean_Cor[1],
    Avg_Cross_Pop_Accuracy = avg_final_pred_accuracy, SD_Cross_Pop_Accuracy = sd_final_pred_accuracy,
    CV_Summary = list(cv_summary_stats)
  ))
}

# =====================================================================================
# Part 8: Execute Scenarios for All Traits and Report Final Results
# =====================================================================================
all_results_list <- list()

# UPGRADE: Start of the main loop to iterate through each trait
for (TRAIT_OF_INTEREST in ALL_TRAITS_IN_FILE) {
  
  cat(paste0("\n\n######################################################################\n"))
  cat(paste0("###   STARTING CROSS-POP ANALYSIS FOR TRAIT: ", TRAIT_OF_INTEREST, "   ###\n"))
  cat(paste0("######################################################################\n"))
  
  # --- Select phenotype data for the current trait ---
  phenotypeVector_current <- pheno_df_aligned[[TRAIT_OF_INTEREST]]
  names(phenotypeVector_current) <- rownames(pheno_df_aligned)
  
  # --- Run Scenario A: Train on NH/HN, Predict on HP ---
  result_A <- perform_cross_population_prediction(
    train_ids = pop_NH_HN_ids, pred_ids = pop_HP_ids,
    train_pop_name = "NH_HN", pred_pop_name = "HP",
    phenotypeVector = phenotypeVector_current # Pass current trait data
  )
  
  # --- Run Scenario B: Train on HP, Predict on NH/HN ---
  result_B <- perform_cross_population_prediction(
    train_ids = pop_HP_ids, pred_ids = pop_NH_HN_ids,
    train_pop_name = "HP", pred_pop_name = "NH_HN",
    phenotypeVector = phenotypeVector_current # Pass current trait data
  )
  
  # UPGRADE: Store results for the current trait
  all_results_list[[TRAIT_OF_INTEREST]] <- bind_rows(result_A, result_B)
  
  # UPGRADE: Save detailed CV summary for the current trait
  cv_summary_A <- result_A$CV_Summary[[1]]
  cv_summary_B <- result_B$CV_Summary[[1]]
  write.csv(cv_summary_A, paste0("results/GS_CrossPop_CV_Summary_Train-NH_HN_", TRAIT_OF_INTEREST, ".csv"), row.names = FALSE)
  write.csv(cv_summary_B, paste0("results/GS_CrossPop_CV_Summary_Train-HP_", TRAIT_OF_INTEREST, ".csv"), row.names = FALSE)
}

# =====================================================================================
# Part 9: Final Summary and Cleanup
# =====================================================================================
# --- IMPORTANT: Stop the parallel cluster ---
cat("\n[CLEANUP] All traits analyzed. Stopping the parallel cluster...\n")
if(exists("cl")) parallel::stopCluster(cl)
plan(sequential)

# --- Combine results from all traits into a final summary table ---
final_summary_all_traits <- bind_rows(all_results_list, .id = "Trait") %>%
    dplyr::select(Trait, everything(), -CV_Summary)

cat("\n\n====================================================================\n")
cat("            FINAL CROSS-POPULATION PREDICTION SUMMARY (ALL TRAITS)\n")
cat("====================================================================\n\n")
print(final_summary_all_traits)

# --- Save the final combined summary to a CSV file ---
output_filename <- "results/GS_Cross_Population_Summary_All_Traits.csv"
write.csv(final_summary_all_traits, output_filename, row.names = FALSE)
cat(paste0("\n\n[--- FINISHED ---] Analysis complete. Final summary saved to '", output_filename, "'\n"))
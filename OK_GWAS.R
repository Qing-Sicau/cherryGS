# before run the R, run these three commands in bash
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export DT_NUM_THREADS=1

# ==============================================================================
# GWASpoly Pipeline: Verified
# 添加了: Parallelism + Detailed Time Tracking + Error Logging
# ==============================================================================

# 1. Environment & Logger Setup
Sys.setenv(OMP_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)

library(data.table)
setDTthreads(1)      # CRITICAL: Force data.table to single thread
library(GWASpoly)
library(rsvd)
library(ggplot2)
library(parallel)

# --- Configuration ---
LOG_FILE        <- "pipeline_log.txt"
GENO_FILE       <- "dosage_matrix.csv"
PHENO_RAW       <- "phenotype_blup_with_family.csv"
PHENO_PCA       <- "phenotype_with_PCA.csv"

# Hardware & Params
PLOIDY          <- 4
N_PERM          <- 1000       # Total permutations
BP_WINDOW       <- 1e6
N_CORES_K       <- 12         # Cores for Kinship
N_CORES_GWAS    <- 4          # Cores for Single Scan
N_CORES_PERM    <- 12         # Cores for Parallel Permutation

# --- Logger Function ---
# Writes to both Console and File with Timestamp
log_info <- function(msg, type = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  formatted_msg <- paste0("[", timestamp, "] [", type, "] ", msg)
  cat(formatted_msg, "\n")
  write(formatted_msg, file = LOG_FILE, append = TRUE)
}

# Initialize Log
if(file.exists(LOG_FILE)) file.remove(LOG_FILE)
log_info("=== Pipeline Started ===")
log_info(paste("Configuration: Cores_Perm =", N_CORES_PERM, ", Permutations =", N_PERM))

# ------------------------------------------------------------------------------
# 2. Data Preparation (PCA)
# ------------------------------------------------------------------------------
step_start <- Sys.time()
log_info("Step 1: Checking PCA status...")

if(!file.exists(PHENO_PCA)) {
  log_info("PCA file not found. Starting rsvd calculation...")
  
  # Read Genotype
  t0 <- Sys.time()
  geno <- fread(GENO_FILE, header = TRUE, check.names = FALSE, data.table = FALSE)
  log_info(paste("Genotype loaded in", round(difftime(Sys.time(), t0, units="secs"), 1), "s"))
  
  marker <- geno[[1]]
  geno_num <- as.matrix(geno[,-c(1:5)])
  rownames(geno_num) <- marker
  
  # Transpose
  X <- t(geno_num)
  rm(geno, geno_num); gc()
  
  # Impute & Filter
  log_info("Imputing missing values and filtering monomorphic markers...")
  col_vars <- apply(X, 2, var, na.rm = TRUE)
  X <- X[, col_vars > 1e-8]
  
  # Simple Imputation Loop
  for(j in seq_len(ncol(X))) {
    if(anyNA(X[,j])) X[is.na(X[,j]), j] <- mean(X[,j], na.rm = TRUE)
  }
  
  # RSVD
  log_info("Running Randomized SVD (k=10)...")
  t_pca <- Sys.time()
  pca_rsvd <- rsvd(X, k = 10, p = 10, q = 3)
  log_info(paste("SVD completed in", round(difftime(Sys.time(), t_pca, units="secs"), 1), "s"))
  
  # Calculate Scores
  pca_scores <- pca_rsvd$u[, 1:3] %*% diag(pca_rsvd$d[1:3])
  pca_df <- as.data.frame(pca_scores)
  colnames(pca_df) <- paste0("PC", 1:3)
  pca_df$ID <- rownames(X)
  
  # Merge Pheno
  pheno_raw <- fread(PHENO_RAW, check.names = FALSE, data.table = FALSE)
  if(!"ID" %in% names(pheno_raw)) names(pheno_raw)[1] <- "ID"
  
  pheno_pca <- merge(pheno_raw, pca_df, by = "ID", all.x = TRUE)
  fwrite(pheno_pca, PHENO_PCA, row.names = FALSE)
  
  log_info(paste("PCA integrated. File saved to:", PHENO_PCA))
  rm(X, pca_rsvd, pca_df, pca_scores, col_vars); gc(full=TRUE)
  
} else {
  log_info("PCA file exists. Skipping calculation.")
  pheno_pca <- fread(PHENO_PCA, check.names = FALSE, data.table = FALSE)
}
log_info(paste("Step 1 Finished. Total time:", round(difftime(Sys.time(), step_start, units="mins"), 2), "mins"))

# ------------------------------------------------------------------------------
# 3. GWASpoly 
# ------------------------------------------------------------------------------
log_info("Step 2: Loading data into GWASpoly...")

trait_cols <- setdiff(colnames(pheno_pca), c("ID", "PC1", "PC2", "PC3", "Family", "Block", "Rep"))
log_info(paste("Detected", length(trait_cols), "traits to analyze."))

t_read <- Sys.time()
data_raw <- read.GWASpoly(ploidy = PLOIDY,
                          pheno.file = PHENO_PCA,
                          geno.file  = GENO_FILE,
                          format     = "numeric",
                          n.traits   = length(trait_cols),
                          delim      = ",")
log_info(paste("GWASpoly object created in", round(difftime(Sys.time(), t_read, units="secs"), 1), "s"))

# Inject PCs
pheno_ordered <- pheno_pca[match(data_raw@pheno$ID, pheno_pca$ID), ]
data_raw@fixed <- data.frame(PC1 = pheno_ordered$PC1,
                             PC2 = pheno_ordered$PC2,
                             PC3 = pheno_ordered$PC3)
rownames(data_raw@fixed) <- rownames(data_raw@pheno)

# Kinship
log_info("Calculating Kinship (LOCO)...")
t_kin <- Sys.time()
data_K <- set.K(data_raw, LOCO = TRUE, n.core = N_CORES_K)
log_info(paste("Kinship calculated in", round(difftime(Sys.time(), t_kin, units="mins"), 2), "mins"))

# Params
params_robust <- set.params(fixed = c("PC1","PC2","PC3"), fixed.type = rep("numeric", 3), MAF = 0.05, P3D = TRUE)
models_to_test <- c("additive", "1-dom", "2-dom")

# ------------------------------------------------------------------------------
# 4. Worker Function (With Timer & Logger)
# ------------------------------------------------------------------------------
run_perm_worker <- function(i, data_obj, trait_name, models, params, log_file) {
  # 1. Start Timer
  t_worker_start <- Sys.time()
  
  # 2. Shuffle Phenotype
  current_pheno <- data_obj@pheno[, trait_name]
  data_obj@pheno[, trait_name] <- sample(current_pheno)
  
  # 3. Run Scan 
  # n.core MUST be 1 here, 不然报错
  perm_res <- GWASpoly(data = data_obj, models = models, traits = trait_name, 
                       params = params, n.core = 1, quiet = TRUE)
  
  max_score <- max(perm_res@scores[[trait_name]], na.rm = TRUE)
  
  # 4. Stop Timer & Calculate Duration
  t_worker_end <- Sys.time()
  duration <- round(difftime(t_worker_end, t_worker_start, units="secs"), 1)
  
  # 5. Build Log Message
  # Format: [Time] [Perm X] Finished in Y s
  timestamp <- format(t_worker_end, "%H:%M:%S")
  msg <- paste0("[", timestamp, "] [Perm ", i, "] Finished in ", duration, " s")
  
  # 6. Output
  # Display on Screen (Real-time monitoring)
  cat(msg, "\n")
  
  # Write to Log File
  # Use try() to prevent crashing if file is locked by another core (rare but possible)
  try({
    write(msg, file = log_file, append = TRUE)
  }, silent = TRUE)
  
  # Cleanup
  rm(perm_res, current_pheno)
  return(max_score)
}

# ------------------------------------------------------------------------------
# 5. Main Loop with Progress Tracking
# ------------------------------------------------------------------------------
log_info("Step 3: Starting Trait Analysis Loop")
total_traits <- length(trait_cols)
trait_counter <- 0
start_loop_time <- Sys.time()

for (trait in trait_cols) {
  trait_counter <- trait_counter + 1
  trait_start_time <- Sys.time()
  
  msg <- paste0(">>> [", trait_counter, "/", total_traits, "] Processing Trait: ", trait)
  log_info(msg)
  
  out_csv <- paste0("GWAS_Result_", trait, ".csv")
  if(file.exists(out_csv)) {
    log_info("Result file exists. Skipping.")
    next
  }
  
  tryCatch({
    # --- Part A: Main Scan ---
    log_info("   [A] Running Main Scan...")
    t_a <- Sys.time()
    res_scan <- GWASpoly(data_K, models = models_to_test, traits = trait, 
                         params = params_robust, n.core = N_CORES_GWAS, quiet = TRUE)
    log_info(paste("   Main Scan done in", round(difftime(Sys.time(), t_a, units="secs"), 1), "s"))
    
    # --- Part B: Permutations ---
    log_info(paste("   [B] Launching", N_PERM, "permutations on", N_CORES_PERM, "cores..."))
    t_b <- Sys.time()
    
    perm_max_scores_list <- mclapply(1:N_PERM, 
                                     FUN = run_perm_worker, 
                                     data_obj = data_K, 
                                     trait_name = trait,
                                     models = models_to_test,
                                     params = params_robust,
                                     log_file = LOG_FILE,
                                     mc.cores = N_CORES_PERM,
                                     mc.preschedule = TRUE)
    
    # Validation
    if (any(sapply(perm_max_scores_list, inherits, "try-error"))) {
      stop("One or more workers crashed (Check RAM/Threads).")
    }
    
    perm_time <- difftime(Sys.time(), t_b, units="mins")
    log_info(paste("   Permutations finished in", round(perm_time, 2), "mins"))
    
    # Threshold & Save
    perm_max_scores <- unlist(perm_max_scores_list)
    perm_threshold <- quantile(perm_max_scores, 0.95)
    log_info(paste("   Calculated Threshold (p=0.05):", round(perm_threshold, 3)))
    
    # insert threshold slot into the S4 object
    res_final <- set.threshold(res_scan, method = "Bonferroni", level = 0.05)
    res_final@threshold[] <- perm_threshold
    
    qtl <- get.QTL(res_final, bp.window = BP_WINDOW)
    fwrite(qtl, out_csv)
    saveRDS(res_final, file = paste0("RDS_Object_", trait, ".rds"))
    
    # Plots
    log_info("   Generating Plots...")
    try({
      p1 <- manhattan.plot(res_final, traits = trait)
      ggsave(paste0("Manhattan_", trait, ".pdf"), p1, width = 14, height = 6)
      p2 <- qq.plot(res_final, trait = trait)
      ggsave(paste0("QQ_", trait, ".pdf"), p2, width = 12, height = 6)
    }, silent=TRUE)
    
    # --- Loop Statistics ---
    trait_duration <- difftime(Sys.time(), trait_start_time, units="mins")
    log_info(paste("   COMPLETED:", trait, "in", round(trait_duration, 2), "mins"))
    
    # Time Estimation
    avg_time <- difftime(Sys.time(), start_loop_time, units="mins") / trait_counter
    remaining <- avg_time * (total_traits - trait_counter)
    log_info(paste("   ESTIMATED REMAINING TIME:", round(remaining, 1), "mins"))
    
    rm(res_scan, res_final, qtl, perm_max_scores_list)
    gc(full=TRUE)
    
  }, error = function(e) {
    log_info(paste("   !!! CRITICAL ERROR:", trait, "-", e$message), type = "ERROR")
  })
}

log_info("=== Pipeline Finished Successfully ===")
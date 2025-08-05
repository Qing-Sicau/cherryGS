# ================================================================= #
#  polyRAD 最佳实践分析流程 v2.0 
# ================================================================= #

# --- 1. 安装和加载库 ---
# 确保在脚本开始时加载所有需要的包
if (!require("polyRAD")) install.packages("polyRAD")
if (!require("data.table")) install.packages("data.table")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("pheatmap")) install.packages("pheatmap")

library(polyRAD)
library(data.table)
library(ggplot2)
library(pheatmap)


# ================================================================= #
# --- 2. 设置核心参数 (用户仅需修改此部分) ---
# ================================================================= #
PARENT_1_NAME <- "HF1"
PARENT_2_NAME <- "NZH2"
PLOIDY <- 4
INPUT_FILE <- "hn_nh_input_44384_for_polyrad.txt"
# ================================================================= #


# --- 3. 文件名自动管理 ---
# 根据输入文件名自动生成所有输出文件的名称
cat("STEP 1: Setting up filenames based on input...\n")
BASE_NAME <- tools::file_path_sans_ext(INPUT_FILE)

# QC图表文件名
PDF_HINDHE      <- paste0(BASE_NAME, "_1_qc_hind_he.pdf")
PDF_PARENTAL    <- paste0(BASE_NAME, "_2_qc_parental_plot.pdf")
PDF_OVERDISP    <- paste0(BASE_NAME, "_3_qc_overdispersion.pdf")
PDF_ALLELE_FREQ <- paste0(BASE_NAME, "_4_qc_allele_freq_map.pdf")
PDF_HEATMAP     <- paste0(BASE_NAME, "_5_qc_dosage_heatmap.pdf")

# 结果输出文件名
OUTPUT_QTL_CSV      <- paste0(BASE_NAME, "_QTL_dosages.csv")
OUTPUT_GS_CSV       <- paste0(BASE_NAME, "_GS_dosages.csv")
OUTPUT_POLYMAPR_RDS <- paste0(BASE_NAME, "_polymapR.rds")
OUTPUT_RRBLUP_RDS   <- paste0(BASE_NAME, "_rrBLUP.rds")
OUTPUT_MAPPOLY_TXT  <- paste0(BASE_NAME, "_MAPpoly.txt")
cat("All output filenames generated automatically.\n\n")


# --- 4. 数据读取与RADdata对象创建 ---
cat("STEP 2: Reading and preparing data...\n")
dat <- fread(INPUT_FILE)
sample_names <- unique(dat$Sample)
locus_names <- unique(dat$Locus)
n_samples <- length(sample_names)
n_loci <- length(locus_names)
locTable <- data.frame(row.names = locus_names)

ref_wide <- dcast(dat, Sample ~ Locus, value.var = "Ref_Depth", fill = 0)
alt_wide <- dcast(dat, Sample ~ Locus, value.var = "Alt_Depth", fill = 0)
ref_mat <- as.matrix(ref_wide[match(sample_names, ref_wide$Sample), -1])
alt_mat <- as.matrix(alt_wide[match(sample_names, alt_wide$Sample), -1])
colnames(ref_mat) <- names(ref_wide)[-1]; colnames(alt_mat) <- names(alt_wide)[-1]
ref_mat <- ref_mat[, locus_names]; alt_mat <- alt_mat[, locus_names]

alleleDepth_matrix <- cbind(ref_mat, alt_mat)
rownames(alleleDepth_matrix) <- sample_names
alleles2loc_vec <- rep(1:n_loci, 2)
alleleNucleotides_vec <- c(paste0(locus_names, "_Ref"), paste0(locus_names, "_Alt"))
taxaPloidy_vec <- rep(2, n_samples) # 标准用法：设为2
possiblePloidies_list <- list(as.integer(PLOIDY))

rad_data <- RADdata(alleleDepth = alleleDepth_matrix, 
                    alleles2loc = alleles2loc_vec, locTable = locTable,
                    possiblePloidies = possiblePloidies_list, 
                    alleleNucleotides = alleleNucleotides_vec,
                    taxaPloidy = taxaPloidy_vec, contamRate = 0.001)
rad_data <- SetDonorParent(rad_data, PARENT_1_NAME)
rad_data <- SetRecurrentParent(rad_data, PARENT_2_NAME)
cat("RADdata object created successfully.\n\n")


# --- 5. 质量控制 (HindHe) ---
cat("STEP 3: Quality control using HindHeMapping...\n")
hh_matrix <- HindHeMapping(rad_data, n.gen.backcrossing = 0, n.gen.selfing = 0)
hh_means <- colMeans(hh_matrix, na.rm = TRUE)

pdf(file = PDF_HINDHE, width = 8, height = 6)
hist(hh_means, breaks = 50, xlab = "Mean Hind/He per Locus",
     main = "Hind/He Distribution\n(For het parents, focus on removing the long right tail)")
abline(v = (PLOIDY - 1) / PLOIDY, col = "red", lwd = 2, lty = 2)
legend("topright", legend = "Theoretical F1 peak (homozygous parents)", col = "red", lty = 2, bty = "n")
dev.off()
cat(paste("Hind/He distribution plot saved to:", PDF_HINDHE, "\n"))

HINDHE_THRESHOLD <- 1.0
loci_to_keep <- names(hh_means[hh_means < HINDHE_THRESHOLD & !is.na(hh_means)])
original_loci_count <- nLoci(rad_data)
rad_data_filtered <- SubsetByLocus(rad_data, loci = loci_to_keep)
cat(paste("Filtered out", original_loci_count - nLoci(rad_data_filtered), "loci based on Hind/He >", HINDHE_THRESHOLD, "\n"))
cat(paste("Remaining loci:", nLoci(rad_data_filtered), "\n\n"))


# --- 6. 估算并检查亲本基因型 (手动QC图) ---
cat("STEP 4: Estimating and inspecting parental genotypes...\n")
rad_data_filtered <- EstimateParentalGenotypes(rad_data_filtered)
depth_ratios <- rad_data_filtered$depthRatio
parent1_ratios <- depth_ratios[PARENT_1_NAME, (nLoci(rad_data_filtered) + 1):ncol(depth_ratios)]
parent2_ratios <- depth_ratios[PARENT_2_NAME, (nLoci(rad_data_filtered) + 1):ncol(depth_ratios)]

parental_plot_df <- data.frame(Locus = names(parent1_ratios), P1_Alt_Ratio = parent1_ratios, P2_Alt_Ratio = parent2_ratios)

qc_plot <- ggplot(parental_plot_df, aes(x = P1_Alt_Ratio, y = P2_Alt_Ratio)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(-0.1, 1.1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(-0.1, 1.1)) +
  labs(title = "Parental Genotype Segregation Plot (Manual QC)",
       subtitle = "Each point is a locus, plotted by allele ratio in parents.",
       x = paste(PARENT_1_NAME, "Alt Allele Ratio"), y = paste(PARENT_2_NAME, "Alt Allele Ratio")) +
  theme_bw() +
  geom_vline(xintercept = c(0, 0.25, 0.5, 0.75, 1), linetype = "dashed", color = "grey") +
  geom_hline(yintercept = c(0, 0.25, 0.5, 0.75, 1), linetype = "dashed", color = "grey")

pdf(file = PDF_PARENTAL, width = 8, height = 8); print(qc_plot); dev.off()
cat(paste("CRITICAL QC: Parental genotype plot saved to:", PDF_PARENTAL, "\n\n"))


# --- 7. 参数寻优 (Overdispersion) ---
cat("STEP 5: Testing for optimal overdispersion parameter...\n")
overdispersion_test <- TestOverdispersion(rad_data_filtered, to_test = seq(2, 20, by = 2))
overdisp_toplot <- overdispersion_test[names(overdispersion_test) != "optimal"]
pdf(file = PDF_OVERDISP, width = 8, height = 6)
boxplot(overdisp_toplot, xlab = "Overdispersion Parameter", ylab = "P-value Distribution",
        main = "P-value Distribution by Overdispersion Parameter\n(A uniform distribution near 0.5 is ideal)")
dev.off()
cat(paste("Overdispersion test boxplot saved to:", PDF_OVERDISP, "\n"))

optimal_od <- overdispersion_test$optimal
if (is.null(optimal_od) || is.na(optimal_od)) {
  if(is.na(optimal_od)){ cat("Optimal overdispersion was at the limit of the tested range. Consider testing a different range.\n") }
  cat("Could not determine an optimal overdispersion value. Using fallback value of 9 for this run.\n")
  optimal_od <- 9
}
cat(paste("Optimal overdispersion parameter found/set to:", optimal_od, "\n\n"))


# --- 8. 运行最终作图流程 (含连锁分析) ---
cat("STEP 6: Running final mapping population pipeline...\n")
tryCatch({
  loc_info <- data.frame(Locus = GetLoci(rad_data_filtered))
  valid_loci <- grepl(":", loc_info$Locus)
  if(sum(valid_loci) < length(valid_loci)){
    cat(paste("Warning:", sum(!valid_loci), "locus names do not contain ':' and will be excluded from linkage analysis.\n"))
  }
  loc_info_split <- strsplit(loc_info$Locus[valid_loci], ':', fixed = TRUE)
  # 创建包含所有位点信息的locTable，但只为有效位点填充Chr和Pos
  locTable_update <- data.frame(row.names = loc_info$Locus)
  locTable_update$Chr <- NA
  locTable_update$Pos <- NA
  locTable_update[valid_loci, "Chr"] <- sapply(loc_info_split, `[`, 1)
  locTable_update[valid_loci, "Pos"] <- as.numeric(sapply(loc_info_split, `[`, 2))
  rad_data_filtered$locTable <- locTable_update
}, error = function(e) {
  stop("Could not parse locus names into Chr:Pos format. Linkage analysis requires this. Please check your locus names.")
})

rad_data_final <- PipelineMapping2Parents(rad_data_filtered,
                                          overdispersion = optimal_od,
                                          useLinkage = TRUE,
                                          linkageDist = 1e7,
                                          minLinkageCorr = 0.15,
                                          freqAllowedDeviation = 0.05)
cat("Main analysis pipeline complete.\n\n")


# --- 9. 最终结果可视化 (手动QC图) ---
cat("STEP 7: Visualizing final allele frequency map...\n")
progeny_names <- GetTaxa(rad_data_final)[!GetTaxa(rad_data_final) %in% c(PARENT_1_NAME, PARENT_2_NAME)]
progeny_dosages <- GetWeightedMeanGenotypes(rad_data_final, minval = 0, maxval = 1)[progeny_names, ]
observed_freq <- colMeans(progeny_dosages, na.rm = TRUE)
map_plot_df <- data.frame(Locus = names(observed_freq),
                          ExpectedFreq = rad_data_final$alleleFreq,
                          ObservedFreq = observed_freq)
map_plot_df <- map_plot_df[!is.na(map_plot_df$ExpectedFreq), ]
manual_map_plot <- ggplot(map_plot_df, aes(x = as.factor(ExpectedFreq), y = ObservedFreq)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
  labs(title = "Allele Frequency Map (Manual Plot)",
       subtitle = "Distribution of Observed vs. Expected Allele Frequencies in Progeny",
       x = "Expected Allele Frequency (from parental genotypes)",
       y = "Observed Mean Allele Dosage in Progeny") +
  theme_bw()
pdf(file = PDF_ALLELE_FREQ, width = 10, height = 8); print(manual_map_plot); dev.off()
cat(paste("Final allele frequency map plot saved to:", PDF_ALLELE_FREQ, "\n\n"))


# --- 10. 导出最终结果 (QTL, GS, etc.) ---
cat("STEP 8: Exporting final results for downstream analysis...\n")

# --- 10.1 为QTL/连锁图谱分析导出 ---
cat("Exporting discrete integer dosages (for Linkage Mapping / QTL)...\n")
# 优化：只计算一次，复用结果
probable_genotypes_list <- GetProbableGenotypes(rad_data_final, multiallelic = "ignore")
qtl_dosage_matrix <- probable_genotypes_list$genotypes
qtl_dosage_df <- as.data.frame(qtl_dosage_matrix)
qtl_dosage_df <- cbind(Sample = rownames(qtl_dosage_df), qtl_dosage_df)
fwrite(qtl_dosage_df, file = OUTPUT_QTL_CSV, row.names = FALSE)
cat(paste(" -> QTL dosages saved to:", OUTPUT_QTL_CSV, "\n"))

# --- 10.2 为GS分析导出 ---
cat("Exporting continuous dosages (for Genomic Selection)...\n")
gs_dosage_matrix <- GetWeightedMeanGenotypes(rad_data_final)
gs_dosage_df <- as.data.frame(gs_dosage_matrix)
gs_dosage_df <- cbind(Sample = rownames(gs_dosage_df), gs_dosage_df)
fwrite(gs_dosage_df, file = OUTPUT_GS_CSV, row.names = FALSE)
cat(paste(" -> GS dosages saved to:", OUTPUT_GS_CSV, "\n"))

# --- 10.3 导出至专用多倍体软件格式 (推荐) ---
# for polymapR (手动优化版)
cat("Exporting to polymapR format...\n")
dosage_matrix_t <- t(qtl_dosage_matrix)
all_samples_ordered <- c(PARENT_1_NAME, PARENT_2_NAME, progeny_names)
all_samples_ordered <- all_samples_ordered[all_samples_ordered %in% colnames(dosage_matrix_t)]
polymapR_dosages_manual <- dosage_matrix_t[, all_samples_ordered]
saveRDS(polymapR_dosages_manual, file = OUTPUT_POLYMAPR_RDS)
cat(paste(" -> polymapR object saved to:", OUTPUT_POLYMAPR_RDS, "\n"))

# for rrBLUP
cat("Exporting to rrBLUP format...\n")
grm_input_matrix <- Export_rrBLUP_Amat(rad_data_final)
saveRDS(grm_input_matrix, file = OUTPUT_RRBLUP_RDS)
cat(paste(" -> rrBLUP object saved to:", OUTPUT_RRBLUP_RDS, "\n"))

# for MAPpoly (带函数修正)
cat("Exporting to MAPpoly format...\n")
## Export_MAPpoly修正
Export_MAPpoly_mod <- function (object, file, pheno = NULL, ploidyIndex = 1, progeny = GetTaxa(object)[!GetTaxa(object) %in% 
                                c(GetDonorParent(object), GetRecurrentParent(object), GetBlankTaxa(object))], 
                                digits = 3) 
{
  if (is.null(object$likelyGeno_donor) || is.null(object$posteriorProb)) {
    stop("PipelineMapping2Parents needs to be run before using Export_MAPpoly.")
  }
  if (!is.null(pheno) && is.null(colnames(pheno))) {
    stop("pheno should be a matrix or data frame with column names")
  }
  if (!is.null(pheno) && nrow(pheno) != length(progeny)) {
    stop("Need one row of pheno for every progeny.")
  }
  if (!is.null(pheno) && !is.null(row.names(pheno)) && !identical(progeny, 
                                                                  row.names(pheno))) {
    warning("Please check that progeny vector and rows of pheno are in same order.")
  }
  if (!is.null(pheno) && any(grepl(" ", colnames(pheno)))) {
    stop("Phenotype names should not have spaces.")
  }
  if (!all(progeny %in% GetTaxa(object))) {
    stop("Not all progeny names found in object.")
  }
  if (any(grepl(" ", progeny))) {
    stop("Taxa names should not have spaces.")
  }
  if (ploidyIndex > length(object$possiblePloidies)) {
    stop("ploidyIndex should be the index of the desired ploidy within object$possiblePloidies (not the ploidy itself).")
  }
  
  ploidy <- object$possiblePloidies[[ploidyIndex]]
  if (length(ploidy) != 1) {
    stop("Export is for autopolyploids only.")
  }
  
  # 关键修正：我们不仅需要计算最终倍性pld.p，还需要保留正确的索引值pld.mult.p
  pld.mult.p <- unique(object$taxaPloidy[progeny])
  if (length(pld.mult.p) > 1) {
    stop("All progeny must have the same ploidy multiplier (taxaPloidy).")
  }
  
  pld.r <- object$taxaPloidy[GetRecurrentParent(object)] * ploidy/2
  pld.d <- object$taxaPloidy[GetDonorParent(object)] * ploidy/2
  pld.p <- pld.mult.p * ploidy/2 # pld.p (最终分析倍性) 保持原来的计算方式

  donorGen <- object$likelyGeno_donor[as.character(pld.d), ]
  recurGen <- object$likelyGeno_recurrent[as.character(pld.r), ]
  keepal <- which(!is.na(donorGen) & !is.na(recurGen) & !(donorGen == 0 & recurGen == ploidy) & !(donorGen == ploidy & recurGen == 0))
  # keepal <- keepal[!keepal %in% OneAllelePerMarker(object, commonAllele = TRUE)]
  
  if (any(grepl(" ", GetAlleleNames(object)[keepal]))) {
    stop("Allele and locus names should not have spaces.")
  }

  if (is.null(object$locTable$Chr) || all(is.na(object$locTable$Chr))) {
    chrnum <- NA_integer_
  } else {
    # Assuming .chromosome_to_integer is an internal polyRAD function
    chrnum <- polyRAD:::.chromosome_to_integer(object$locTable$Chr[object$alleles2loc[keepal]])
  }
  if (is.null(object$locTable$Pos) || all(is.na(object$locTable$Pos))) {
    position <- NA_integer_
  } else {
    position <- object$locTable$Pos[object$alleles2loc[keepal]]
  }
  

  cat(c(paste("ploidy", pld.p), paste("nind", length(progeny)), 
        paste("nmrk", length(keepal)), paste("mrknames", 
                                             paste(GetAlleleNames(object)[keepal], collapse = " ")), 
                                             paste("indnames", paste(progeny,collapse = " ")), 
                                             paste("dosageP", paste(donorGen[keepal], 
                                                                    collapse = " ")), paste("dosageQ", paste(recurGen[keepal], 
                                             collapse = " ")), paste("seq", paste(chrnum, collapse = " ")), 
        paste("seqpos", paste(position, collapse = " ")), paste("nphen", ifelse(is.null(pheno), 0, ncol(pheno))), 
        "pheno---------------------------------------"), 
      file = file, sep = "\n")
  if (!is.null(pheno)) {
    for (i in 1:ncol(pheno)) {
      cat(paste(colnames(pheno)[i], paste(pheno[, i], collapse = " ")), file = file, sep = "\n", append = TRUE)
    }
  }
  cat("geno----------------------------------------", file = file, 
      sep = "\n", append = TRUE)

  # 在这里，我们使用正确的索引 pld.mult.p 来访问 posteriorProb
  genotab <- data.frame(rep(GetAlleleNames(object)[keepal], 
                            each = length(progeny)), rep(progeny, times = length(keepal)), 
                        matrix(round(object$posteriorProb[[ploidyIndex, as.character(pld.mult.p)]][, # <-- 使用修正的索引
                        progeny, keepal], digits), byrow = TRUE, nrow = length(progeny) * length(keepal), ncol = pld.p + 1))
  write.table(genotab, file = file, append = TRUE, quote = FALSE, 
              col.names = FALSE, row.names = FALSE)
}
### 修正结束


tryCatch({
  Export_MAPpoly_mod(object = rad_data_final, file = OUTPUT_MAPPOLY_TXT, pheno = NULL, ploidyIndex = 1)
  cat(paste(" -> MAPpoly text file saved to:", OUTPUT_MAPPOLY_TXT, "\n\n"))
}, error = function(e) {
  cat("\n!!!!!!!! ERROR in Export_MAPpoly !!!!!!!!\n")
  cat("错误信息:", e$message, "\n")
  })


# --- 11. [可选] 绘制基因型热图 ---
cat("STEP 9 (Optional): Generating genotype heatmap...\n")
if (ncol(qtl_dosage_matrix) > 0) {
  n_loci_to_plot <- min(ncol(qtl_dosage_matrix), 5000)
  loci_to_plot_idx <- sample(1:ncol(qtl_dosage_matrix), n_loci_to_plot)
  dosage_matrix_subset <- qtl_dosage_matrix[, loci_to_plot_idx]
  
  variance_check <- apply(dosage_matrix_subset, 2, var, na.rm = TRUE)
  if(sum(variance_check > 0, na.rm = TRUE) > 0) {
    dosage_matrix_subset <- dosage_matrix_subset[, variance_check > 0, drop = FALSE]
    
    cat("Imputing NA values for heatmap visualization...\n")
    imputed_matrix_for_plot <- dosage_matrix_subset
    for (i in 1:ncol(imputed_matrix_for_plot)) {
      col_mean <- mean(imputed_matrix_for_plot[, i], na.rm = TRUE)
      if(is.nan(col_mean)) { col_mean <- 0 }
      imputed_matrix_for_plot[is.na(imputed_matrix_for_plot[, i]), i] <- col_mean
    }
    
    cat(paste("Generating heatmap for", ncol(imputed_matrix_for_plot), "polymorphic loci...\n"))
    pdf(file = PDF_HEATMAP, width = 10, height = 8)
    pheatmap(imputed_matrix_for_plot,
             cluster_rows = TRUE, cluster_cols = TRUE,
             show_rownames = FALSE, show_colnames = FALSE,
             main = paste("Genotype Dosage Heatmap (sample of", ncol(imputed_matrix_for_plot), "loci)"))
    dev.off()
    cat(paste("Optional dosage heatmap saved to:", PDF_HEATMAP, "\n"))
  } else {
    cat("Skipping heatmap: No loci with variance remaining after sampling.\n")
  }
} else {
  cat("Skipping heatmap generation as there are no final loci.\n")
}

cat("\n=========================================\n")
cat("          Pipeline Finished          \n")
cat("=========================================\n")
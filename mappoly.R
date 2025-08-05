# ================================================================= #
#  mappoly 最佳实践分析流程 v3.0 (Vignette增强版)
# ================================================================= #
#  作者: Google Gemini (根据用户需求深度定制)
#  输入: polyRAD 使用 Export_MAPpoly 生成的概率文件
#  核心策略: 遵循Vignette流程，但使用物理信息分组以处理大规模数据
# ================================================================= #

# --- 1. 安装和加载库 ---
if (!require("mappoly")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("mmollina/mappoly", dependencies = TRUE)
}
library(mappoly)
library(ggplot2)
library(parallel)
library(knitr) # 用于更好地打印摘要表格

# --- 2. 设置核心参数 ---
# 输入文件，由您的 polyRAD 脚本生成
INPUT_FILE_FROM_POLYRAD <- "hn_nh_input_44384_for_polyrad_MAPpoly.txt" 
# 预估的基因分型错误率 (用于图谱精修)
GENOTYPING_ERROR <- 0.05 
# 并行计算使用的核心数
N.CORES <- 15
# 用于Bonferroni校正的P值阈值
ALPHA <- 0.05


# ==============================================================================
# 阶段一: 数据导入与Vignette标准质控
# ==============================================================================
cat("\n--- STAGE 1: Data Import and Vignette-based Quality Control ---\n")

# 1.1 ---- 读取 polyRAD 生成的基因型概率文件 ----
dat <- read_geno_prob(file.in = INPUT_FILE_FROM_POLYRAD, prob.thres = 0.95)
print(dat, detailed = TRUE)

# 1.2 ---- 严格质量控制 (遵循Vignette) ----
# 1.2.1 过滤缺失率过高的标记
cat("\nFiltering markers with > 10% missing data...\n")
dat_filt1 <- filter_missing(input.data = dat, type = "marker", filter.thres = 0.10, inter = FALSE)

# 1.2.2 过滤缺失率过高的个体
cat("Filtering individuals with > 10% missing data...\n")
dat_filt2 <- filter_missing(input.data = dat_filt1, type = "individual", filter.thres = 0.10, inter = FALSE)

# 1.2.3 过滤分离严重失真的标记
cat("Filtering markers with significant segregation distortion...\n")
pval_bonferroni <- ALPHA / dat_filt2$n.mrk
mrks_chi_filt <- filter_segregation(dat_filt2, chisq.pval.thres = pval_bonferroni, inter = FALSE)

# 1.2.4 标记去冗余 (Marker Binning - Vignette中的 'elim_redundant')
cat("Performing Marker Binning (eliminating redundant markers)...\n")
# elim_redundant 功能上等同于 polymapR 的 screen_for_duplicate_markers
seq_init <- make_seq_mappoly(mrks_chi_filt)
red_mrks <- elim_redundant(seq_init)
seq_qc_final <- make_seq_mappoly(red_mrks)

cat("\nQuality control complete.\n")
cat("Original markers:", dat$n.mrk, "| Markers after final QC and Binning:", seq_qc_final$n.mrk, "\n\n")


# ==============================================================================
# 阶段二: 构建连锁图谱 (融合Vignette流程与大规模数据策略)
# ==============================================================================
cat("--- STAGE 2: Linkage Map Construction (Scalable Vignette Workflow) ---\n")

# 2.1 ---- 逐条染色体进行标记排序和图谱构建 ----
# 我们利用已知的物理位置进行分组，从而绕过全局两两计算的内存瓶颈
chromosomes <- unique(seq_qc_final$chrom)
maps_list <- list()

for (chr in chromosomes) {
  cat("\n========================================================\n")
  cat("Now processing chromosome:", chr, "\n")
  cat("========================================================\n")
  
  # a. 获取当前染色体的标记序列
  seq_chr <- make_seq_mappoly(seq_qc_final, arg = chr)
  cat(" ->", seq_chr$n.mrk, "markers on this chromosome.\n")
  
  # b. 为当前染色体计算两点信息 (计算规模已大大降低)
  cat(" -> Performing pairwise RF estimation for this chromosome...\n")
  tpt_chr <- est_pairwise_rf(input.seq = seq_chr, ncpus = N.CORES)
  
  # c. 将两点信息转换为矩阵，为MDS排序做准备
  mat_chr <- rf_list_to_matrix(input.twopt = tpt_chr)
  
  # d. 使用MDS算法获取稳健的初始标记顺序 (Vignette推荐步骤)
  cat(" -> Performing MDS ordering...\n")
  mds_ord_chr <- mds_mappoly(input.mat = mat_chr)
  seq_mds_chr <- make_seq_mappoly(mds_ord_chr)
  
  # e. 使用HMM序贯算法进行最终的标记排序和距离计算
  cat(" -> Performing sequential HMM mapping...\n")
  map_chr <- est_rf_hmm_sequential(
    input.seq = seq_mds_chr,
    twopt = tpt_chr,
    start.set = 10,
    thres.twopt = 5,
    thres.hmm = 50,
    extend.tail = 40,
    phase.number.limit = 20,
    verbose = TRUE
  )
  
  # f. 将构建好的图谱存入列表
  maps_list[[chr]] <- map_chr
}

cat("\nInitial map construction for all chromosomes complete.\n\n")


# ==============================================================================
# 阶段三: 图谱精修、可视化与输出
# ==============================================================================
cat("--- STAGE 3: Map Refinement, Visualization, and Export ---\n")

# 3.1 ---- 考虑基因分型错误率，重新估计图谱距离 ----
cat("Refining maps with a global genotyping error model of", GENOTYPING_ERROR, "...\n")

refine_map_with_error <- function(map, error_rate) {
  # 确保map不为空
  if (!is.null(map)) {
    est_full_hmm_with_global_error(input.map = map, error = error_rate, tol = 1e-4, verbose = FALSE)
  } else {
    return(NULL)
  }
}

cl <- makeCluster(N.CORES)
clusterExport(cl, "est_full_hmm_with_global_error")
clusterEvalQ(cl, library(mappoly))
maps_refined <- parLapply(cl, maps_list, fun = refine_map_with_error, error_rate = GENOTYPING_ERROR)
stopCluster(cl)

# 3.2 ---- 将被bin的冗余标记加回图谱 ----
cat("Re-inserting binned markers into the final map...\n")
maps_final <- lapply(maps_refined, update_map, red.mrk)

# 3.3 ---- 可视化与结果输出 ----
cat("Visualizing and exporting final maps...\n")
pdf("mappoly_final_genetic_map.pdf", width = 15, height = 10)
plot_map_list(maps_final, col = "ggstyle")
dev.off()

map_summary <- summary_maps(maps_final)
cat("\n--- Final Map Summary ---\n")
print(kable(map_summary))
write.csv(map_summary, "mappoly_final_map_summary.csv", row.names = FALSE)

export_map_list(maps_final, file = "mappoly_final_phased_map.csv")
save(dat, seq_qc_final, maps_final, file = "mappoly_final_results.RData")

cat("\n--- ANALYSIS COMPLETE ---\n")
cat("Final genetic map, summary, and R objects have been saved to the working directory.\n")
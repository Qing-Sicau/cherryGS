#==============================================================================
# STEP 0: INITIAL SETUP & PACKAGE INSTALLATION
#==============================================================================

# --- Install MAPpoly and other required packages ---
# Note: Only run these lines if the packages are not already installed.
if (!require("mappoly")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("mmollina/mappoly", dependencies = TRUE)
}
if (!require("vcfR")) install.packages("vcfR") # Added vcfR
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("parallel")) install.packages("parallel")
if (!require("RColorBrewer")) install.packages("RColorBrewer")


# --- Load Libraries ---
library(mappoly)
library(vcfR) # Added vcfR
library(ggplot2)
library(parallel)

# --- Set Global Parameters ---
N.CORES <- 15
EXPECTED_LGS <- 8
GENOTYPING_ERROR <- 0.05
# IMPORTANT: Place your VCF file in the main project directory.
# The script will navigate into subdirectories to find it.
VCF_FILE <- "ultimate_qc_passed.vcf.GQ15DP30MAF005.gz" # Path relative to the population-specific directories

# Create directories to store outputs
dir.create("mappoly_analysis", showWarnings = FALSE)
setwd("mappoly_analysis")
dir.create("population_HN", showWarnings = FALSE)



#==============================================================================
# PART I: ANALYSIS OF POPULATION 1 (HF1 x NZH2)
#==============================================================================
cat("--- Starting Analysis for Population 1: HN (HF1 x NZH2) ---\n")


#------------------------------------------------------------------------------
# 1.1 Data Loading using vcfR to Subset the Population
#------------------------------------------------------------------------------
# This method reads the entire VCF into R memory to filter it.
# Note: This can be slow and memory-intensive for very large VCF files.

# Step 1: Read the entire VCF file using vcfR.
cat("Step 1/5: Reading full VCF with vcfR...\n")
vcf_full <- read.vcfR(VCF_FILE)
setwd("./population_HN")
# Step 2: Define the parents and offspring for the HN population.
cat("Step 2/5: Identifying samples for the HN population...\n")
parents_hn <- c("HF1", "NZH2")
all_samples_in_vcf <- colnames(vcf_full@gt)[-1] # [-1] removes the "FORMAT" column
offspring_hn <- all_samples_in_vcf[grepl("^(HN|NH)", all_samples_in_vcf)]
samples_to_keep_hn <- c(parents_hn, offspring_hn)

# Step 3: Subset the vcfR object to keep only the desired samples.
cat("Step 3/5: Subsetting VCF object in memory...\n")
cols_to_keep <- which(colnames(vcf_full@gt) %in% c("FORMAT", samples_to_keep_hn))
vcf_subset_hn <- vcf_full
vcf_subset_hn@gt <- vcf_subset_hn@gt[, cols_to_keep]

# Step 4: Write the subset to a temporary, clean VCF file.
cat("Step 4/5: Writing temporary VCF file for HN population...\n")
temp_vcf_for_hn <- "temp_HN_population.vcf.gz"
write.vcf(vcf_subset_hn, file = temp_vcf_for_hn)

# Step 5: Use MAPpoly to read the clean, temporary VCF file.
cat("Step 5/5: Reading subsetted VCF with MAPpoly...\n")
dat_hn <- read_vcf(
  file = temp_vcf_for_hn, # Read the newly created temporary file
  parent.1 = "NZH2",
  parent.2 = "HF1",
  ploidy = 4,
  filter.non.conforming = TRUE
)

# Cleanup the large R object and temporary file to save memory/space
rm(vcf_full, vcf_subset_hn)
# file.remove(temp_vcf_for_hn) # Optional: uncomment to delete the temp file after reading

cat("Data loading complete. Marker numbers should now be much higher.\n")

#------------------------------------------------------------------------------
# 1.2 Exploratory Analysis
#------------------------------------------------------------------------------
# The rest of the pipeline continues as before with the clean data.
print(dat_hn, detailed = TRUE)

pdf("1_plot_marker_info_HN.pdf", width = 12, height = 8)
plot(dat_hn)
dev.off()

#------------------------------------------------------------------------------
# 1.3 Quality Control
#------------------------------------------------------------------------------
dat_hn_filt <- filter_missing(input.data = dat_hn, type = "marker", filter.thres = 0.10, inter = T)
dat_hn_filt <- filter_missing(input.data = dat_hn_filt, type = "individual", filter.thres = 0.025, inter = T)
print(dat_hn_filt)

pval.bonf_hn <- 0.05 / dat_hn_filt$n.mrk
mrks_chi_filt_hn <- filter_segregation(dat_hn_filt, chisq.pval.thres = pval.bonf_hn, inter = T)
seq_init_hn <- make_seq_mappoly(mrks_chi_filt_hn)

pdf("2_plot_initial_sequences_HN.pdf")
plot(seq_init_hn)
dev.off()

#------------------------------------------------------------------------------
# 1.4 Two-Point Analysis
#------------------------------------------------------------------------------
cat("Performing two-point analysis for HN...\n")
all_rf_pairwise_hn <- est_pairwise_rf(input.seq = seq_init_hn, ncpus = N.CORES)
cat("Done with pairwise computation for HN.\n")

mat_rf_hn <- rf_list_to_matrix(input.twopt = all_rf_pairwise_hn, thresh.LOD.ph = 5, thresh.LOD.rf = 5)

#------------------------------------------------------------------------------
# 1.5 Linkage Group Formation
#------------------------------------------------------------------------------
groups_hn <- group_mappoly(input.mat = mat_rf_hn, expected.groups = 8, inter = F)
print(groups_hn)

pdf("3_plot_linkage_groups_HN.pdf", width=10, height=10)
plot(groups_hn)
dev.off()

lg_list_hn <- make_seq_mappoly(groups_hn, "all")

#------------------------------------------------------------------------------
# 1.6 Marker Ordering and Map Estimation (MDS approach)
#------------------------------------------------------------------------------
cat("Performing MDS ordering for HN...\n")
tpt_list_hn <- make_pairs_mappoly(input.twopt = all_rf_pairwise_hn, input.seq = lg_list_hn)
mds_ord_hn <- mds_mappoly(input.list = tpt_list_hn)
cat("Done with MDS ordering for HN.\n")

seq_mds_hn <- make_seq_mappoly(mds_ord_hn)

phasing_and_hmm_rf <- function(X, dat, twopt.list) {
  tpt_i <- twopt.list[[as.character(X$seq.num[1])]]
  map <- est_rf_hmm_sequential(
    input.seq = X, start.set = 10, thres.twopt = 10, thres.hmm = 10,
    extend.tail = 40, info.tail = TRUE, twopt = tpt_i, sub.map.size.diff.limit = 5,
    phase.number.limit = 20, reestimate.single.ph.configuration = TRUE,
    tol = 1e-3, tol.final = 1e-4, verbose = FALSE
  )
  return(map)
}

cat("Starting parallel map estimation for HN...\n")
cl <- makeCluster(N.CORES)
clusterExport(cl, varlist = c("dat_hn_filt", "tpt_list_hn", "est_rf_hmm_sequential"))
clusterEvalQ(cl, library(mappoly))
maps_hn <- parLapply(cl, seq_mds_hn, fun = phasing_and_hmm_rf, dat = dat_hn_filt, twopt.list = tpt_list_hn)
stopCluster(cl)
cat("Finished map estimation for HN.\n")

#------------------------------------------------------------------------------
# 1.7 Map Refinement
#------------------------------------------------------------------------------
refine_map_with_error <- function(X, error_rate) {
  est_full_hmm_with_global_error(input.map = X, error = error_rate, tol = 1e-4, verbose = FALSE)
}

cat("Refining HN maps with global error model...\n")
cl <- makeCluster(N.CORES)
clusterExport(cl, varlist = c("refine_map_with_error", "est_full_hmm_with_global_error"))
clusterEvalQ(cl, library(mappoly))
maps_hn_err <- parLapply(cl, maps_hn, fun = refine_map_with_error, error_rate = GENOTYPING_ERROR)
stopCluster(cl)

maps_hn_final <- lapply(maps_hn_err, update_map)

#------------------------------------------------------------------------------
# 1.8 Map Visualization and Summary
#------------------------------------------------------------------------------
pdf("4_final_map_HN.pdf", width = 15, height = 10)
plot_map_list(maps_hn_final, col = "ggstyle")
dev.off()

map_summary_hn <- summary_maps(maps_hn_final)
print(knitr::kable(map_summary_hn))
write.csv(map_summary_hn, "map_summary_HN.csv")
export_map_list(maps_hn_final, file = "phased_map_HN.csv")
save(dat_hn_filt, maps_hn_final, file = "mappoly_HN_results.RData")

setwd("../") # Return to main analysis directory


#==============================================================================
# PART II: ANALYSIS OF POPULATION 2 (HF1 x PJHH)
#==============================================================================
cat("\n--- Starting Analysis for Population 2: HP (HF1 x PJHH) ---\n")
#
# For this second population, you will repeat the entire process from PART I.
# The key is to change the parent and offspring identifiers in the
# "1.1 Data Loading" section. For example:
#
# setwd("./population_HP")
# parents_hp <- c("HF1", "PJHH")
# offspring_hp <- all_samples_in_vcf[grepl("^HP", all_samples_in_vcf)]
# ... and so on ...
#
# The rest of the analysis flow is identical.
#
#==============================================================================

cat("\n--- ANALYSIS COMPLETE ---\n")

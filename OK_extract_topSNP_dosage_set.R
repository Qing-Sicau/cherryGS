library(data.table)
library(dplyr)
library(stringr)

# =========================
# User configuration
# =========================
DOSAGE_FILE <- "dosage_matrix.csv"
TOP_SNP_DIR <- "GWAS_topSNPs"
OUT_DIR <- "DL_feature_sets"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

top_k_values <- c(500, 1000, 3000, 6000)

# =========================
# Load dosage matrix
# =========================
cat("[INFO] Loading dosage matrix...\n")
dosage <- fread(DOSAGE_FILE, check.names = FALSE, data.table = FALSE)

cat("[INFO] Dosage dimensions:", nrow(dosage), "markers x", ncol(dosage), "columns\n")

# Try to identify marker column
if ("Marker" %in% colnames(dosage)) {
  marker_col <- "Marker"
} else {
  marker_col <- colnames(dosage)[1]
  message("[WARN] No column named 'Marker'. Using first column as marker column: ", marker_col)
}

# Optional: create CHROM:POS marker ID if needed
if (all(c("CHROM", "POS") %in% colnames(dosage))) {
  dosage$Marker_CHROM_POS <- paste0(dosage$CHROM, ":", dosage$POS)
} else if (all(c("Chrom", "Position") %in% colnames(dosage))) {
  dosage$Marker_CHROM_POS <- paste0(dosage$Chrom, ":", dosage$Position)
}

# =========================
# Find top SNP files
# =========================
top_files <- list.files(
  TOP_SNP_DIR,
  pattern = "_GWAS_top(500|1000|3000|6000)\\.csv$",
  full.names = TRUE
)

cat("[INFO] Found", length(top_files), "top SNP csv files.\n")

extract_one <- function(top_file) {
  base <- basename(top_file)
  
  # Parse trait and top-k
  k <- str_match(base, "_GWAS_top(\\d+)\\.csv$")[, 2]
  trait <- str_replace(base, paste0("_GWAS_top", k, "\\.csv$"), "")
  
  cat("[INFO] Processing:", trait, "top", k, "\n")
  
  top_df <- fread(top_file, data.table = FALSE)
  
  if (!"Marker" %in% colnames(top_df)) {
    stop("No Marker column found in: ", top_file)
  }
  
  top_markers <- unique(top_df$Marker)
  
  # First try matching by Marker column
  matched <- dosage[dosage[[marker_col]] %in% top_markers, , drop = FALSE]
  
  # If matching failed, try CHROM:POS
  if (nrow(matched) == 0 && "Marker_CHROM_POS" %in% colnames(dosage)) {
    matched <- dosage[dosage$Marker_CHROM_POS %in% top_markers, , drop = FALSE]
  }
  
  n_found <- nrow(matched)
  cat("   Found", n_found, "of", length(top_markers), "markers\n")
  
  if (n_found == 0) {
    warning("No markers matched for ", base, ". Check marker ID format.")
    return(NULL)
  }
  
  # Keep original dosage columns, remove helper column if present
  if ("Marker_CHROM_POS" %in% colnames(matched)) {
    matched$Marker_CHROM_POS <- NULL
  }
  
  out_file <- file.path(
    OUT_DIR,
    paste0(trait, "_top", k, "_dosage.csv")
  )
  
  fwrite(matched, out_file)
  cat("   Saved:", out_file, "\n")
  
  data.frame(
    Trait = trait,
    SNP_Set = paste0("Top", k),
    Expected_n = length(top_markers),
    Extracted_n = n_found,
    File = out_file
  )
}

summary_list <- lapply(top_files, extract_one)
summary_df <- bind_rows(summary_list)

fwrite(summary_df, file.path(OUT_DIR, "feature_set_extraction_summary.csv"))

cat("[DONE] Extraction finished.\n")
cat("[DONE] Summary saved to:", file.path(OUT_DIR, "feature_set_extraction_summary.csv"), "\n")
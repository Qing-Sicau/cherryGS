# ==============================================================================
#                      SETUP AND INSTALLATION
# ==============================================================================
# (This part remains the same)
if (!requireNamespace("polymapR", quietly = TRUE)) install.packages("polymapR")
if (!requireNamespace("vcfR", quietly = TRUE)) install.packages("vcfR")
if (!requireNamespace("updog", quietly = TRUE)) install.packages("updog")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
if (!requireNamespace("rtracklayer", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install("rtracklayer")
}

# Load libraries for the session
library(polymapR)
library(vcfR)
library(updog)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rtracklayer)

# It's good practice to set a seed for reproducibility
set.seed(12345)
n_cpu=parallelly::availableCores()/2

# =====================================================================
#           STEP 1: LOAD DOSAGE DATA
# =====================================================================
ALL_dosages <- read.csv("polyrad_hn_nh_out_dosages_QTL_dosages.csv", sep = ",",  header = TRUE, 
                        stringsAsFactors =FALSE)
ALL_dosages <- t(ALL_dosages)

ALL_dosages <- ALL_dosages %>%
  dplyr::mutate(snp = paste0(CHROM, "_", POS)) %>%
  dplyr::select(snp, 5:ncol(ALL_dosages))

row.names(ALL_dosages) <- ALL_dosages$snp

# direct import from polyRAD output
ALL_dosages <- readRDS("polyrad_hn_nh_out_polymapR_input.rds")

gff_file <- "Ppse_renamed.gff"
output_dir <- "polymapR_analysis_corrected"
dir.create(output_dir, showWarnings = FALSE)

dim(ALL_dosages)

# --- Define Parents and F1 Populations ---
P1 <- "HF1"
P2 <- "NZH2"
parents_nh <- c(P1, P2)
offspring_f1_nh <- colnames(ALL_dosages)[grepl("^NH|^HN",colnames(ALL_dosages))]
dosage_matrix_nh <- as.matrix(ALL_dosages[, c(parents_nh, offspring_f1_nh)])
dosage_matrix_nh <- as.matrix(ALL_dosages)

# --- F1 SKEWNESS ---

F1checked <- checkF1(dosage_matrix = dosage_matrix_nh,
                     parent1 = P1,
                     parent2 = P2,
                     F1 = offspring_f1_nh,
                     polysomic = TRUE, 
                     disomic = FALSE, 
                     mixed = FALSE, 
                     ploidy = 4)

# select those passed the chi square test markers

filtered_F1 <- subset(F1checked$checked_F1,
                      qall_mult > 0.9 &
                      qall_weights > 0.9 &
                      Pvalue_bestfit > 0.1 &
                      (matchParent_bestfit == "Yes" | matchParent_bestfit == "OneOk"))

# filtered_F1 <- subset(F1checked$checked_F1, 
#                       qall_mult > 0 &
#                         qall_weights > 0.5 &
#                         Pvalue_bestfit > 0.1 &
#                         (matchParent_bestfit == "Yes" | matchParent_bestfit == "OneOk")) 
dim(filtered_F1)

selected_markers <- filtered_F1$MarkerName
filtered_nh_dosages <- dosage_matrix_nh[rownames(dosage_matrix_nh) %in% selected_markers,]

# QC for all markers and individuals
PCA_progeny(dosage_matrix = filtered_nh_dosages, 
            highlight = list(c(P1, P2)), 
            colors = "red")

mds <- marker_data_summary(dosage_matrix = filtered_nh_dosages,
                           ploidy = 4,
                           pairing = "random",
                           parent1 = P1,
                           parent2 = P2,
                           progeny_incompat_cutoff = 0.05)

pq_before_convert <- parental_quantities(dosage_matrix = filtered_nh_dosages, 
                                        las = 2, parent1 = P1, parent2 = P2) 

segregating_data <- convert_marker_dosages(dosage_matrix = filtered_nh_dosages, 
                                           ploidy = 4, parent1 = P1, parent2 = P2)

pq_after_convert <- parental_quantities(dosage_matrix = segregating_data, 
                                         las = 2, parent1 = P1, parent2 = P2)

screened_data <- screen_for_NA_values(dosage_matrix = segregating_data, 
                                      parentnames = parents_nh,
                                      margin = 1, # margin 1 means markers
                                      cutoff =  0.20,
                                      print.removed = FALSE)
                                
screened_data2 <- screen_for_NA_values(dosage_matrix = screened_data, 
                                       cutoff = 0.1, 
                                       parentnames = parents_nh,
                                       margin = 2, 
                                       print.removed = FALSE)

screened_data3 <- screen_for_duplicate_individuals(dosage_matrix = screened_data2, 
                                                   cutoff = 0.95, 
                                                   plot_cor = T)

screened_data4 <- screen_for_duplicate_markers(dosage_matrix = screened_data3)

filtered_data <- screened_data4$filtered_dosage_matrix

pq_screened_data <- parental_quantities(dosage_matrix = filtered_data,parent1 = P1, parent2 = P2)

# linkage constraction

SN_SN_P1 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    parent1 = P1,
                    parent2 = P2,
                    which_parent = 1,
                    ploidy = 4,
                    pairing = "random",
                    ncores = 16)

r_LOD_plot(linkage_df = SN_SN_P1, r_max = 0.5)


P1deviations <- SNSN_LOD_deviations(linkage_df = SN_SN_P1,
                                    ploidy = 4,
                                    N = ncol(filtered_data) - 2, #The F1 population size
                                    alpha = c(0.05,0.2),
                                    plot_expected = TRUE,
                                    phase="coupling")



P1_homologues <- cluster_SN_markers(linkage_df = SN_SN_P1, 
                                    LOD_sequence = seq(3, 15, 0.2), 
                                    LG_number = 8,
                                    ploidy = 4,
                                    parentname = P1,
                                    plot_network = FALSE,
                                    plot_clust_size = F,min_clust_size = 7)


P1_hom_LOD4.6 <- P1_homologues[["4.6"]]

t <- table(P1_hom_LOD4.6$cluster)
t

print(paste("Number of clusters:",length(t)))

LGHomDf_P1 <- define_LG_structure(cluster_list = P1_homologues, 
                                  LOD_chm = 4.6, 
                                  LOD_hom = 7, 
                                  LG_number = 8)

head(LGHomDf_P1)


#### 第二种方法，先聚类，再分染色体
SN_SN_P1_coupl <- SN_SN_P1[SN_SN_P1$phase == "coupling",] # select only markerpairs in coupling
P1_homologues_1 <- cluster_SN_markers(linkage_df = SN_SN_P1_coupl, 
                                      LOD_sequence = c(3:12), 
                                      LG_number = 8,
                                      ploidy = 4,
                                      parentname = P1,
                                      plot_network = FALSE,
                                      plot_clust_size = FALSE)

SN_DN_P1 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    markertype2 = c(2,0),parent1 = P1, parent2 = P2,
                    which_parent = 1,
                    ploidy = 4,
                    pairing = "random")

LGHomDf_P1_1 <- bridgeHomologues(cluster_stack = P1_homologues_1[["6"]], 
                                 linkage_df = SN_DN_P1, 
                                 LOD_threshold = 4, 
                                 automatic_clustering = TRUE, 
                                 LG_number = 5,
                                 parentname = "P1")




SN_DN_P1 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    markertype2 = c(2,0),
                    parent1 = parent_HF1, parent2 = parent_NZH2,
                    ploidy = 4,
                    pairing = "random",
                    ncores=3)




# ==============================================================================
#             STEP 2: Data Screening and Quality Control
# ==============================================================================

screen_data <- function(dosage_matrix, population_name, parent1_name, parent2_name) {
  cat(paste("\n--- Screening Population:", population_name, "---\n"))
  
  if (nrow(dosage_matrix) == 0 || ncol(dosage_matrix) == 0) {
    warning(paste("Input matrix for", population_name, "is empty. Skipping screening. Please check your sample names."), call. = FALSE)
    # Return an empty list structure to allow the script to continue
    return(list(screened_data = dosage_matrix, binned_markers = list()))
  }
  
  # 1. Screen for missing values (NA)
  screened_markers <- screen_for_NA_values(
    dosage_matrix = dosage_matrix,
    margin = 1,
    cutoff = 0.1,
    parentnames = c(parent1_name, parent2_name)
  )
  cat(paste(nrow(dosage_matrix) - nrow(screened_markers), "markers removed due to high NA.\n"))
  
  # If screening removed all individuals, handle gracefully
  if (ncol(screened_markers) == 0) {
    warning(paste("All individuals removed for", population_name, "after marker screening. Skipping further steps for this population."), call. = FALSE)
    return(list(screened_data = screened_markers, binned_markers = list()))
  }
  
  # 2. Remove individuals with > 10% missing data
  screened_inds <- screen_for_NA_values(
    dosage_matrix = screened_markers,
    margin = 2,
    cutoff = 0.1,
    parentnames = c(parent1_name, parent2_name)
  )
  cat(paste(ncol(screened_markers) - ncol(screened_inds), "individuals removed due to high NA.\n"))
  
  # If screening removed all markers/individuals, handle gracefully
  if (nrow(screened_inds) == 0 || ncol(screened_inds) == 0) {
    warning(paste("All data removed for", population_name, "after individual screening. Skipping further steps for this population."), call. = FALSE)
    return(list(screened_data = screened_inds, binned_markers = list()))
  }
  
  # 3. Screen for duplicate markers
  screened_dups <- screen_for_duplicate_markers(screened_inds, plot_cluster_size = FALSE)
  screened_data <- screened_dups$filtered_dosage_matrix
  cat(paste(nrow(screened_inds) - nrow(screened_data), "duplicate markers removed.\n"))
  
  binned_markers <- screened_dups$bin_list
  
  cat(paste("Screening complete for", population_name, ". Final matrix size:",
            nrow(screened_data), "markers x", ncol(screened_data), "individuals.\n"))
  
  return(list(screened_data = screened_data, binned_markers = binned_markers))
}

results_pop1 <- screen_data(dosage_matrix_pop1, "HN (HF1xNZH2)", parent_HF1, parent_NZH2)
screened_data_pop1 <- results_pop1$screened_data
binned_markers_pop1 <- results_pop1$binned_markers



# ==============================================================================
#                   STEP 3: Segregation Analysis
# ==============================================================================

##################   fixed checkF1   ##################
checkF1_fixed <- function(input_type = "discrete", dosage_matrix, probgeno_df, parent1, parent2, F1, 
                          ancestors = character(0), polysomic, disomic, mixed, ploidy, ploidy2, 
                          outfile = "", critweight = c(1, 0.4, 0.4), Pvalue_threshold = 1e-04, 
                          fracInvalid_threshold = 0.05, fracNA_threshold = 0.25, shiftmarkers, 
                          parentsScoredWithF1 = TRUE, shiftParents = parentsScoredWithF1, 
                          showAll = FALSE, append_shf = FALSE) {
  
  input_type <- match.arg(input_type, choices = c("discrete", "probabilistic"))
  if (input_type == "discrete") {
    dosage_matrix <- polymapR:::test_dosage_matrix(dosage_matrix)
  } else {
    probgeno_df <- polymapR:::test_probgeno_df(probgeno_df)
  }
  
  if (ploidy%%2 != 0) stop("checkF1: odd ploidy not allowed")
  if (missing(ploidy2) || is.na(ploidy2)) {
    ploidy2 <- ploidy
  } else if (ploidy2%%2 != 0) {
    stop("checkF1: odd ploidy2 not allowed")
  }
  
  ploidyF1 <- (ploidy + ploidy2)/2
  if (!polysomic && !disomic && !mixed) {
    stop("checkF1: at least one of polysomic, disomic and mixed must be TRUE")
  }
  
  seginfo <- calcSegtypeInfo(ploidy, ploidy2)
  allsegtypenames <- names(seginfo)
  seginfo <- polymapR:::selSegtypeInfo(seginfo, polysomic, disomic, mixed)
  seginfoSummary <- polymapR:::segtypeInfoSummary(seginfo)
  
  if (is.null(critweight) || is.na(critweight[1])) {
    critweight <- NA
  } else {
    if (!is.numeric(critweight) || length(critweight) != 3 || 
        sum(is.na(critweight)) > 0 || sum(critweight) == 0) {
      stop("invalid critweight")
    }
  }
  
  if (missing(shiftmarkers)) shiftmarkers <- NA
  if (is.data.frame(shiftmarkers)) {
    if (sum(is.na(match(c("MarkerName", "shift"), names(shiftmarkers)))) > 0) {
      stop("checkF1: shiftmarkers must have columns MarkerName and shift")
    }
    shiftmarkers$shift[is.na(shiftmarkers$shift)] <- 0
    if (nrow(shiftmarkers) == 0) {
      shiftmarkers <- NA
    } else {
      if (sum(is.na(shiftmarkers$shift)) > 0 || 
          sum(!(shiftmarkers$shift %in% ((-ploidyF1):ploidyF1))) > 0) {
        stop("checkF1: shiftmarkers contains invalid shift values")
      }
      shiftmarkers$MarkerName <- as.character(shiftmarkers$MarkerName)
      if (length(unique(shiftmarkers$MarkerName)) < nrow(shiftmarkers)) {
        stop("checkF1: some markernames occur more than once in shiftmarkers")
      }
    }
  }
  
  seg_invalidrate <- 0.03
  file.del <- is.na(outfile) || outfile == ""
  if (file.del) outfile <- "checkF1.tmp"
  if (!polymapR:::checkFilename(outfile)) {
    stop(paste("checkF1: cannot write file", outfile))
  }
  
  if (missing(parent1) || is.logical(parent1)) parent1 <- character(0)
  if (missing(parent2) || is.logical(parent2)) parent2 <- character(0)
  if (missing(ancestors) || is.logical(ancestors)) ancestors <- character(0)
  parent1 <- as.character(parent1[!is.na(parent1)])
  parent2 <- as.character(parent2[!is.na(parent2)])
  ancestors <- as.character(ancestors[!is.na(ancestors)])
  
  qnames <- c("q1_segtypefit", " q2_parents", "q3_fracscored", "qall_mult")
  if (length(critweight) == 3) qnames[5] <- "qall_weights"
  
  if (input_type == "discrete") {
    mrknames <- rownames(dosage_matrix)
  } else {
    mrknames <- sort(as.character(unique(probgeno_df$MarkerName)))
  }
  
  createResultsdf <- function(mrkcount) {
    mat <- matrix(integer((2 + ploidyF1 + 2 + length(parent1) + length(parent2) + 
                             length(ancestors)) * mrkcount), nrow = mrkcount)
    colnames(mat) <- c("parent1", "parent2", paste("F1", 0:ploidyF1, sep = "_"), 
                       "F1_NA", parent1, parent2, ancestors)
    segdf <- data.frame(frqInvalid = character(mrkcount), 
                        Pvalue = character(mrkcount), 
                        matchParent = factor(rep(NA, mrkcount), 
                                             levels = c("No", "OneOK", "Unknown", "Yes")))
    bres <- data.frame(m = integer(mrkcount), MarkerName = character(mrkcount), mat)
    
    if (showAll) {
      for (sg in 1:length(seginfo)) {
        df <- segdf
        names(df) <- paste(names(df), names(seginfo)[sg], sep = "_")
        bres <- data.frame(bres, df)
      }
    }
    
    segdf <- data.frame(fit = factor(rep(NA, mrkcount), levels = names(seginfo)), segdf)
    for (fi in c("bestfit", "bestParentfit")) {
      df <- segdf
      names(df) <- c(fi, paste(names(df)[2:4], fi, sep = "_"))
      bres <- data.frame(bres, df)
    }
    
    bres <- data.frame(bres, 
                       q1_segtypefit = character(mrkcount), 
                       q2_parents = character(mrkcount), 
                       q3_fracscored = character(mrkcount), 
                       qall_mult = character(mrkcount))
    if (length(critweight) == 3) {
      bres <- data.frame(bres, qall_weights = character(mrkcount))
    }
    if (is.data.frame(shiftmarkers)) {
      bres <- data.frame(bres, shift = integer(mrkcount))
    }
    bres
  }
  
  segtypeBestSelcrit <- function(candidates) {
    if (length(candidates) == 0) return(0)
    candSelcrit <- selcrit[candidates]
    candidates[which.max(candSelcrit)]
  }
  
  compareFit <- function(newsegtype, oldsegtype) {
    (results$fracInvalid[newsegtype] <= max(0.05, 1.5 * results$fracInvalid[oldsegtype])) && 
      (results$Pvalue[newsegtype] >= min(0.01, 0.1 * results$Pvalue[oldsegtype]))
  }
  
  shiftdosages <- function(dosages, shift, ploidyF1) {
    dosages <- dosages + shift
    below <- !is.na(dosages) & dosages < 0
    dosages[below] <- 0
    above <- !is.na(dosages) & dosages > ploidyF1
    dosages[above] <- ploidyF1
    dosages
  }
  
  batchsize <- 100
  batchnr <- 1
  while (batchsize * (batchnr - 1) < length(mrknames)) {
    minmrk <- batchsize * (batchnr - 1) + 1
    maxmrk <- min(length(mrknames), batchsize * batchnr)
    
    if (input_type == "discrete") {
      batchscores <- dosage_matrix[mrknames[minmrk:maxmrk], , drop = FALSE]
    } else {
      batchscores <- probgeno_df[probgeno_df$MarkerName %in% mrknames[minmrk:maxmrk], ]
    }
    
    bres <- createResultsdf(maxmrk - minmrk + 1)
    
    count_probabi <- function(ploidy, sc) {
      geno.count <- c()
      for (i in 0:ploidy) {
        geno.ea <- sum(as.numeric(as.character(sc[[paste0("P", i)]])), na.rm = TRUE)
        geno.count <- c(geno.count, geno.ea)
      }
      return(geno.count)
    }
    
    for (mrk in minmrk:maxmrk) {
      if (input_type == "discrete") {
        mrknr <- which(rownames(batchscores) == mrknames[mrk])
        sc <- data.frame(SampleName = colnames(batchscores), 
                         geno = as.integer(batchscores[mrknr, ]))
      } else {
        sc <- batchscores[batchscores$MarkerName == mrknames[mrk], ]
        if ("marker" %in% names(sc)) {
          mrknr <- sc$marker[1]
        } else {
          mrknr <- mrk
        }
      }
      
      parent.geno <- list()
      if (input_type == "discrete") {
        parent.geno[[1]] <- sc$geno[match(parent1, sc$SampleName)]
        parent.geno[[2]] <- sc$geno[match(parent2, sc$SampleName)]
        ancestors.geno <- sc$geno[match(ancestors, sc$SampleName)]
        F1.geno <- sc$geno[sc$SampleName %in% F1]
      } else {
        parent.geno[[1]] <- sc$maxgeno[match(parent1, sc$SampleName)]
        parent.geno[[2]] <- sc$maxgeno[match(parent2, sc$SampleName)]
        ancestors.geno <- sc$maxgeno[match(ancestors, sc$SampleName)]
        F1.geno <- sc[sc$SampleName %in% F1, paste0("P", seq(0, max(ploidy, ploidy2), 1))]
      }
      
      shift <- 0
      if (is.data.frame(shiftmarkers)) {
        whichshift <- which(shiftmarkers$MarkerName == mrknames[mrk])
        if (length(whichshift) == 1) {
          shift <- shiftmarkers$shift[whichshift]
          if (shift != 0) {
            F1.geno <- shiftdosages(F1.geno, shift, ploidyF1)
            if (shiftParents) {
              parent.geno[[1]] <- shiftdosages(parent.geno[[1]], shift, ploidyF1)
              parent.geno[[2]] <- shiftdosages(parent.geno[[2]], shift, ploidyF1)
              ancestors.geno <- shiftdosages(ancestors.geno, shift, ploidyF1)
            }
          }
        }
      }
      
      par.geno <- c(0, 0)
      par.lowconf.geno <- c(0, 0)
      par.conflicts <- c(FALSE, FALSE)
      par.NAfrac <- c(0.5, 0.5)
      
      for (parent in 1:2) {
        parresult <- polymapR:::getConsensusGeno(geno = parent.geno[[parent]], 
                                                 maxNAfrac = 0.499, 
                                                 lowconf.NAfrac = 0.751)
        par.geno[parent] <- parresult$geno
        par.lowconf.geno[parent] <- as.numeric(as.character(parresult$lowconf.geno))
        par.conflicts[parent] <- parresult$conflict
        par.NAfrac[parent] <- parresult$NAfrac
      }
      
      if (input_type == "discrete") {
        F1.naCount <- sum(is.na(F1.geno))
        F1.nobs <- length(F1.geno) - F1.naCount
        F1.counts <- tabulate(bin = F1.geno + 1, nbins = ploidyF1 + 1)
      } else {
        F1.naCount <- sum(rowSums(is.na(F1.geno)) == ploidyF1 + 1)
        F1.nobs <- nrow(F1.geno) - F1.naCount
        F1.counts <- count_probabi(ploidy = ploidyF1, sc = F1.geno)
        proba_correct <- 0.05 * nrow(F1.geno)/(ploidy + 1)
        F1.counts[F1.counts < proba_correct] <- 0
        F1.counts <- (F1.counts/sum(F1.counts)) * F1.nobs
      }
      
      bestfit <- NA
      bestParentfit <- NA
      q <- rep(NA, length(qnames))
      results <- data.frame(segtype = names(seginfo), 
                            fracInvalid = rep(1, length(seginfo)), 
                            invalidP = rep(0, length(seginfo)), 
                            Pvalue = rep(0, length(seginfo)), 
                            matchParents = I(as.character(rep(NA, length(seginfo)))))
      
      if (F1.nobs > 10) {
        for (s in 1:length(seginfo)) {
          results$matchParents[s] <- polymapR:::getMatchParents(parGeno = par.geno, 
                                                                seginfoItem = seginfo[[s]])
          exp.geno <- seginfo[[s]]$expgeno
          if (input_type == "discrete") {
            F1.invalid <- length(F1.geno[!(F1.geno %in% exp.geno)]) - F1.naCount
          } else {
            F1.invalid <- sum(F1.counts) - sum(F1.counts[exp.geno + 1])
          }
          results$fracInvalid[s] <- F1.invalid/F1.nobs
          
          if (F1.nobs - F1.invalid > 0) {
            results$invalidP[s] <- pbinom(q = F1.nobs - F1.invalid, 
                                          size = F1.nobs, 
                                          prob = 1 - seg_invalidrate)
            
            if (length(exp.geno) == 1) {
              results$Pvalue[s] <- 1
            } else {
              if (sum(F1.counts[exp.geno + 1]) == 0) {
                results$Pvalue[s] <- 0
                results$invalidP[s] <- 0
              } else {
                suppressWarnings(
                  results$Pvalue[s] <- chisq.test(F1.counts[exp.geno + 1], 
                                                  p = seginfo[[s]]$freq[exp.geno + 1])$p.value)
              }
            }
          }
        }
        
        selcrit <- results$invalidP * results$Pvalue
        bestfit <- which.max(selcrit)
        if (bestfit == 0) {
          stop(paste("Error in checkF1: bestfit is 0 at marker", mrknames[mrk]))
        }
        
        ParentFit <- which(results$matchParents %in% c("Yes", "OneOK", "Unknown"))
        bestParentfit <- segtypeBestSelcrit(ParentFit)
        if (bestParentfit == 0) {
          stop(paste("Error in checkF1: bestParentfit is 0 at marker", mrknames[mrk]))
        }
        
        lowParentFit <- NA
        lowc <- which(!is.na(par.lowconf.geno))
        
        if (length(lowc) > 0) {
          if (length(lowc) == 1) {
            if (is.na(par.geno[3 - lowc])) {
              low.segtypes <- seginfoSummary$segtypenr[seginfoSummary[, 2 + lowc] == 
                                                         par.lowconf.geno[lowc]]
            } else {
              low.segtypes <- seginfoSummary$segtypenr[(seginfoSummary[, 2 + lowc] == 
                                                          par.lowconf.geno[lowc]) & 
                                                         (seginfoSummary[, 5 - lowc] == 
                                                            par.geno[3 - lowc])]
            }
            
            if (length(low.segtypes) > 0) {
              lowParentfit <- segtypeBestSelcrit(low.segtypes)
              if (compareFit(lowParentfit, bestParentfit)) {
                par.geno[lowc] <- par.lowconf.geno[lowc]
                bestParentfit <- lowParentfit
                results$matchParents[bestParentfit] <- 
                  polymapR:::getMatchParents(parGeno = par.geno, 
                                             seginfoItem = seginfo[[bestParentfit]])
              }
            }
          } else {
            low.segtypes <- seginfoSummary$segtypenr[(seginfoSummary[, 3] == 
                                                        par.lowconf.geno[1]) & 
                                                       (seginfoSummary[, 4] == 
                                                          par.lowconf.geno[2])]
            lowParentfit <- segtypeBestSelcrit(low.segtypes)
            
            if (compareFit(lowParentfit, bestParentfit)) {
              par.geno <- par.lowconf.geno
              bestParentfit <- lowParentfit
              results$matchParents[bestParentfit] <- 
                polymapR:::getMatchParents(parGeno = par.geno, 
                                           seginfoItem = seginfo[[bestParentfit]])
            } else {
              lowParentfit <- c(0, 0)
              for (p in 1:2) {
                low.segtypes <- seginfoSummary$segtypenr[seginfoSummary[, 2 + p] == 
                                                           par.lowconf.geno[p]]
                lowParentfit[p] <- segtypeBestSelcrit(low.segtypes)
                if (!compareFit(lowParentfit[p], bestParentfit)) {
                  lowParentfit[p] <- 0
                }
              }
              
              p <- which(lowParentfit != 0)
              if (length(p) == 1) {
                par.geno[p] <- par.lowconf.geno[p]
                bestParentfit <- lowParentfit[p]
                results$matchParents[bestParentfit] <- 
                  polymapR:::getMatchParents(parGeno = par.geno, 
                                             seginfoItem = seginfo[[bestParentfit]])
              }
            }
          }
        }
        
        if (input_type == "discrete") {
          F1_NAfrac <- F1.naCount/length(F1.geno)
        } else {
          F1_NAfrac <- F1.naCount/nrow(F1.geno)
        }
        
        q <- polymapR:::calc_qall(Pvalue_threshold, fracInvalid_threshold, 
                                  fracNA_threshold, Pvalue = results$Pvalue[bestParentfit], 
                                  fracInvalid = results$fracInvalid[bestParentfit], 
                                  F1.NAfrac = F1_NAfrac, 
                                  matchParents = results$matchParents[bestParentfit], 
                                  bestfit = bestfit, bestParentfit = bestParentfit, 
                                  par.conflicts = par.conflicts, par.NAfrac = par.NAfrac, 
                                  critweight = critweight, 
                                  parentsScoredWithF1 = parentsScoredWithF1)
      }
      
      bix <- mrk - minmrk + 1
      bres$m[bix] <- mrknr
      if (shift == 0 || !append_shf) {
        bres$MarkerName[bix] <- mrknames[mrk]
      } else {
        bres$MarkerName[bix] <- paste(mrknames[mrk], "shf", sep = "_")
      }
      
      bres[bix, 3:4] <- par.geno
      bres[bix, 5:(5 + ploidyF1)] <- F1.counts
      bres$F1_NA[bix] <- F1.naCount
      
      startcol <- ploidyF1 + 7
      if (length(parent1) > 0) {
        bres[bix, startcol:(startcol - 1 + length(parent1))] <- parent.geno[[1]]
        startcol <- startcol + length(parent1)
      }
      
      if (length(parent2) > 0) {
        bres[bix, startcol:(startcol - 1 + length(parent2))] <- parent.geno[[2]]
        startcol <- startcol + length(parent2)
      }
      
      if (length(ancestors) > 0) {
        bres[bix, startcol:(startcol - 1 + length(ancestors))] <- ancestors.geno[[1]]
        startcol <- startcol + length(ancestors)
      }
      
      if (showAll) {
        bres[bix, startcol + seq(0, by = 3, length.out = length(seginfo))] <- 
          sprintf("%.4f", results$fracInvalid)
        bres[bix, startcol + seq(1, by = 3, length.out = length(seginfo))] <- 
          sprintf("%.4f", results$Pvalue)
        bres[bix, startcol + seq(2, by = 3, length.out = length(seginfo))] <- 
          results$matchParents
        startcol <- startcol + 3 * length(seginfo)
      }
      
      if (is.na(bestfit)) {
        bres[bix, startcol:(startcol + 3)] <- rep(NA, 4)
      } else {
        bres[bix, startcol] <- results$segtype[bestfit]
        bres[bix, startcol + 1] <- sprintf("%.4f", results$fracInvalid[bestfit])
        bres[bix, startcol + 2] <- sprintf("%.4f", results$Pvalue[bestfit])
        bres[bix, startcol + 3] <- results$matchParents[bestfit]
      }
      
      startcol <- startcol + 4
      
      if (is.na(bestParentfit)) {
        bres[bix, startcol:(startcol + 3)] <- rep(NA, 4)
      } else {
        bres[bix, startcol] <- results$segtype[bestParentfit]
        bres[bix, startcol + 1] <- sprintf("%.4f", results$fracInvalid[bestParentfit])
        bres[bix, startcol + 2] <- sprintf("%.4f", results$Pvalue[bestParentfit])
        bres[bix, startcol + 3] <- results$matchParents[bestParentfit]
      }
      
      startcol <- startcol + 4
      
      if (is.na(q[1])) {
        bres[bix, startcol:(startcol - 1 + length(q))] <- rep(NA, length(q))
      } else {
        bres[bix, startcol:(startcol - 1 + length(q))] <- sprintf("%.4f", q)
      }
      
      startcol <- startcol + length(q)
      
      if (is.data.frame(shiftmarkers)) {
        bres[bix, startcol] <- shift
      }
    }
    
    if (batchnr == 1) {
      write.table(bres, file = outfile, quote = FALSE, sep = "\t", na = "", 
                  row.names = FALSE, col.names = TRUE)
    } else {
      write.table(bres, file = outfile, append = TRUE, quote = FALSE, sep = "\t", 
                  na = "", row.names = FALSE, col.names = FALSE)
    }
    
    batchnr <- batchnr + 1
  }
  
  output <- read.table(outfile, header = TRUE, sep = "\t", na.strings = "", 
                       check.names = FALSE)
  
  if (input_type == "discrete") {
    output <- polymapR:::chk2integer(output)
  }
  
  output$bestfit <- factor(as.character(output$bestfit), levels = allsegtypenames)
  output$bestParentfit <- factor(as.character(output$bestParentfit), levels = allsegtypenames)
  
  if (file.del) file.remove(outfile)
  
  return(invisible(list(checked_F1 = output, 
                        meta = list(parent1 = parent1, parent2 = parent2, F1 = F1, 
                                    ancestors = ancestors, polysomic = polysomic, 
                                    disomic = disomic, mixed = mixed, ploidy = ploidy, 
                                    ploidy2 = ploidy2, outfile = outfile, 
                                    critweight = critweight, 
                                    Pvalue_threshold = Pvalue_threshold, 
                                    fracInvalid_threshold = fracInvalid_threshold, 
                                    fracNA_threshold = fracNA_threshold, 
                                    shiftmarkers = shiftmarkers, 
                                    parentsScoredWithF1 = parentsScoredWithF1))))
}

############### end of function ######################


# --- Population 1: HN (HF1 x NZH2) ---
chk_pop1 <- checkF1_fixed(
  dosage_matrix = screened_data_pop1,
  parent1 = parent_HF1,
  parent2 = parent_NZH2,
  F1 = f1_pop_samples[f1_pop_samples %in% colnames(screened_data_pop1)],
  ploidy = 4,
  polysomic = TRUE,
  disomic = FALSE, # Assuming polysomic inheritance
  mixed = FALSE
)

seg_ok_pop1 <- chk_pop1$checked_F1[
  chk_pop1$checked_F1$Pvalue_bestParentfit > 0.01 & 
    chk_pop1$checked_F1$frqInvalid_bestParentfit < 0.1 &
    chk_pop1$checked_F1$qall_mult > 0,
]

dosage_clean_pop1 <- screened_data_pop1[rownames(screened_data_pop1) %in% seg_ok_pop1$MarkerName, ]

dosage_clean_pop1 <- screen_for_duplicate_individuals(dosage_matrix = dosage_clean_pop1, 
                                                      cutoff = 0.95, 
                                                      plot_cor = TRUE)
cat("Segregation analysis complete.\n")


# ==============================================================================
#               STEP 4: Individual Linkage Map Construction (Final Integrated Version)
# ==============================================================================

# --- User Input Required ---
# IMPORTANT: Replace with the haploid chromosome number of your species
LG_number <- 8 

# --- Re-usable mapping function (Includes all data sanitation steps) ---
build_parental_maps <- function(dosage_matrix, segregation_info, parent1_name, parent2_name, pop_name) {
  cat(paste("\n--- Building maps for population:", pop_name, "---\n"))

    cat("--- Performing Comprehensive Data Integrity Check ---\n")
  
  # Step 1: Hard filter for markers with NA dosage scores in either parent column.
  cat("Initial number of markers:", nrow(dosage_matrix), "\n")
  dosage_matrix <- dosage_matrix[
    !is.na(dosage_matrix[, parent1_name]) & !is.na(dosage_matrix[, parent2_name]),
  ]
  cat("Markers remaining after cleaning parental NAs:", nrow(dosage_matrix), "\n")
  
  # Step 2: Use screen_for_NA_values to filter markers with too many NAs in the progeny.
  dosage_matrix <- screen_for_NA_values(
    dosage_matrix = dosage_matrix, 
    margin = 1, # '1' checks markers (rows)
    cutoff = 0.15,
    parentnames = c(parent1_name, parent2_name),
    print.removed = TRUE
  )
  cat("Markers remaining after cleaning progeny NAs:", nrow(dosage_matrix), "\n")
  
  # Step 3: Synchronize the segregation table and the dosage matrix to a final, clean set.
  seg_table <- as.data.frame(segregation_info$checked_F1)
  
  # Step 3a: First, clean the seg_table to remove any markers that have NA for a name.
  seg_table <- seg_table[!is.na(seg_table$MarkerName), ]
  
  # Step 3b: Find the intersection of marker names that exist in BOTH the cleaned dosage_matrix
  # AND the cleaned seg_table. This is the definitive list of valid markers.
  valid_markers_from_dosage <- rownames(dosage_matrix)
  valid_markers_from_seg <- seg_table$MarkerName
  final_valid_marker_names <- intersect(valid_markers_from_dosage, valid_markers_from_seg)
  
  # Step 3c: Re-synchronize BOTH the dosage_matrix and seg_table to this final list of names.
  dosage_matrix <- dosage_matrix[final_valid_marker_names, ]
  seg_table <- seg_table[seg_table$MarkerName %in% final_valid_marker_names, ]
  
  cat("Final number of markers for mapping after full synchronization:", nrow(seg_table), "\n")

  
  parents <- c(parent1_name, parent2_name)
  map_lists <- list()
  
  plot_LOD_profile <- function(cluster_list) {
    # Create a data frame for plotting
    lod_data <- data.frame(
      LOD_score = as.numeric(names(cluster_list)),
      num_clusters = sapply(cluster_list, function(df) length(unique(df$cluster)))
    )
    
    # Generate the plot
    p <- ggplot(lod_data, aes(x = LOD_score, y = num_clusters)) +
      geom_line(color = "blue", size = 1) +
      geom_point(color = "red", size = 3) +
      geom_text(aes(label = num_clusters), vjust = -1.5, color = "black") +
      labs(
        title = "Number of Clusters vs. LOD Score",
        subtitle = "Identify the LOD score where cluster number equals your haploid chromosome number",
        x = "LOD Score Threshold",
        y = "Number of Resulting Clusters"
      ) +
      theme_minimal() +
      scale_x_continuous(breaks = lod_data$LOD_score) # Ensure all LOD scores are shown as ticks
    
    return(p)
  }
  
  for (i in 1:2) {
    target_parent <- parents[i]
    cat(paste("\n- Building map for parent:", target_parent, "-\n"))
    
    # --- Marker Filtering
    if (i == 1) {
      target_marker_names <- seg_table$MarkerName[
        !is.na(seg_table$parent1) & seg_table$parent1 == 1 &
          !is.na(seg_table$parent2) & seg_table$parent2 == 0
      ]
    } else { # i == 2
      target_marker_names <- seg_table$MarkerName[
        !is.na(seg_table$parent1) & seg_table$parent1 == 0 &
          !is.na(seg_table$parent2) & seg_table$parent2 == 1
      ]
    }
    
    markers_to_map <- intersect(rownames(dosage_matrix), target_marker_names)
    if (length(markers_to_map) < 10) { # Increased threshold for robustness
      warning(paste("Insufficient simplex markers (<10) for", target_parent, ". Skipping map construction."), call. = FALSE)
      next 
    }
    cat(paste("Found", length(markers_to_map), "simplex markers for", target_parent, "to build skeleton map.\n"))
    subset_dosage_matrix <- dosage_matrix[markers_to_map, ]
    
    # --- Linkage Calculation ---
    SN_SN_links <- linkage(
      dosage_matrix = subset_dosage_matrix, markertype1 = c(1, 0),
      which_parent = i, parent1 = parent1_name, parent2 = parent2_name,
      ploidy = 4, ncores = n_cpu, LOD_threshold = 3
    )
    if (!is.data.frame(SN_SN_links) || nrow(SN_SN_links) == 0) {
      warning(paste("No linkages detected for parent:", target_parent, ". Skipping."), call. = FALSE)
      next
    }
    
    # --- Marker Clustering 
    SN_clusters <- cluster_SN_markers(
      linkage_df = SN_SN_links, LOD_sequence = seq(3, 10, 1),
      LG_number = LG_number, ploidy = 4, parentname = target_parent
    )
    
    cat("Generating LOD profile plot to determine optimal LOD_chm...\n")
    p <- plot_LOD_profile(SN_clusters)
    print(p) # Display the plot in an interactive session
    
    # Save the plot for inspection
    plot_filename <- file.path(output_dir, paste0("LOD_profile_", pop_name, "_", target_parent, ".png"))
    ggsave(plot_filename, p, width = 8, height = 6)
    cat(paste("LOD profile plot saved to:", plot_filename, "\n"))
    
    
    LOD_chm <- as.numeric(readline(prompt = paste("Examine the plot. Enter the optimal LOD_chm for", target_parent, "(where cluster number equals", LG_number, "): ")))
    LOD_hom <- LOD_chm + 2 # A common heuristic, can be adjusted.
    cat(paste("Using LOD_chm =", LOD_chm, "and LOD_hom =", LOD_hom, "\n"))


    LG_hom_structure <- define_LG_structure(
      cluster_list = SN_clusters, LOD_chm = LOD_chm,
      LOD_hom = LOD_hom, LG_number = LG_number
    )
    
    assigned_markers <- homologue_lg_assignment(
      dosage_matrix = dosage_matrix,
      LG_hom_stack = LG_hom_structure,
      assigned_list = list(),
      assigned_markertypes = list(),
      which_parent = i, 
      parent1 = parent1_name, 
      parent2 = parent2_name,
      ploidy = 4, 
      LG_number = LG_number,
      LOD_threshold = 5,
      convert_palindrome_markers = FALSE, 
      write_intermediate_files = FALSE
    )
    marker_assignments[[target_parent]] <- assigned_markers
    
    all_linkages_list <- finish_linkage_analysis(
      marker_assignment = assigned_markers,
      dosage_matrix = dosage_matrix,
      which_parent = i, 
      parent1 = parent1_name, 
      parent2 = parent2_name,
      ploidy = 4, 
      LG_number = LG_number, 
      ncores = 80
    )
    all_linkages[[target_parent]] <- all_linkages_list
    
    maplist <- MDSMap_from_list(linkage_list = all_linkages_list)
    map_lists[[target_parent]] <- maplist
  }
  return(list(map_lists = map_lists, all_linkages = all_linkages, marker_assignments = marker_assignments))
}

# --- Run the mapping pipeline ---
# The function calls do not need to be changed.
mapping_results_pop1 <- build_parental_maps(
  dosage_matrix = dosage_clean_pop1, 
  segregation_info = chk_pop1, 
  parent1_name = parent_HF1, 
  parent2_name = parent_NZH2, 
  pop_name = "HN (HF1xNZH2)"
)



# ==============================================================================
#                         STEP 5: Map Integration
# ==============================================================================
cat("\n--- Integrating Maps ---\n")
maplist_ref_parent <- mapping_results_pop1$map_lists[[parent_HF1]]
maplist_ref_parent_dense <- add_dup_markers(maplist = maplist_ref_parent,
                                            bin_list = binned_markers_pop1)
integrated_maplist <- maplist_ref_parent_dense$maplist
names(integrated_maplist) <- paste0("LG", 1:LG_number)
cat("Designated map from HF1 (Pop HN) as the reference integrated map.\n")

# ==============================================================================
#              STEP 6: Map Quality Assessment and Visualization
# ==============================================================================
cat("\n--- Quality Assessment for Integrated Map ---\n")
check_map(maplist = integrated_maplist,
          linkage_list = mapping_results_pop1$all_linkages[[parent_HF1]],
          prefix = "Integrated_Map_QC")

pdf("Integrated_Map_Visualization.pdf", width = 10, height = 8)
plot_map(maplist = integrated_maplist, bg_col = "lightgrey", main = "Integrated Map (ref: HF1 from HN cross)")
dev.off()

maplist_NZH2_pop1 <- mapping_results_pop1$map_lists[[parent_NZH2]]
maplist_HF1_pop3 <- mapping_results_pop3$map_lists[[parent_HF1]]

pdf("Map_Comparisons.pdf", width = 12, height = 9)
compare_maps(maplist = list("Ref: HF1 (from HN)" = integrated_maplist,
                            "NZH2 (from HN)" = maplist_NZH2_pop1,
                            "HF1 (from HP)" = maplist_HF1_pop3),
             bg.col = c("cornflowerblue", "khaki", "salmon"))
title("Comparison of Linkage Maps")
dev.off()
cat("Quality assessment and visualization plots have been generated.\n")

# ==============================================================================
#               STEP 7: Marey Maps (Genetic vs. Physical)
# ==============================================================================
cat("\n--- Preparing data for Marey Maps ---\n")
physical_map_df <- as.data.frame(rownames(dosage_matrix_full)) %>%
  rename(marker = 1) %>%
  tidyr::separate(marker, into = c("chromosome", "physical_pos"), sep = "_", remove = FALSE, convert = TRUE) %>%
  filter(!is.na(physical_pos))

if (nrow(physical_map_df) == 0) {
  cat("WARNING: Marker names in VCF do not seem to be in 'chr_pos' format. Attempting GFF-based parsing.\n")
  gff <- rtracklayer::import(gff_file)
  physical_map_df <- as.data.frame(gff) %>%
    filter(type == "gene" | type == "mRNA") %>%
    mutate(physical_pos = (start + end) / 2) %>%
    select(marker = ID, chromosome = seqnames, physical_pos) %>%
    distinct(marker, .keep_all = TRUE)
}
cat("Loaded", nrow(physical_map_df), "unique physical marker positions.\n")

genetic_map_df <- bind_rows(integrated_maplist, .id = "LG") %>%
  rename(marker = marker, genetic_pos = position)

marey_map_data <- inner_join(genetic_map_df, physical_map_df, by = "marker")

cat("Creating Marey plots with", nrow(marey_map_data), "common markers.\n")

marey_plot <- ggplot(marey_map_data, aes(x = physical_pos / 1e6, y = genetic_pos)) +
  geom_point(alpha = 0.6, color = "purple") +
  facet_wrap(~ LG, scales = "free_x", ncol = 4) +
  labs(
    title = "Marey Map: Genetic Position (cM) vs. Physical Position (Mb)",
    x = "Physical Position (Mb)",
    y = "Genetic Position (cM)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_rect(fill = "black"),
    strip.text = element_text(color = "white", face = "bold")
  )

ggsave("Marey_Map_Comparison.pdf", marey_plot, width = 16, height = 12)
ggsave("Marey_Map_Comparison.png", marey_plot, width = 16, height = 12, dpi = 300)

cat("\nAnalysis complete! All results are saved in the directory:", getwd(), "\n")
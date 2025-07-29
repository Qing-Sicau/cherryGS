# Genomic Prediction Tutorial in R
# Author: Adapted from Miguel Pérez-Enciso's tutorial
# Purpose: Demonstrate genomic prediction methods using R, including Ridge, LASSO, Elastic Net,
#          GBLUP, Bayesian methods, Single-Step GBLUP, and Deep Learning (MLP and CNN).
# Date: July 28, 2025
# Status: Corrected, with publication-quality plots.
# Annotations: This tutorial was authored by Dr. Miguel Pérez-Enciso, an influential scientist
#              in the field of genomic prediction. It is highly practical as it covers a wide
#              range of mainstream Genomic Selection (GS) models, from classical statistical
#              methods to cutting-edge deep learning. Let's explore the fascinating world of
#              genomic prediction step by step.
# QING CHEN, supnovel@sicau.edu.cn

# =====================================================================================
# Part 1: Introduction & Setup
# =====================================================================================
#
# ### 1. Why do we need Genomic Prediction (GS)?
# Before diving into the code, let's discuss a key question for breeding graduate students:
# What is GS, and why is it more powerful than Marker-Assisted Selection (MAS)?
#
# - The Limitation of MAS: MAS typically targets a few QTLs with large effects (major genes).
#   For instance, if you discover a gene for disease resistance, you can use tightly linked
#   markers to select offspring. However, for quantitative traits like yield or plant height,
#   which are controlled by hundreds or thousands of genes with small effects, MAS falls short.
#   The effect of a single minor gene is too small to be "discovered" through significance testing.
#
# - The Revolutionary Idea of GS: Proposed in 2001, GS takes the opposite approach. It stops
#   trying to find "significant" markers and instead assumes ALL markers might be associated
#   with the target trait, even if their effects are minuscule. It uses a statistical model
#   to simultaneously estimate the effects of thousands of SNP markers. Then, by taking a
#   weighted sum of all marker effects for a line, it calculates its "Genomic Estimated
#   Breeding Value" (GEBV). This GEBV is our prediction of the line's genetic merit.
#
# This shift in thinking solves the main problem of MAS, enabling selection for complex
# quantitative traits, significantly shortening breeding cycles, and improving selection accuracy.
#
# ### 2. R Environment and Required Packages
# This section is like the "reagent preparation" step before an experiment. It loads all the
# R packages needed for our analysis pipeline. Packages are toolboxes written by others
# that we can use directly.
#
# --- 检查并安装所需包 ---
required_packages <- c(
  "learnr", "devtools", "usethis", "roxygen2", "tidyverse", "gradethis",
  "downloadthis", "BGLR", "Matrix", "glmnet", "ggplot2", "AGHmatrix", "ggpubr",
  "tensorflow", "keras", "tfdatasets"
)
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Please install missing packages: ", paste(missing_packages, collapse = ", "),
       ". Use install.packages() or remotes::install_github() as needed.")
}

# Key Packages:
# - glmnet: Developed by one of the inventors of LASSO, it's extremely fast for "regularized"
#   regression. We'll use it for Ridge and LASSO.
# - BGLR (Bayesian Generalized Linear Regression): Developed by Dr. Pérez-Enciso himself,
#   it's the de facto standard for Bayesian GS methods.
# - AGHmatrix: Very useful in animal and plant breeding for easily calculating relationship
#   matrices (A matrix) from pedigrees.
# - tidyverse, ggplot2: Data science powerhouses for efficient data wrangling and creating
#   beautiful plots.
# Breeding Insight: An artisan is only as good as their tools. Familiarizing yourself with
# these core packages is your first step into computational biology.
lapply(required_packages, library, character.only = TRUE)

# the following packages are used for tensorflow gpu, please install before run here
library(reticulate)
library(tensorflow)
# The following line ensures R uses the specific conda environment where you installed TensorFlow.
# this can be achieved by installing conda, and use conda to install tensorflow-gpu, cudatoolkit
use_condaenv("tf210", required = TRUE)

# --- 定义全局参数和绘图主题 ---
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, error = FALSE, message = FALSE)

lambda <- 0.1             # Regularization parameter for glmnet
heritability <- 0.4       # Heritability for GBLUP and ssGBLUP
num_families <- 10        # Number of families for pedigree simulation
epochs_mlp <- 12          # Training epochs for the simple MLP
epochs_cnn <- 20          # Training epochs for the CNN
validation_split <- 0.1   # Validation set split for deep learning

# Define a unified, publication-quality ggplot2 theme for all plots
theme_publication <- function(base_size = 12, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10, color = "black"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      plot.margin = unit(c(1, 1, 1, 1), "cm") # Increase plot margins
    )
}

# =====================================================================================
# Part 2: Data Loading & Inspection
# =====================================================================================

# --- 加载并准备小麦数据 ---
data(wheat, package = "BGLR")

# Here we load the data. This classic dataset from CIMMYT is included in the BGLR package.

# genotypeMatrix: A matrix where rows are wheat lines (individuals) and columns are SNP markers.
# The values (0/1) represent the genotype at each SNP. This is our genotype data, X.
genotypeMatrix <- wheat.X

# phenotypeMatrix: A matrix where rows are the same wheat lines and columns are different
# traits (phenotypes) like yield, disease resistance, etc. This is our phenotype data, Y.
phenotypeMatrix <- wheat.Y

numIndividuals <- nrow(genotypeMatrix)
numMarkers <- ncol(genotypeMatrix)

# We will select the second trait from the phenotype matrix for our analysis.
phenotypeVector <- phenotypeMatrix[, 2]

# Breeding Insight: In GS, we often face a "p >> n" problem, where the number of markers (p)
# is far greater than the number of individuals (n). In this wheat data, p could be thousands
# while n is only a few hundred. Standard linear regression cannot be solved in this case
# (far more unknowns than equations), which is why we need the special statistical methods that follow.


# --- 数据分区 ---
# This is one of the most core, fundamental concepts in machine learning and genomic prediction:
# splitting the dataset into a Training Set and a Test Set.
#
# - Training Set: Contains both genotypes and phenotypes. We use this data to "train" our model,
#   i.e., to let the model learn the relationship between genotypes and phenotypes and estimate
#   the effect of each SNP. It's like giving a student a practice workbook with answers.
# - Test Set: Also contains genotypes and phenotypes. However, during the prediction phase, we
#   pretend we don't know its phenotypes. We feed only the genotypes to the trained model and let it
#   predict the phenotypic values. Then, we compare the model's predictions with the true phenotypes
#   (e.g., by calculating a correlation) to evaluate how good the model is. It's like a final exam
#   to see if the student (model) has truly learned.
#
# Why is this necessary? To prevent Overfitting. If we use the entire dataset to train and then
# evaluate the model on the same data, the model will "memorize" the answers and yield a deceptively
# high accuracy. Only by testing it on new, unseen data (the test set) can we know its true
# predictive power.
set.seed(123)
testSetIndices <- sort(sample(1:numIndividuals, size = numIndividuals * 0.2, replace = FALSE))

# The training set is created by excluding the test individuals.
genotypeTrain <- genotypeMatrix[-testSetIndices, ]
phenotypeTrain <- phenotypeVector[-testSetIndices]
# The test set is created using the selected individuals.
genotypeTest <- genotypeMatrix[testSetIndices, ]
phenotypeTest <- phenotypeVector[testSetIndices]

# For some models (like GBLUP), we use all individuals but mask the test phenotypes with NA.
phenotypeWithNAs <- phenotypeVector
phenotypeWithNAs[testSetIndices] <- NA

# --- 函数定义 ---
# This function implements VanRaden's (2008) method 1 for calculating the Genomic Relationship Matrix (G).
# It's currently the most common method.
vanraden <- function(genotypeMatrix) {
  # Calculate allele frequencies (p). The mean of 0/1 coded markers is 2p.
  alleleFrequencies <- apply(genotypeMatrix, 2, mean) * 0.5
  varianceSum <- 2 * sum(alleleFrequencies * (1 - alleleFrequencies))
  if (varianceSum == 0) {
    warning("All markers are monomorphic. Returning identity matrix.")
    return(diag(nrow(genotypeMatrix)))
  }
  # This is the core calculation: Z * Z'. `scale` centers the genotype matrix (M - P),
  # and `tcrossprod` computes this centered matrix multiplied by its transpose.
  genomicRelationshipMatrix <- tcrossprod(scale(genotypeMatrix, scale = FALSE))
  # The result is scaled by the sum of SNP variances.
  genomicRelationshipMatrix <- genomicRelationshipMatrix / varianceSum
  # A small inflation of the diagonal can improve numerical stability.
  diag(genomicRelationshipMatrix) <- diag(genomicRelationshipMatrix) * 1.05
  return(genomicRelationshipMatrix)
}

# This function elegantly sets up Henderson's Mixed Model Equations (MME),
# which are the core equations for solving BLUP.
mme <- function(phenotypeVector, covarianceMatrix, heritability, invert = FALSE) {
  if (invert) {
    # If we need the inverse, check for positive definiteness and invert.
    eig <- eigen(covarianceMatrix, symmetric = TRUE, only.values = TRUE)
    if (any(eig$values <= 1e-8)) stop("Covariance matrix is not positive definite.")
    covarianceMatrix <- solve(covarianceMatrix)
  }
  numObs <- length(phenotypeVector)
  y_obs <- phenotypeVector
  y_obs[is.na(y_obs)] <- 0 # Temporarily replace NAs with 0 for matrix math
  
  # Z_fixed is an incidence matrix for the mean. It's 1 for observed, 0 for missing.
  Z_fixed <- rep(1, numObs)
  Z_fixed[is.na(phenotypeVector)] <- 0
  # Z_random is a diagonal matrix indicating which individuals have phenotypes.
  Z_random <- Diagonal(numObs, x = Z_fixed)
  
  # lambda is the variance ratio (sigma_e^2 / sigma_g^2)
  lambda_mme <- (1 - heritability) / heritability
  # This builds the part of the MME related to random effects.
  V_mme <- covarianceMatrix * lambda_mme + t(Z_random) %*% Z_random
  
  # Assemble the full Left-Hand Side (LHS) of the MME.
  LHS <- matrix(0, nrow = numObs + 1, ncol = numObs + 1)
  LHS[1, 1] <- sum(Z_fixed)
  LHS[1, -1] <- Z_fixed
  LHS[-1, 1] <- Z_fixed
  LHS[-1, -1] <- as.matrix(V_mme)
  
  # Assemble the Right-Hand Side (RHS) of the MME.
  RHS <- c(sum(y_obs), as.vector(t(Z_random) %*% y_obs))
  return(list(LHS = LHS, RHS = RHS))
}

# This function calculates the inverse of the H matrix for ssGBLUP.
# This is the magic of ssGBLUP: H-inverse can be constructed easily without ever
# forming the massive H matrix itself. H⁻¹ = A⁻¹ + [0, 0; 0, G⁻¹ - A₂₂⁻¹]
doH_1 <- function(pedigreeRelationshipMatrix, grmForGenotyped, genotypedIndicesInPedigree) {
  grmInverse <- solve(grmForGenotyped) # G⁻¹
  A22 <- pedigreeRelationshipMatrix[genotypedIndicesInPedigree, genotypedIndicesInPedigree]
  pedigreeRelationshipInverseForGenotyped <- solve(A22) # A₂₂⁻¹
  pedigreeRelationshipInverse <- solve(pedigreeRelationshipMatrix) # A⁻¹
  
  hMatrixInverse <- pedigreeRelationshipInverse
  # Add the genomic information part: G⁻¹ - A₂₂⁻¹
  hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] <-
    hMatrixInverse[genotypedIndicesInPedigree, genotypedIndicesInPedigree] + grmInverse - pedigreeRelationshipInverseForGenotyped
  return(hMatrixInverse)
}

# --- 数据检查：PCA (Population Structure) ---
# "Garbage in, garbage out." Before running any complex model, checking data quality is crucial.
# Principal Component Analysis (PCA) is the most common method for checking population structure.
# What is PCA? Think of it as taking a "group photo" of your breeding materials from a special
# angle that best showcases the genetic relationships among all individuals. Genetically close
# individuals will cluster together in the plot.
principalComponents <- eigen(tcrossprod(scale(genotypeMatrix)) / numMarkers)
varianceExplainedPC1 <- round(principalComponents$values[1] / sum(principalComponents$values), 3) * 100
varianceExplainedPC2 <- round(principalComponents$values[2] / sum(principalComponents$values), 3) * 100
ggplot(data.frame(PC1 = principalComponents$vectors[, 1], PC2 = principalComponents$vectors[, 2]),
       aes(x = PC1, y = PC2)) +
  geom_point(alpha = 0.7) +
  labs(title = "Population Structure (PCA)",
       x = paste0("PC1 (", varianceExplainedPC1, "%)"),
       y = paste0("PC2 (", varianceExplainedPC2, "%)")) +
  theme_publication()
# Breeding Insight: The PCA plot clearly shows two subgroups in the population. This is common
# in breeding, perhaps they originate from two different breeding programs. This information is
# vital because if you ignore population structure, the model might mistake the "subgroup effect"
# (e.g., individuals in group A are generally higher-yielding than in group B) for the effect of
# certain SNP markers, leading to spurious associations and inaccurate predictions.

# --- 数据检查：等位基因频率 (Allele Frequency) ---
# This code calculates and visualizes the Minor Allele Frequency (MAF) for each SNP marker.
# Why care about MAF? If a SNP's MAF is very low (e.g., <1%), it means almost all individuals
# have the same genotype at that locus. Such markers carry very little genetic information, are
# of little help for building a predictive model, and add computational noise. In practice,
# we often filter out markers with very low MAF.
alleleFrequencies <- apply(genotypeMatrix, 2, mean) * 0.5
minorAlleleFrequencies <- ifelse(alleleFrequencies > 0.5, 1 - alleleFrequencies, alleleFrequencies)
ggplot(data.frame(MAF = minorAlleleFrequencies), aes(x = MAF)) +
  geom_histogram(bins = 30, fill = "royalblue", color = "black", alpha = 0.8) +
  labs(title = "Minor Allele Frequency (MAF) Distribution", x = "Minor Allele Frequency", y = "Frequency") +
  theme_publication()
# Breeding Insight: This histogram shows the frequency distribution of markers. The fact that
# most markers have a relatively high frequency suggests the data likely comes from a SNP chip,
# rather than whole-genome resequencing (which would uncover many more rare variants).
# High-frequency markers are more informative for building models.

# --- 练习：表型分布 (Phenotype Distribution) ---
# This exercise checks if our target trait (phenotype) roughly follows a normal distribution.
# Most GS models for continuous traits (like GBLUP, Bayesian regression) assume that the
# phenotype (or rather, the model's residuals) is normally distributed.
shapiro_result <- shapiro.test(phenotypeVector)
ggplot(data.frame(Phenotype = phenotypeVector), aes(x = Phenotype)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black", alpha = 0.8) +
  # We overlay a theoretical normal curve based on the data's mean and standard deviation.
  stat_function(fun = dnorm, args = list(mean = mean(phenotypeVector), sd = sd(phenotypeVector)),
                color = "darkred", linewidth = 1.2) +
  labs(title = "Phenotype Distribution",
       subtitle = paste("Shapiro-Wilk test p-value =", round(shapiro_result$p.value, 4)),
       x = "Phenotype Value", y = "Density") +
  theme_publication()
# Breeding Insight: If your data is severely skewed, applying the model directly might lead to
# inaccurate results. You might need to transform the data (e.g., log-transform) or use a
# model better suited for non-normal data (e.g., a generalized linear model).

# --- 训练集与测试集的PCA ---
# This is a sanity check to ensure our random split didn't create training and test sets
# with drastically different population structures. Ideally, the red and black dots should be
# well-mixed.
genotypeTrain_noFixedMarkers <- genotypeTrain[, which(apply(genotypeTrain, 2, sd) != 0)]
pcTrain_obj <- prcomp(genotypeTrain_noFixedMarkers, scale. = TRUE, center = TRUE)
pcTest <- predict(pcTrain_obj, newdata = genotypeTest[, intersect(colnames(genotypeTrain_noFixedMarkers), colnames(genotypeTest))])
plot_data <- data.frame(
  PC1 = c(pcTrain_obj$x[, 1], pcTest[, 1]),
  PC2 = c(pcTrain_obj$x[, 2], pcTest[, 2]),
  Set = rep(c("Train", "Test"), times = c(nrow(genotypeTrain), nrow(genotypeTest)))
)
ggplot(plot_data, aes(x = PC1, y = PC2, color = Set)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = c("Train" = "black", "Test" = "red")) +
  labs(title = "PCA of Train and Test Partitions", x = "Principal Component 1", y = "Principal Component 2") +
  theme_publication()

# Define a standardized scatter plot function for plotting all model predictions.
# This helps us visually compare the performance of different models.
plot_predictions <- function(test_phenotypes, predicted_values, model_name) {
  df <- data.frame(Observed = test_phenotypes, Predicted = predicted_values)
  # The correlation (r) between observed and predicted values is our primary metric for accuracy.
  ggplot(df, aes(x = Observed, y = Predicted)) +
    geom_point(alpha = 0.6, color = "blue") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
    stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top",
             aes(label = ..r.label..), size = 4.5) +
    labs(title = model_name, x = "Observed Phenotype", y = "Predicted Phenotype") +
    theme_publication()
}

# =====================================================================================
# Part 3: Classic Statistical Models (L1 & L2 Regularization)
# =====================================================================================
# To handle the p >> n problem, statisticians introduced "regularization," which you can
# think of as adding a "penalty term" or "constraint" to the model to prevent overfitting.

# --- 岭回归 (Ridge Regression, L2 Penalty) ---
# Ridge regression adds an L2 penalty term to the loss function, which is the sum of the
# squared values of all SNP effects, multiplied by a coefficient lambda.
# Loss = Σ(y - ŷ)² + λΣ(β²)
# The penalty forces the model to shrink all SNP effects towards zero. The larger the lambda,
# the stronger the shrinkage.
# Core Idea: Ridge assumes that no single SNP is particularly outstanding. It prefers to assign
# a small effect to every SNP, which aligns well with the genetic architecture of complex traits
# controlled by many genes of small effect.
fitRidge <- glmnet(y = phenotypeTrain, x = genotypeTrain, lambda = lambda, alpha = 0)
# `alpha = 0` is the key parameter that specifies Ridge Regression in glmnet.
predictionsRidge <- predict(fitRidge, newx = genotypeTest)[, 1]
plot_predictions(phenotypeTest, predictionsRidge, "Ridge Regression")

# --- LASSO (L1 Penalty) ---
# LASSO (Least Absolute Shrinkage and Selection Operator) is similar, but its penalty is
# the sum of the absolute values of the SNP effects.
# Loss = Σ(y - ŷ)² + λΣ|β|
# This seemingly small change has a profound consequence: LASSO can shrink many SNP effects
# to exactly zero!
# Core Idea: LASSO performs variable selection. It assumes that only a few SNPs are truly
# important (non-zero effect), while most are noise (zero effect).
fitLasso <- glmnet(y = phenotypeTrain, x = genotypeTrain, lambda = lambda, alpha = 1)
# `alpha = 1` specifies LASSO regression.
predictionsLasso <- predict(fitLasso, newx = genotypeTest)[, 1]
plot_predictions(phenotypeTest, predictionsLasso, "LASSO")

# --- 弹性网络 (Elastic Net) ---
# Elastic Net is a hybrid of Ridge and LASSO, using both L1 and L2 penalties.
# Core Idea: It tries to combine the advantages of both. In breeding, when many SNPs are
# in high linkage disequilibrium (their effects are correlated), LASSO might randomly pick
# just one. Elastic Net tends to select these correlated SNPs together as a group, which
# is often more biologically plausible.
fitElasticNet <- glmnet(y = phenotypeTrain, x = genotypeTrain, lambda = lambda, alpha = 0.5)
# An `alpha` value between 0 and 1 engages Elastic Net. alpha closer to 1 is more like LASSO;
# closer to 0 is more like Ridge.
predictionsElasticNet <- predict(fitElasticNet, newx = genotypeTest)[, 1]
plot_predictions(phenotypeTest, predictionsElasticNet, "Elastic Net (α = 0.5)")

# --- 练习：弹性网络的Alpha网格搜索 ---
# In practice, you need to find the optimal values for hyperparameters like lambda and alpha.
# Cross-validation is the standard method. Here, we do a simple grid search for alpha.
alpha_grid <- seq(0, 1, by = 0.1)
correlations <- sapply(alpha_grid, function(a) {
  fit <- glmnet(y = phenotypeTrain, x = genotypeTrain, lambda = lambda, alpha = a)
  predictions <- predict(fit, newx = genotypeTest)[, 1]
  cor(predictions, phenotypeTest)
})
ggplot(data.frame(Alpha = alpha_grid, Correlation = correlations), aes(x = Alpha, y = Correlation)) +
  geom_line(linewidth = 1.2, color = "royalblue") +
  geom_point(size = 3, color = "darkred") +
  labs(title = "Elastic Net Performance vs. Alpha",
       x = expression(bold(Alpha) ~ "(0=Ridge, 1=LASSO)"),
       y = "Predictive Correlation (r)") +
  scale_x_continuous(breaks = alpha_grid) +
  theme_publication()

# =====================================================================================
# Part 4: The Breeder's Classic - GBLUP
# =====================================================================================
# GBLUP (Genomic Best Linear Unbiased Prediction) is the most classic and widely used GS
# method in animal and plant breeding. Its approach differs from Ridge/LASSO.
# - Ridge/LASSO approach: Estimate the effect of each SNP (a marker-effect model).
# - GBLUP approach: Don't estimate SNP effects directly. Instead, use all SNP information to
#   calculate a genomic relationship matrix (G) among individuals. Then, substitute this G
#   matrix for the traditional pedigree-based A matrix in the classic BLUP equations to directly
#   estimate the breeding value of each individual (an animal model).
# It has been mathematically proven that GBLUP is equivalent to Ridge Regression. They are two
# different paths to the same solution. GBLUP's advantage is its seamless integration into the
# Mixed Linear Model framework familiar to traditional breeders.

# --- GBLUP ---
# Calculate the Genomic Relationship Matrix (G) for all individuals.
genomicRelationshipMatrix <- vanraden(genotypeMatrix)

# In the GBLUP framework, we put all individuals (train + test) into one model. We set the
# phenotypes of the test set to NA. The model will then "predict" these missing values,
# which become our predicted breeding values.
# The `mme` function sets up the mixed model equations. We pass the inverse of G.
mixedModelEquations <- mme(phenotypeWithNAs, genomicRelationshipMatrix, heritability, invert = TRUE)
# `solve(A, b)` in R solves the linear system Ax = b. Here, we solve the MME.
# The result contains the intercept (which we drop with [-1]) and the GEBVs for all individuals.
breedingValueEstimates <- solve(mixedModelEquations$LHS, mixedModelEquations$RHS)[-1]
# We extract the predicted GEBVs for the individuals in the test set.
predictionsGBLUP <- breedingValueEstimates[testSetIndices]
plot_predictions(phenotypeTest, predictionsGBLUP, "GBLUP")

# =====================================================================================
# Part 5: The Bayesian Family
# =====================================================================================
# Bayesian methods are another major school of thought in GS. Their philosophy differs from
# the frequentist approach (like Ridge, LASSO).
# - Frequentist view: A SNP effect is a fixed, unknown true value that we try to "estimate".
# - Bayesian view: A SNP effect is not a fixed value but a random variable that follows a
#   certain probability distribution. We use data to update our "prior" belief about this
#   effect into a "posterior" belief.
# The core difference between various Bayesian GS methods lies in their "prior assumption"
# about the distribution of SNP effects. The BGLR package makes these models easy to implement.

# --- 贝叶斯模型 ---
# Bayesian Ridge Regression (BRR)
# Prior Assumption: All SNP effects are drawn from a single normal distribution, N(0, σ²_β).
# This is essentially the Bayesian equivalent of Ridge Regression and GBLUP.
ETA_BRR <- list(list(X = genotypeMatrix, model = "BRR"))
# We run the MCMC. nIter is the total number of iterations, and burnIn is the initial
# number of iterations to discard while the chain converges.
fitBayesianRidge <- BGLR(y = phenotypeWithNAs, ETA = ETA_BRR, nIter = 5000, burnIn = 2000, verbose = FALSE)
# The `yHat` component of the BGLR output contains the posterior mean of the predicted values.
plot_predictions(phenotypeTest, fitBayesianRidge$yHat[testSetIndices], "Bayesian Ridge Regression")

# BayesA
# Prior Assumption: Each SNP effect comes from its own, separate normal distribution, N(0, σ²_βi).
# This allows each SNP to have its own specific variance, making it more flexible than BRR.
ETA_BayesA <- list(list(X = genotypeMatrix, model = "BayesA"))
fitBayesA <- BGLR(y = phenotypeWithNAs, ETA = ETA_BayesA, nIter = 5000, burnIn = 2000, verbose = FALSE)
plot_predictions(phenotypeTest, fitBayesA$yHat[testSetIndices], "BayesA")

# BayesB
# Prior Assumption: Uses a "spike-and-slab" prior. It assumes a proportion (π) of SNPs have a non-zero
# effect drawn from a distribution (the slab), while the rest (1-π) have an effect of exactly zero (the spike).
# This is conceptually similar to LASSO as it performs variable selection. In BayesB, the non-zero
# effects each have their own variance (like BayesA).
ETA_BayesB <- list(list(X = genotypeMatrix, model = "BayesB"))
fitBayesB <- BGLR(y = phenotypeWithNAs, ETA = ETA_BayesB, nIter = 5000, burnIn = 2000, verbose = FALSE)
plot_predictions(phenotypeTest, fitBayesB$yHat[testSetIndices], "BayesB")

# BayesC
# Prior Assumption: Also uses a spike-and-slab prior. However, all non-zero SNP effects share
# a common variance (like Bayesian Ridge Regression).
ETA_BayesC <- list(list(X = genotypeMatrix, model = "BayesC"))
fitBayesC <- BGLR(y = phenotypeWithNAs, ETA = ETA_BayesC, nIter = 5000, burnIn = 2000, verbose = FALSE)
plot_predictions(phenotypeTest, fitBayesC$yHat[testSetIndices], "BayesC")

# Bayesian GBLUP (implemented via RKHS in BGLR)
# This is another way to perform GBLUP within the Bayesian framework. Instead of providing the
# marker matrix `X`, we provide the pre-computed genomic relationship matrix `K`.
ETA_BGBLUP <- list(list(K = genomicRelationshipMatrix, model = "RKHS"))
fitBayesianGBLUP <- BGLR(y = phenotypeWithNAs, ETA = ETA_BGBLUP, nIter = 5000, burnIn = 2000, verbose = FALSE)
plot_predictions(phenotypeTest, fitBayesianGBLUP$yHat[testSetIndices], "Bayesian GBLUP (RKHS)")

# =====================================================================================
# Part 6: The Ultimate Weapon - Single-Step GBLUP (ssGBLUP)
# =====================================================================================
# ssGBLUP is one of the most important recent advancements in GS.
# What pain point does it solve? In real-world breeding programs, we often have genotype
# data for a small number of key parents or elite lines (as genotyping is expensive) but
# pedigree data for thousands of individuals.
# - GBLUP can only use genotyped individuals.
# - Traditional BLUP can only use pedigreed individuals.
# - ssGBLUP is amazing because it integrates genotype, phenotype, and pedigree data into a
#   single analysis, leveraging ALL available information!
# Its core is the construction of a hybrid relationship matrix, H, which seamlessly blends
# the pedigree relationship matrix (A) and the genomic relationship matrix (G).

# --- 单步GBLUP (Single-Step GBLUP) ---
# Since this dataset doesn't have a pedigree, we simulate one. In a real application,
# you would use your actual pedigree data.
# We use clustering to create fake "families" and assign them virtual parents.
hierarchicalClustering <- hclust(dist(genotypeMatrix), method = "ward.D2")
familyAssignments <- cutree(hierarchicalClustering, num_families)
numFounders <- 2 * num_families
totalPedigreeSize <- numIndividuals + numFounders
pedigreeMatrix <- matrix(0, nrow = totalPedigreeSize, ncol = 3)
pedigreeMatrix[, 1] <- seq(totalPedigreeSize)
pedigreeMatrix[(numFounders + 1):totalPedigreeSize, 2] <- familyAssignments * 2 - 1
pedigreeMatrix[(numFounders + 1):totalPedigreeSize, 3] <- familyAssignments * 2
# Now, we use AGHmatrix to compute the pedigree-based relationship matrix (A).
pedigreeRelationshipMatrix <- Amatrix(pedigreeMatrix)
genotypedIndicesInPedigree <- (numFounders + 1):totalPedigreeSize
genotypeMatrixForSSGBLUP <- genotypeMatrix[, which(apply(genotypeMatrix, 2, sd) != 0)]
grmForGenotyped <- vanraden(genotypeMatrixForSSGBLUP)
# This is the essence of ssGBLUP. We compute the inverse of the H matrix using its
# computationally efficient formula.
hMatrixInverse <- doH_1(pedigreeRelationshipMatrix, grmForGenotyped, genotypedIndicesInPedigree)
# Prepare the phenotype vector, including NAs for founders and test set individuals.
testSetIndicesInPedigree <- testSetIndices + numFounders
phenotypeVectorWithFounders <- c(rep(NA, numFounders), phenotypeVector)
phenotypeWithNAsAndFounders <- phenotypeVectorWithFounders
phenotypeWithNAsAndFounders[testSetIndicesInPedigree] <- NA
# Once we have H-inverse, the rest is just like GBLUP. We call the `mme` function,
# but this time we provide H-inverse directly and set `invert=FALSE`.
mixedModelEquationsSS <- mme(phenotypeWithNAsAndFounders, hMatrixInverse, heritability, invert = FALSE)
breedingValueEstimatesSS <- solve(mixedModelEquationsSS$LHS, mixedModelEquationsSS$RHS)[-1]
# The result is GEBVs for ALL individuals, including those without genotypes. This is the power of ssGBLUP.
predictionsSSGBLUP <- breedingValueEstimatesSS[testSetIndicesInPedigree]
plot_predictions(phenotypeVectorWithFounders[testSetIndicesInPedigree], predictionsSSGBLUP, "Single-Step GBLUP (ssGBLUP)")

# =====================================================================================
# Part 7: The Frontier - Deep Learning
# =====================================================================================
# Deep Learning, a branch of machine learning, uses "neural networks." You can think of a
# neural network as a very complex, non-linear function with immense fitting capability.
# Its advantage is the potential to capture complex non-linear effects (like epistasis) that
# linear models cannot. However, they require large datasets, are prone to overfitting, and
# are often "black boxes," making their internal logic hard to interpret.

# --- 深度学习部分 ---
tryCatch({
  # --- 为深度学习准备标准化数据 ---
  # Neural networks generally perform better with standardized (mean=0, sd=1) input data.
  train_mean <- colMeans(genotypeTrain)
  train_sd <- apply(genotypeTrain, 2, sd)
  train_sd[train_sd == 0] <- 1 # Avoid division by zero for monomorphic markers
  genotypeTrain_scaled <- scale(genotypeTrain, center = train_mean, scale = train_sd)
  genotypeTest_scaled <- scale(genotypeTest, center = train_mean, scale = train_sd)
  
  # --- 深度学习: 简单多层感知机 (MLP) ---
  # MLP is the most basic neural network, consisting of multiple "fully-connected" layers.
  # We define the model architecture layer by layer, like stacking blocks.
  mlp_model_simple <- keras_model_sequential(name = "Simple_MLP") %>%
    # Input layer and first hidden layer with 128 neurons and ReLU activation function.
    layer_dense(units = 128, activation = "relu", input_shape = numMarkers) %>%
    # Second hidden layer with 64 neurons.
    layer_dense(units = 64, activation = "relu") %>%
    # Output layer with a single neuron for predicting the continuous phenotype value.
    layer_dense(units = 1, activation = "linear")
  # Compile the model: specify the loss function (how to measure error), the optimizer
  # (how to adjust weights), and metrics to track.
  mlp_model_simple %>% compile(loss = "mse", optimizer = optimizer_adam(), metrics = c("mae"))
  # Train the model: feed it the training data. `epochs` is the number of training cycles.
  mlp_model_simple %>% fit(genotypeTrain_scaled, phenotypeTrain, epochs = epochs_mlp, validation_split = validation_split, verbose = 0)
  predictions_dl_simple <- mlp_model_simple %>% predict(genotypeTest_scaled)
  plot_predictions(phenotypeTest, predictions_dl_simple[, 1], "Simple MLP")
  
  # --- 深度学习: 使用tfdatasets的MLP ---
  # This is a more advanced way to build an MLP using the tfdatasets pipeline, which is
  # more efficient for very large datasets.
  train_df <- as_tibble(genotypeTrain)
  test_df <- as_tibble(genotypeTest)
  train_df$y <- phenotypeTrain
  test_df$y <- phenotypeTest
  feature_spec <- feature_spec(train_df, y ~ .) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>%
    fit()
  input <- layer_input_from_dataset(train_df %>% select(-y))
  output <- input %>%
    layer_dense_features(dense_features(feature_spec)) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 32, activation = "softplus") %>%
    layer_dense(units = 1, activation = "linear")
  mlp_model <- keras_model(input, output, name = "Sophisticated_MLP")
  mlp_model %>% compile(loss = "mse", optimizer = optimizer_rmsprop(), metrics = list("mae"))
  mlp_model %>% fit(x = train_df %>% select(-y), y = train_df$y, epochs = epochs_cnn, validation_split = 0.2, verbose = 0)
  test_predictions_mlp <- mlp_model %>% predict(test_df %>% select(-y))
  plot_predictions(test_df$y, test_predictions_mlp[, 1], "Sophisticated MLP")
  
  # --- 深度学习: 卷积神经网络 (CNN) ---
  # CNNs excelled first in image recognition. Their core is the "convolutional layer," which
  # uses a sliding window (kernel) to scan the input data and capture local patterns.
  # Application to Genomics: We can treat the sequence of SNPs on a chromosome as a 1D signal.
  # A CNN can scan this sequence to learn combined effects of adjacent SNPs (epistasis),
  # something that is difficult for linear models to capture.
  # CNNs require a specific input shape (samples, steps, channels).
  xtrain_cnn <- array(genotypeTrain_scaled, dim = c(nrow(genotypeTrain_scaled), ncol(genotypeTrain_scaled), 1))
  xtest_cnn <- array(genotypeTest_scaled, dim = c(nrow(genotypeTest_scaled), ncol(genotypeTest_scaled), 1))
  y_train_matrix <- as.matrix(phenotypeTrain)
  y_test_matrix <- as.matrix(phenotypeTest)
  cnn_model <- keras_model_sequential(name = "CNN") %>%
    # The core 1D convolutional layer.
    layer_conv_1d(filters = 32, kernel_size = 3, strides = 3, input_shape = c(dim(xtrain_cnn)[2], 1), activation = "relu") %>%
    # Pooling layer to downsample and extract key features.
    layer_max_pooling_1d(pool_size = 2) %>%
    # Flatten the multi-dimensional features into a 1D vector before feeding to dense layers.
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  cnn_model %>% compile(loss = "mse", optimizer = "sgd")
  cnn_model %>% fit(x = xtrain_cnn, y = y_train_matrix, epochs = epochs_cnn, validation_split = 0.2, verbose = 0)
  test_predictions_cnn <- cnn_model %>% predict(xtest_cnn)
  plot_predictions(y_test_matrix[, 1], test_predictions_cnn[, 1], "Convolutional Neural Network (CNN)")
  
}, error = function(e) {
  stop("Error in TensorFlow/Keras section: ", e$message,
       ". Please ensure TensorFlow and Keras are installed and configured correctly.")
})

# =====================================================================================
# Conclusion
# =====================================================================================
#
# We have completed this exciting journey through genomic prediction with R. We started from
# the basic concepts and learned:
# - Why GS is needed: For effective selection of complex traits controlled by many small-effect genes.
# - The importance of data preparation: Checking population structure (PCA) and marker quality (MAF) is essential.
# - The golden rule of model evaluation: Use train/test splits (cross-validation) for an honest assessment.
# - The two major statistical schools:
#   - Regularized Regression (glmnet): Solves p>>n with penalties (Ridge L2, LASSO L1).
#   - Mixed Linear Models (GBLUP): The familiar breeder's framework using a genomic relationship matrix.
# - The Bayesian family (BGLR): Offers flexibility through different prior assumptions on SNP effects.
# - The integrator (ssGBLUP): The powerful, practical method for combining genomic, pedigree, and phenotypic data.
# - The future direction (Deep Learning): Using neural networks to potentially capture complex non-additive effects.
#
# This tutorial is not just about learning to type code, but about learning to think like a
# computational breeder: starting from a biological question, choosing appropriate statistical
# models, and rigorously analyzing and interpreting the results.

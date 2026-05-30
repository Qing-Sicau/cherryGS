# =====================================================================================
#   DL model investigation by re-evaluating MLP/CNN after GWAS-based SNP feature screening.
#
# Analyses included:
#   1) Full-population repeated k-fold CV using All SNPs and DL_feature_sets top-k SNPs
#   2) Cross-population prediction: NH/HN -> HP and HP -> NH/HN
#   3) Training-size saturation analysis using increasing proportions of the training set
#
# Main outputs:
#   DL_feature_analysis/results/DL_full_population_feature_results.csv
#   DL_feature_analysis/results/DL_cross_population_feature_results.csv
#   DL_feature_analysis/results/DL_saturation_feature_results.csv
#   DL_feature_analysis/results/DL_tuning_summary.csv
#   DL_feature_analysis/results/DL_training_history.csv
#   DL_feature_analysis/plots/*.pdf
#
# Notes:
#   - GPU mode runs Keras sequentially to avoid GPU memory contention.
#   - CPU-only mode can parallelize trait x SNP-set tasks with PSOCK workers.
#   - On Ubuntu/Linux GPU servers, run this inside the configured conda environment.
#   - Analysis configuration is hard-coded below; edit TRAITS_TO_RUN_FIXED to select traits.
#
# =====================================================================================

# ----------------------------- Fixed/selectable run configuration --------------------
# This script is designed for a Ubuntu GPU server with invidia A800, for example:
#   conda activate poly_map
#   Rscript DL_feature_analysis_ubuntu_A800_trait_select_cleaned.R
# Edit TRAITS_TO_RUN_FIXED below to choose which traits to run.

GENOTYPE_FILE  <- "genotype.dosages.tsv"
PHENOTYPE_FILE <- "phenotype_BLUPs.csv"
FEATURE_DIR    <- "DL_feature_sets"
OUT_DIR        <- "DL_feature_analysis"
CONDA_ENV      <- ifelse(.Platform$OS.type == "windows", "dl_tf", "poly_map")

# Full analysis configuration.
TRAITS_TO_RUN_FIXED <- c(
  "BLUP_booming_day",
  "BLUP_fruit_maturation_period",
  "BLUP_fruit_weight",
  "BLUP_longitudinal_diameter",
  "BLUP_transverse_diameter",
  "BLUP_lateral_diameter",
  "BLUP_fruit_shape_index",
  "BLUP_fruit_stalk_length",
  "BLUP_soluble_solid_content",
  "BLUP_titratable_acid",
  "BLUP_ratio_of_TSS_TA"
)

REQUESTED_SNP_SETS <- c("All", "Top500", "Top1000", "Top3000", "Top6000")
MODELS_TO_RUN <- c("Ridge", "MLP", "CNN")

# Parallel/GPU strategy, matching the cross-pop script style:
#   "auto"   : GPU visible -> serial GPU execution; no GPU -> CPU parallel execution.
#   "serial" : always run one task at a time. If a GPU is visible, TensorFlow may use it.
#   "cpu"    : force CPU-style parallel execution and hide CUDA from TensorFlow.
PARALLEL_MODE <- "auto"
REQUESTED_N_WORKERS <- 12L

# When CPU parallelism is used, hide CUDA from TensorFlow in the master process and workers.
# This avoids repeated cuInit warnings and prevents accidental GPU memory use.
HIDE_CUDA_FOR_CPU_PARALLEL <- TRUE

# Optional TensorFlow GPU detection fallback. Keep FALSE by default because it can initialize
# CUDA and print cuInit warnings on CPU-only nodes with CUDA-enabled TensorFlow builds.
ENABLE_TF_GPU_DETECT <- FALSE

# Low-level math/TensorFlow thread settings inside each R process/worker.
BLAS_THREADS_PER_WORKER <- 1L
TF_INTRAOP_THREADS_PER_WORKER <- 1L
TF_INTEROP_THREADS_PER_WORKER <- 1L
TF_FORCE_GPU_ALLOW_GROWTH_VALUE <- "true"
TF_CPP_MIN_LOG_LEVEL_VALUE <- "2"

NUM_FOLDS         <- 5L
NUM_REPEATS_FULL  <- 20L
NUM_REPEATS_CROSS <- 20L
NUM_REPEATS_SAT   <- 20L
SAT_FRACTIONS     <- c(0.2, 0.4, 0.6, 0.8, 1.0)

TRAINING_EPOCHS <- 100L
BATCH_SIZE      <- 32L  # A800 80G + 128G RAM default; reduce to 16 or 8 only if needed
INNER_VALIDATION_FRACTION <- 0.2

DO_TUNING <- TRUE

RUN_FULL_POPULATION  <- TRUE
RUN_CROSS_POPULATION <- TRUE
RUN_SATURATION       <- TRUE

# Safety switch. In auto mode, FALSE lets the script fall back to CPU parallel execution
# when no GPU is visible. Set TRUE only when you explicitly require GPU execution.
REQUIRE_GPU <- FALSE

# 0 means use TensorFlow memory growth. Set e.g. 40000 to cap TensorFlow at ~40 GB.
GPU_MEMORY_LIMIT_MB <- 0L

# Optional: edit these only if you want to move the conda environment location.
# On Ubuntu, the default expected path is ~/miniconda3/envs/poly_map.
# On Windows, the script will try common miniconda locations for dl_tf.
USER_CONDA_PREFIX <- NA_character_

# ----------------------------- Early WSL2 GPU bootstrap ------------------------------
#   For pip-installed tensorflow[and-cuda] on WSL2, CUDA/cuDNN shared libraries live in
#   <env>/lib/pythonX.Y/site-packages/nvidia/*/lib. TensorFlow needs those paths in
#   LD_LIBRARY_PATH from process startup. Setting LD_LIBRARY_PATH later inside an already
#   running R process can be too late for dlopen(), so this script relaunches itself once
#   with the correct LD_LIBRARY_PATH. You do not need to export anything manually.

is_wsl <- function() {
  if (.Platform$OS.type != "unix") return(FALSE)
  txt <- tryCatch(readLines("/proc/version", warn = FALSE), error = function(e) "")
  any(grepl("microsoft|WSL", txt, ignore.case = TRUE))
}

find_conda_env_prefix_early <- function(env_name) {
  if (!is.na(USER_CONDA_PREFIX) && nzchar(USER_CONDA_PREFIX)) {
    return(normalizePath(path.expand(USER_CONDA_PREFIX), winslash = "/", mustWork = FALSE))
  }

  if (.Platform$OS.type == "windows") {
    candidates <- c(
      file.path(Sys.getenv("USERPROFILE"), ".conda", "envs", env_name),
      file.path(Sys.getenv("USERPROFILE"), "miniconda3", "envs", env_name),
      file.path(Sys.getenv("USERPROFILE"), "anaconda3", "envs", env_name),
      file.path("C:/ProgramData/miniconda3/envs", env_name),
      file.path("C:/ProgramData/anaconda3/envs", env_name)
    )
  } else {
    candidates <- c(
      file.path(Sys.getenv("HOME"), "miniconda3", "envs", env_name),
      file.path(Sys.getenv("HOME"), "anaconda3", "envs", env_name),
      file.path(Sys.getenv("HOME"), "mambaforge", "envs", env_name),
      file.path(Sys.getenv("HOME"), "micromamba", "envs", env_name)
    )
  }

  candidates <- candidates[dir.exists(candidates)]
  if (length(candidates) > 0) return(normalizePath(candidates[1], winslash = "/", mustWork = FALSE))
  NA_character_
}

python_executable_for_env_early <- function(env_base) {
  if (is.na(env_base) || !nzchar(env_base)) return(NA_character_)
  if (.Platform$OS.type == "windows") {
    candidates <- c(file.path(env_base, "python.exe"))
  } else {
    candidates <- c(file.path(env_base, "bin", "python"), file.path(env_base, "bin", "python3"))
  }
  candidates <- normalizePath(candidates, winslash = "/", mustWork = FALSE)
  hit <- candidates[file.exists(candidates) & file.access(candidates, mode = 1) == 0]
  if (length(hit) > 0) hit[1] else NA_character_
}

get_site_packages_early <- function(python_bin) {
  if (is.na(python_bin) || !nzchar(python_bin)) return(NA_character_)
  code <- "import site; print(site.getsitepackages()[0])"
  out <- tryCatch(system2(python_bin, args = c("-c", shQuote(code)), stdout = TRUE, stderr = TRUE),
                  error = function(e) character(0))
  out <- out[grepl("site-packages$", out)]
  if (length(out) == 0) NA_character_ else normalizePath(out[1], winslash = "/", mustWork = FALSE)
}

bootstrap_reexec_for_wsl_gpu <- function() {
  if (.Platform$OS.type == "windows") return(invisible(FALSE))
  if (!is_wsl()) return(invisible(FALSE))
  if (identical(Sys.getenv("DL_RSCRIPT_REEXEC_WITH_GPU_LIBS", unset = ""), "1")) return(invisible(FALSE))

  env_base <- find_conda_env_prefix_early(CONDA_ENV)
  python_bin <- python_executable_for_env_early(env_base)
  site_pkgs <- get_site_packages_early(python_bin)

  if (is.na(env_base) || is.na(python_bin) || is.na(site_pkgs)) {
    return(invisible(FALSE))
  }

  nvidia_lib_dirs <- Sys.glob(file.path(site_pkgs, "nvidia", "*", "lib"))
  ld_paths <- unique(normalizePath(c(
    "/usr/lib/wsl/lib",
    file.path(env_base, "lib"),
    file.path(site_pkgs, "tensorflow"),
    nvidia_lib_dirs,
    Sys.getenv("LD_LIBRARY_PATH", unset = "")
  ), winslash = "/", mustWork = FALSE))
  ld_paths <- ld_paths[nzchar(ld_paths)]
  new_ld <- paste(ld_paths, collapse = ":")

  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- cmd_args[grepl("^--file=", cmd_args)]
  if (length(file_arg) == 0) return(invisible(FALSE))
  script_file <- sub("^--file=", "", file_arg[1])
  trailing <- commandArgs(trailingOnly = TRUE)
  rscript <- file.path(R.home("bin"), "Rscript")

  message("[BOOTSTRAP] Relaunching Rscript once with WSL2 TensorFlow GPU library paths.")
  status <- system2(
    "env",
    args = c(
      "DL_RSCRIPT_REEXEC_WITH_GPU_LIBS=1",
      paste0("LD_LIBRARY_PATH=", new_ld),
      paste0("RETICULATE_PYTHON=", python_bin),
      rscript,
      script_file,
      trailing
    )
  )
  quit(save = "no", status = status, runLast = FALSE)
}

bootstrap_reexec_for_wsl_gpu()

# GPU / threading / parallel execution selection.
early_has_visible_gpu <- function() {
  cuda_visible <- Sys.getenv("CUDA_VISIBLE_DEVICES", unset = NA_character_)
  if (!is.na(cuda_visible) && trimws(cuda_visible) %in% c("", "-1")) return(FALSE)

  nvidia_smi <- Sys.which("nvidia-smi")
  if (!nzchar(nvidia_smi)) return(FALSE)

  smi_out <- tryCatch(
    system2(nvidia_smi, args = "-L", stdout = TRUE, stderr = FALSE),
    error = function(e) character(0)
  )
  length(smi_out) > 0 && any(grepl("GPU", smi_out, fixed = TRUE))
}

PARALLEL_MODE <- tolower(PARALLEL_MODE)
if (!PARALLEL_MODE %in% c("auto", "serial", "cpu")) {
  stop("PARALLEL_MODE must be one of: auto, serial, cpu")
}

requested_workers <- as.integer(REQUESTED_N_WORKERS)
requested_workers <- max(1L, requested_workers)
max_safe_workers <- max(1L, parallel::detectCores(logical = TRUE) - 2L)
gpu_detected_by_system <- early_has_visible_gpu()

USE_GPU_FOR_TF <- FALSE
if (PARALLEL_MODE == "cpu") {
  USE_GPU_FOR_TF <- FALSE
} else if (PARALLEL_MODE %in% c("auto", "serial") && isTRUE(gpu_detected_by_system)) {
  USE_GPU_FOR_TF <- TRUE
}

if (!isTRUE(USE_GPU_FOR_TF) && isTRUE(HIDE_CUDA_FOR_CPU_PARALLEL)) {
  Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
}

Sys.setenv(
  OPENBLAS_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  MKL_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  OMP_NUM_THREADS = BLAS_THREADS_PER_WORKER,
  TF_NUM_INTRAOP_THREADS = TF_INTRAOP_THREADS_PER_WORKER,
  TF_NUM_INTEROP_THREADS = TF_INTEROP_THREADS_PER_WORKER,
  TF_FORCE_GPU_ALLOW_GROWTH = TF_FORCE_GPU_ALLOW_GROWTH_VALUE,
  TF_CPP_MIN_LOG_LEVEL = TF_CPP_MIN_LOG_LEVEL_VALUE
)

if (PARALLEL_MODE == "serial" || isTRUE(USE_GPU_FOR_TF)) {
  N_WORKERS <- 1L
} else {
  N_WORKERS <- min(requested_workers, max_safe_workers)
}

# ----------------------------- Package setup -----------------------------------------
required_packages <- c(
  "data.table", "dplyr", "tidyr", "stringr", "purrr", "ggplot2",
  "caret", "glmnet", "keras3", "reticulate", "tensorflow", "Cairo"
)

missing_pkgs <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "),
       "\nInstall them with install.packages(c('", paste(missing_pkgs, collapse = "','"), "'))")
}

suppressPackageStartupMessages({
  invisible(lapply(required_packages, library, character.only = TRUE))
})

dir.create(file.path(OUT_DIR, "results"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "plots"), recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(OUT_DIR, "DL_feature_analysis_log.txt")
if (file.exists(log_file)) file.remove(log_file)

log_message <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste0(..., collapse = ""))
  message(msg)
  write(msg, file = log_file, append = TRUE)
}

safe_name <- function(x) gsub("[^A-Za-z0-9_]+", "_", x)

# ----------------------------- TensorFlow setup --------------------------------------

# Resolve the configured conda environment and lock reticulate to its Python.
# This script supports both Windows conda envs and WSL2/Linux conda envs.
resolve_conda_env <- function(env_name) {
  # First use fixed/common locations. This avoids needing DL_CONDA_PREFIX.
  early <- find_conda_env_prefix_early(env_name)
  if (!is.na(early) && nzchar(early)) {
    return(normalizePath(early, winslash = "/", mustWork = FALSE))
  }

  # Fallback: ask reticulate/conda.
  envs <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
  if (is.null(envs) || nrow(envs) == 0 || !(env_name %in% envs$name)) {
    stop(
      "Conda environment not found: ", env_name,
      "\nAvailable environments: ",
      if (is.null(envs) || nrow(envs) == 0) "<none>" else paste(envs$name, collapse = ", "),
      "\nExpected WSL2 default: ~/miniconda3/envs/dl_tf_wsl",
      "\nEdit USER_CONDA_PREFIX near the top of this script if your environment is elsewhere."
    )
  }

  py <- normalizePath(envs$python[match(env_name, envs$name)], winslash = "/", mustWork = FALSE)
  if (.Platform$OS.type == "windows") {
    normalizePath(dirname(py), winslash = "/", mustWork = FALSE)
  } else {
    d <- dirname(py)
    if (basename(d) == "bin") normalizePath(dirname(d), winslash = "/", mustWork = FALSE) else normalizePath(d, winslash = "/", mustWork = FALSE)
  }
}

python_executable_for_env <- function(env_base) {
  # Be tolerant if the caller accidentally passed <env>/bin rather than <env>.
  if (.Platform$OS.type == "windows") {
    candidates <- c(file.path(env_base, "python.exe"), file.path(dirname(env_base), "python.exe"))
  } else {
    candidates <- c(file.path(env_base, "bin", "python"), file.path(env_base, "python"))
  }
  candidates <- normalizePath(candidates, winslash = "/", mustWork = FALSE)
  hit <- candidates[file.exists(candidates) & file.access(candidates, mode = 1) == 0]
  if (length(hit) > 0) return(hit[1])
  candidates[1]
}

get_python_site_packages <- function(python_bin) {
  # Query Python site-packages safely. system2() needs the Python code quoted as
  # a single shell argument on WSL/Linux; otherwise the shell can split at ';' or '('.
  code <- "import site; print(site.getsitepackages()[0])"
  out <- tryCatch(
    system2(
      python_bin,
      args = c("-c", shQuote(code)),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) character(0)
  )

  # TensorFlow may print warnings to stderr; keep only the actual site-packages path.
  out <- out[grepl("site-packages$", out)]
  if (length(out) == 0) return(NA_character_)
  normalizePath(out[1], winslash = "/", mustWork = FALSE)
}

prepend_env_path <- function(var, values, sep = .Platform$path.sep) {
  # Safe named Sys.setenv(). Directly passing a named vector to Sys.setenv can fail
  # with "all arguments must be named"; do.call() with a named list is robust.
  values <- values[!is.na(values) & nzchar(values)]
  if (length(values) == 0) return(invisible(FALSE))
  values <- unique(normalizePath(values, winslash = "/", mustWork = FALSE))
  old <- Sys.getenv(var, unset = "")
  new_value <- paste(c(values, old), collapse = sep)
  args <- stats::setNames(list(new_value), var)
  do.call(Sys.setenv, args)
  invisible(TRUE)
}

env_base <- resolve_conda_env(CONDA_ENV)
python_bin <- python_executable_for_env(env_base)
Sys.setenv(DL_CONDA_PREFIX = env_base, DL_CONDA_ENV = CONDA_ENV)

if (!file.exists(python_bin) || file.access(python_bin, mode = 1) != 0) {
  stop(
    "Cannot access Python executable for conda environment '", CONDA_ENV, "'.",
    "
Expected Python: ", python_bin,
    "
file.exists(): ", file.exists(python_bin),
    "
file.access(..., mode = 1): ", file.access(python_bin, mode = 1),
    "
Repair/recreate the conda env, or set DL_CONDA_PREFIX to the working TensorFlow env."
  )
}

if (.Platform$OS.type == "windows") {
  conda_paths <- c(
    env_base,
    file.path(env_base, "Library", "mingw-w64", "bin"),
    file.path(env_base, "Library", "usr", "bin"),
    file.path(env_base, "Library", "bin"),
    file.path(env_base, "Scripts"),
    file.path(env_base, "bin")
  )
  prepend_env_path("PATH", conda_paths, sep = ";")
} else {
  # WSL2/Linux: TensorFlow pip CUDA wheels place CUDA/cuDNN shared libraries under
  # site-packages/nvidia/*/lib. Add them before reticulate initializes Python.
  site_pkgs <- get_python_site_packages(python_bin)
  nvidia_lib_dirs <- if (!is.na(site_pkgs)) Sys.glob(file.path(site_pkgs, "nvidia", "*", "lib")) else character(0)
  linux_paths <- c(file.path(env_base, "bin"), file.path(env_base, "lib"))
  wsl_cuda_paths <- c("/usr/lib/wsl/lib")

  prepend_env_path("PATH", linux_paths, sep = ":")
  prepend_env_path("LD_LIBRARY_PATH", c(wsl_cuda_paths, file.path(env_base, "lib"), nvidia_lib_dirs), sep = ":")

  # Optional but useful if ptxas was installed by nvidia-cuda-nvcc-cu12.
  ptxas_candidates <- if (!is.na(site_pkgs)) Sys.glob(file.path(site_pkgs, "nvidia", "cuda_nvcc", "*", "bin", "ptxas")) else character(0)
  if (length(ptxas_candidates) > 0) {
    prepend_env_path("PATH", dirname(ptxas_candidates[1]), sep = ":")
  }
}

Sys.setenv(RETICULATE_PYTHON = python_bin)
reticulate::use_python(python_bin, required = TRUE)

print("--- Checking Python configuration ---")
print(reticulate::py_config())

required_py_modules <- c("tensorflow", "keras", "numpy")
missing_py_modules <- required_py_modules[
  !vapply(required_py_modules, reticulate::py_module_available, logical(1))
]
if (length(missing_py_modules) > 0) {
  stop(
    "Python packages missing from conda environment '", CONDA_ENV, "': ",
    paste(missing_py_modules, collapse = ", "),
    "
Install them in the selected conda environment. For WSL2, for example:
",
    "conda activate poly_map
python -m pip install 'tensorflow[and-cuda]' keras numpy"
  )
}
# TensorFlow device setup. In auto mode, GPU execution is serial; CPU-only execution can use workers.
tensorflow::tf$random$set_seed(1234)

gpus <- tensorflow::tf$config$list_physical_devices("GPU")

if (length(gpus) > 0) {
  USE_GPU_FOR_TF <- TRUE
  N_WORKERS <- 1L

  log_message("SETUP: TensorFlow detected GPU(s): ",
              paste(vapply(gpus, function(x) x$name, character(1)), collapse = ", "))
  log_message("PARALLEL: GPU visible to TensorFlow; forcing N_WORKERS=1 to avoid GPU memory contention.")

  if (is.finite(GPU_MEMORY_LIMIT_MB) && GPU_MEMORY_LIMIT_MB > 0) {
    log_message("SETUP: Limiting TensorFlow GPU memory to ", GPU_MEMORY_LIMIT_MB, " MB on GPU 0.")
    tensorflow::tf$config$set_logical_device_configuration(
      gpus[[1]],
      list(tensorflow::tf$config$LogicalDeviceConfiguration(memory_limit = as.numeric(GPU_MEMORY_LIMIT_MB)))
    )
    logical_gpus <- tensorflow::tf$config$list_logical_devices("GPU")
    log_message("SETUP: Logical GPU(s): ",
                paste(vapply(logical_gpus, function(x) x$name, character(1)), collapse = ", "))
  } else {
    for (gpu in gpus) {
      try(tensorflow::tf$config$experimental$set_memory_growth(gpu, TRUE), silent = TRUE)
    }
    log_message("SETUP: TensorFlow memory growth enabled.")
  }
} else {
  msg <- "SETUP: TensorFlow did not detect a GPU."
  USE_GPU_FOR_TF <- FALSE

  if (isTRUE(REQUIRE_GPU)) {
    stop(msg, " Stop now because REQUIRE_GPU=TRUE.")
  } else {
    log_message(msg, " Continue on CPU because REQUIRE_GPU=FALSE.")
  }

  if (PARALLEL_MODE == "serial") {
    N_WORKERS <- 1L
  } else {
    N_WORKERS <- min(requested_workers, max_safe_workers)
  }
}

if (!isTRUE(USE_GPU_FOR_TF) && isTRUE(HIDE_CUDA_FOR_CPU_PARALLEL)) {
  Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
  log_message("PARALLEL: CPU-style execution selected; CUDA_VISIBLE_DEVICES=-1.")
}

log_message("PARALLEL: Final worker count = ", N_WORKERS,
            " | PARALLEL_MODE = ", PARALLEL_MODE,
            " | GPU detected by nvidia-smi = ", gpu_detected_by_system,
            " | TensorFlow GPU enabled = ", USE_GPU_FOR_TF)

# ----------------------------- Memory cleanup helpers --------------------------------

# WSL2 can be killed abruptly when TensorFlow/Keras, reticulate and R keep old graphs or
# large arrays alive across repeated CV folds.  These helpers are intentionally called
# before and after every DL fit, and also after each fold/repetition block.

safe_rm <- function(..., envir = parent.frame()) {
  vars <- as.character(substitute(list(...)))[-1]
  vars <- vars[vars %in% ls(envir = envir, all.names = TRUE)]
  if (length(vars) > 0) rm(list = vars, envir = envir)
  invisible(NULL)
}

clear_keras_session <- function(verbose = FALSE) {
  if (verbose) log_message("MEMORY: clearing TensorFlow/Keras/Python/R objects.")

  # Clear the Keras graph/session first.  This is the most important step when many
  # models are created sequentially in repeated CV.
  try(tensorflow::tf$keras$backend$clear_session(), silent = TRUE)

  # Ask Python's garbage collector to release reticulate-owned TensorFlow objects.
  try({
    py_gc <- reticulate::import("gc", delay_load = TRUE)
    py_gc$collect()
  }, silent = TRUE)

  # R-side garbage collection. Calling twice is sometimes useful after large arrays.
  invisible(gc(verbose = FALSE))
  invisible(gc(verbose = FALSE))
  invisible(NULL)
}

log_memory_snapshot <- function(prefix = "MEMORY") {
  # Lightweight optional memory logging.  Keeps the log informative without depending
  # on extra R packages.  nvidia-smi may be unavailable on CPU-only systems.
  if (.Platform$OS.type == "unix" && nzchar(Sys.which("free"))) {
    mem <- tryCatch(system2("free", args = c("-h"), stdout = TRUE, stderr = FALSE),
                    error = function(e) character(0))
    if (length(mem) >= 2) log_message(prefix, ": ", paste(mem[1:2], collapse = " | "))
  }
  if (nzchar(Sys.which("nvidia-smi"))) {
    gpu <- tryCatch(system2(
      "nvidia-smi",
      args = c("--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"),
      stdout = TRUE, stderr = FALSE
    ), error = function(e) character(0))
    if (length(gpu) > 0) log_message(prefix, ": GPU ", paste(gpu, collapse = " | "))
  }
  invisible(NULL)
}

# ----------------------------- Data loading functions --------------------------------
get_marker_id <- function(df) {
  cn <- colnames(df)

  if ("Marker" %in% cn) return(as.character(df$Marker))
  if ("marker" %in% cn) return(as.character(df$marker))

  chrom_col <- intersect(c("CHROM", "Chrom", "chrom", "CHR", "Chr", "chr"), cn)[1]
  pos_col <- intersect(c("POS", "Position", "position", "pos", "BP", "bp"), cn)[1]
  if (!is.na(chrom_col) && !is.na(pos_col)) {
    return(paste0(df[[chrom_col]], ":", df[[pos_col]]))
  }

  # Fallback: first column, but warn.
  warning("Could not infer Marker/CHROM:POS columns. Using first column as marker ID: ", cn[1])
  as.character(df[[1]])
}

sort_by_genome_position <- function(df) {
  cn <- colnames(df)
  chrom_col <- intersect(c("CHROM", "Chrom", "chrom", "CHR", "Chr", "chr"), cn)[1]
  pos_col <- intersect(c("POS", "Position", "position", "pos", "BP", "bp"), cn)[1]
  if (is.na(chrom_col) || is.na(pos_col)) return(df)

  df %>%
    dplyr::mutate(
      .chrom_sort = suppressWarnings(as.numeric(gsub("[^0-9]", "", .data[[chrom_col]]))),
      .chrom_sort = ifelse(is.na(.chrom_sort), as.numeric(factor(.data[[chrom_col]])), .chrom_sort),
      .pos_sort = suppressWarnings(as.numeric(.data[[pos_col]]))
    ) %>%
    dplyr::arrange(.chrom_sort, .pos_sort) %>%
    dplyr::select(-.chrom_sort, -.pos_sort)
}

read_dosage_as_matrix <- function(file, sample_ids) {
  log_message("DATA: Reading dosage file: ", file)

  sep <- if (grepl("\\.tsv$|\\.txt$", file, ignore.case = TRUE)) "\t" else ","
  df <- data.table::fread(file, sep = sep, check.names = FALSE, data.table = FALSE)
  df <- sort_by_genome_position(df)

  marker_ids <- get_marker_id(df)
  sample_cols <- intersect(colnames(df), sample_ids)

  if (length(sample_cols) == 0) {
    stop("No sample columns in genotype file matched phenotype IDs: ", file,
         "\nCheck whether the file uses the same sample IDs as phenotype_BLUPs.csv.")
  }

  mat <- as.matrix(df[, sample_cols, drop = FALSE])
  storage.mode(mat) <- "numeric"
  mat_t <- t(mat)
  rownames(mat_t) <- sample_cols
  colnames(mat_t) <- make.unique(marker_ids)

  mat_t
}

load_all_feature_matrices <- function(pheno_ids) {
  feature_mats <- list()

  # Full matrix
  if ("All" %in% REQUESTED_SNP_SETS) {
    feature_mats[["All"]] <- list()
    full_mat <- read_dosage_as_matrix(GENOTYPE_FILE, pheno_ids)
    # It will be reused across traits.
    for (tr in colnames(pheno_df)) {
      feature_mats[["All"]][[tr]] <- full_mat
    }
  }

  # Feature-reduced matrices from DL_feature_sets
  if (!dir.exists(FEATURE_DIR)) {
    log_message("DATA: WARNING - FEATURE_DIR does not exist: ", FEATURE_DIR)
    return(feature_mats)
  }

  feature_files <- list.files(
    FEATURE_DIR,
    pattern = "_top(500|1000|3000|6000).*\\.(csv|tsv|txt)$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  feature_files <- feature_files[!grepl("summary|markers|regions", basename(feature_files), ignore.case = TRUE)]

  log_message("DATA: Found ", length(feature_files), " feature-reduced dosage files in ", FEATURE_DIR)

  for (ff in feature_files) {
    bn <- basename(ff)
    m <- stringr::str_match(bn, "^(.*)_top(500|1000|3000|6000).*\\.(csv|tsv|txt)$")
    if (any(is.na(m))) {
      log_message("DATA: Skipping unrecognized feature file name: ", bn)
      next
    }

    trait <- m[2]
    snp_set <- paste0("Top", m[3])

    if (!snp_set %in% REQUESTED_SNP_SETS) next

    if (!trait %in% colnames(pheno_df)) {
      log_message("DATA: WARNING - Trait parsed from file is not in phenotype file: ", trait, " | file: ", bn)
      next
    }

    if (is.null(feature_mats[[snp_set]])) feature_mats[[snp_set]] <- list()
    feature_mats[[snp_set]][[trait]] <- read_dosage_as_matrix(ff, pheno_ids)
  }

  feature_mats
}

align_and_impute_global <- function(geno, pheno_df) {
  common <- intersect(rownames(geno), rownames(pheno_df))
  geno <- geno[common, , drop = FALSE]
  pheno_aligned <- pheno_df[common, , drop = FALSE]

  # Remove fully invariant markers and impute remaining missing values with global means.
  vars <- apply(geno, 2, var, na.rm = TRUE)
  keep <- which(is.finite(vars) & vars > 1e-8)
  geno <- geno[, keep, drop = FALSE]

  col_means <- colMeans(geno, na.rm = TRUE)
  col_means[is.nan(col_means)] <- 0
  miss <- which(is.na(geno), arr.ind = TRUE)
  if (nrow(miss) > 0) geno[miss] <- col_means[miss[, "col"]]

  list(geno = geno, pheno = pheno_aligned)
}

scale_split <- function(x_train, x_test) {
  # Remove markers that are invariant in the actual training set.
  vars <- apply(x_train, 2, var, na.rm = TRUE)
  keep <- which(is.finite(vars) & vars > 1e-8)
  x_train <- x_train[, keep, drop = FALSE]
  x_test  <- x_test[, keep, drop = FALSE]

  train_mean <- colMeans(x_train, na.rm = TRUE)
  train_mean[is.nan(train_mean)] <- 0
  train_sd <- apply(x_train, 2, sd, na.rm = TRUE)
  train_sd[is.na(train_sd) | train_sd == 0] <- 1

  miss_train <- which(is.na(x_train), arr.ind = TRUE)
  if (nrow(miss_train) > 0) x_train[miss_train] <- train_mean[miss_train[, "col"]]
  miss_test <- which(is.na(x_test), arr.ind = TRUE)
  if (nrow(miss_test) > 0) x_test[miss_test] <- train_mean[miss_test[, "col"]]

  list(
    train = scale(x_train, center = train_mean, scale = train_sd),
    test = scale(x_test, center = train_mean, scale = train_sd),
    n_markers = length(keep)
  )
}

# ----------------------------- Model functions ---------------------------------------
default_mlp_params <- function() {
  list(neurons = 64, dropout_rate = 0.4, learning_rate = 0.001)
}

default_cnn_params <- function() {
  list(filters = 32, kernel_size = 8, learning_rate = 0.001)
}

build_mlp_model <- function(num_markers, params) {
  model <- keras_model_sequential(input_shape = c(as.integer(num_markers))) %>%
    layer_dense(
      units = as.integer(params$neurons),
      kernel_regularizer = regularizer_l2(l2 = 0.001)
    ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_dropout(rate = as.numeric(params$dropout_rate)) %>%
    layer_dense(
      units = as.integer(max(8, round(params$neurons / 2))),
      kernel_regularizer = regularizer_l2(l2 = 0.001)
    ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_dense(units = 1)

  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(learning_rate = params$learning_rate)
  )
  model
}

build_cnn_model <- function(num_markers, params) {
  kernel <- as.integer(min(params$kernel_size, num_markers))
  pool <- ifelse(num_markers >= 4, 4, 1)

  model <- keras_model_sequential(input_shape = c(as.integer(num_markers), 1L)) %>%
    layer_conv_1d(
      filters = as.integer(params$filters),
      kernel_size = kernel,
      kernel_regularizer = regularizer_l2(l2 = 0.001)
    ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  if (pool > 1) {
    model <- model %>% layer_max_pooling_1d(pool_size = pool)
  }

  model <- model %>%
    layer_flatten() %>%
    layer_dense(units = 64, kernel_regularizer = regularizer_l2(l2 = 0.001)) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_dense(units = 1)

  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(learning_rate = params$learning_rate)
  )
  model
}

callbacks_for_training <- function() {
  list(
    callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)
  )
}

history_to_df <- function(history, trait, snp_set, model, analysis, repetition, fold,
                          training_fraction = NA_real_, direction = NA_character_) {
  metrics <- as.data.frame(history$metrics)
  if (!"loss" %in% colnames(metrics)) metrics$loss <- NA_real_
  if (!"val_loss" %in% colnames(metrics)) metrics$val_loss <- NA_real_

  tibble::tibble(
    Trait = trait,
    SNP_Set = snp_set,
    Model = model,
    Analysis = analysis,
    Repetition = repetition,
    Fold = fold,
    Training_fraction = training_fraction,
    Direction = direction,
    Epoch = seq_len(nrow(metrics)),
    loss = metrics$loss,
    val_loss = metrics$val_loss
  )
}

fit_predict_model <- function(model_name, x_train, y_train, x_test, y_test,
                              params, trait, snp_set, analysis,
                              repetition, fold, training_fraction = NA_real_,
                              direction = NA_character_) {
  x_train <- as.matrix(x_train)
  x_test <- as.matrix(x_test)
  storage.mode(x_train) <- "double"
  storage.mode(x_test) <- "double"
  y_train <- as.numeric(y_train)
  y_test <- as.numeric(y_test)

  if (length(unique(y_train[!is.na(y_train)])) < 3) {
    return(list(cor = NA_real_, history = NULL, n_epochs = NA_integer_))
  }

  if (model_name == "Ridge") {
    fit <- glmnet::cv.glmnet(x_train, y_train, alpha = 0, family = "gaussian")
    pred <- predict(fit, newx = x_test, s = "lambda.min")[, 1]
    out <- list(
      cor = suppressWarnings(cor(pred, y_test, use = "complete.obs")),
      history = NULL,
      n_epochs = NA_integer_
    )
    safe_rm(fit, pred)
    invisible(gc(verbose = FALSE))
    return(out)
  }

  # DL models: clear old TensorFlow graphs before building the next model and guarantee
  # cleanup even if model$fit() or predict() raises OOM / ResourceExhaustedError.
  clear_keras_session(verbose = FALSE)
  callbacks <- callbacks_for_training()
  batch_size_this_fit <- as.integer(BATCH_SIZE)

  on.exit({
    safe_rm(model, hist, pred, hist_df, x_train_cnn, x_test_cnn, callbacks,
            batch_size_this_fit, envir = environment())
    clear_keras_session(verbose = FALSE)
  }, add = TRUE)

  if (model_name == "MLP") {
    model <- build_mlp_model(ncol(x_train), params$mlp)
    hist <- model %>% fit(
      x_train, y_train,
      epochs = as.integer(TRAINING_EPOCHS),
      batch_size = batch_size_this_fit,
      validation_split = INNER_VALIDATION_FRACTION,
      verbose = 0,
      callbacks = callbacks
    )
    pred <- model %>% predict(x_test, verbose = 0)
    hist_df <- history_to_df(
      hist, trait, snp_set, model_name, analysis, repetition, fold,
      training_fraction, direction
    )
    return(list(
      cor = suppressWarnings(cor(as.numeric(pred[, 1]), y_test, use = "complete.obs")),
      history = hist_df,
      n_epochs = nrow(hist_df)
    ))
  }

  if (model_name == "CNN") {
    x_train_cnn <- array(as.numeric(x_train), dim = c(nrow(x_train), ncol(x_train), 1))
    x_test_cnn  <- array(as.numeric(x_test),  dim = c(nrow(x_test),  ncol(x_test), 1))
    model <- build_cnn_model(ncol(x_train), params$cnn)
    hist <- model %>% fit(
      x_train_cnn, y_train,
      epochs = as.integer(TRAINING_EPOCHS),
      batch_size = batch_size_this_fit,
      validation_split = INNER_VALIDATION_FRACTION,
      verbose = 0,
      callbacks = callbacks
    )
    pred <- model %>% predict(x_test_cnn, verbose = 0)
    hist_df <- history_to_df(
      hist, trait, snp_set, model_name, analysis, repetition, fold,
      training_fraction, direction
    )
    return(list(
      cor = suppressWarnings(cor(as.numeric(pred[, 1]), y_test, use = "complete.obs")),
      history = hist_df,
      n_epochs = nrow(hist_df)
    ))
  }

  stop("Unknown model: ", model_name)
}

tune_params_once <- function(geno, y, trait, snp_set) {
  on.exit({
    safe_rm(x_train_raw, x_val_raw, x_train, x_val, y_train, y_val,
            x_train_cnn, x_val_cnn, model, hist, grid, p, idx, sc,
            envir = environment())
    clear_keras_session(verbose = FALSE)
  }, add = TRUE)

  params_out <- list(mlp = default_mlp_params(), cnn = default_cnn_params())

  if (!DO_TUNING) {
    return(params_out)
  }

  set.seed(123)
  idx <- make_train_validation_indices(y, p = 0.8)
  x_train_raw <- geno[idx, , drop = FALSE]
  x_val_raw <- geno[-idx, , drop = FALSE]
  y_train <- y[idx]
  y_val <- y[-idx]

  sc <- scale_split(x_train_raw, x_val_raw)
  x_train <- sc$train
  x_val <- sc$test

  callbacks <- callbacks_for_training()

  tuning_records <- list()

  if ("MLP" %in% MODELS_TO_RUN) {
    grid <- expand.grid(
      neurons = c(64, 128),
      dropout_rate = c(0.4, 0.6),
      learning_rate = c(0.005, 0.001),
      KEEP.OUT.ATTRS = FALSE
    )
    best_loss <- Inf
    for (j in seq_len(nrow(grid))) {
      p <- as.list(grid[j, ])
      clear_keras_session(verbose = FALSE)
      model <- build_mlp_model(ncol(x_train), p)
      hist <- model %>% fit(
        x_train, y_train,
        epochs = TRAINING_EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data = list(x_val, y_val),
        verbose = 0,
        callbacks = callbacks
      )
      val_loss <- min(hist$metrics$val_loss, na.rm = TRUE)
      tuning_records[[length(tuning_records) + 1]] <- tibble::tibble(
        Trait = trait, SNP_Set = snp_set, Model = "MLP", Trial = j,
        val_loss = val_loss, neurons = p$neurons,
        dropout_rate = p$dropout_rate, learning_rate = p$learning_rate,
        filters = NA, kernel_size = NA
      )
      if (is.finite(val_loss) && val_loss < best_loss) {
        best_loss <- val_loss
        params_out$mlp <- p
      }
      safe_rm(model, hist, envir = environment())
      clear_keras_session(verbose = FALSE)
    }
  }

  if ("CNN" %in% MODELS_TO_RUN) {
    grid <- expand.grid(
      filters = c(32, 64),
      kernel_size = c(8, 12),
      learning_rate = c(0.005, 0.001),
      KEEP.OUT.ATTRS = FALSE
    )
    best_loss <- Inf
    x_train_cnn <- array(as.numeric(x_train), dim = c(nrow(x_train), ncol(x_train), 1))
    x_val_cnn <- array(as.numeric(x_val), dim = c(nrow(x_val), ncol(x_val), 1))

    for (j in seq_len(nrow(grid))) {
      p <- as.list(grid[j, ])
      clear_keras_session(verbose = FALSE)
      model <- build_cnn_model(ncol(x_train), p)
      hist <- model %>% fit(
        x_train_cnn, y_train,
        epochs = TRAINING_EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data = list(x_val_cnn, y_val),
        verbose = 0,
        callbacks = callbacks
      )
      val_loss <- min(hist$metrics$val_loss, na.rm = TRUE)
      tuning_records[[length(tuning_records) + 1]] <- tibble::tibble(
        Trait = trait, SNP_Set = snp_set, Model = "CNN", Trial = j,
        val_loss = val_loss, neurons = NA,
        dropout_rate = NA, learning_rate = p$learning_rate,
        filters = p$filters, kernel_size = p$kernel_size
      )
      if (is.finite(val_loss) && val_loss < best_loss) {
        best_loss <- val_loss
        params_out$cnn <- p
      }
      safe_rm(model, hist, envir = environment())
      clear_keras_session(verbose = FALSE)
    }
    safe_rm(x_train_cnn, x_val_cnn, envir = environment())
    clear_keras_session(verbose = FALSE)
  }

  attr(params_out, "tuning_records") <- dplyr::bind_rows(tuning_records)
  params_out
}

append_csv <- function(df, file) {
  if (is.null(df) || nrow(df) == 0) return(invisible(NULL))
  data.table::fwrite(df, file, append = file.exists(file), col.names = !file.exists(file))
}


# Robust CV helpers for quantitative traits.
# caret::createFolds() can fail for edge-case phenotype distributions, so these helpers
# create regression-style stratified folds using phenotype quantile bins.
make_regression_folds <- function(y, k) {
  y <- as.numeric(y)
  n <- length(y)
  if (n < 2) stop("Need at least 2 individuals to make CV folds.")
  k <- min(as.integer(k), n)
  if (k < 2) stop("NUM_FOLDS must be at least 2 after limiting by sample size.")

  probs <- seq(0, 1, length.out = min(6, n) + 1)
  breaks <- unique(as.numeric(stats::quantile(y, probs = probs, na.rm = TRUE, type = 7)))
  if (length(breaks) >= 3) {
    bins <- cut(y, breaks = breaks, include.lowest = TRUE, labels = FALSE)
    bins[is.na(bins)] <- 1L
  } else {
    bins <- rep(1L, n)
  }

  folds <- vector("list", k)
  for (i in seq_len(k)) folds[[i]] <- integer(0)

  for (b in sort(unique(bins))) {
    idx <- sample(which(bins == b))
    for (j in seq_along(idx)) {
      fold_id <- ((j - 1) %% k) + 1L
      folds[[fold_id]] <- c(folds[[fold_id]], idx[j])
    }
  }

  folds <- lapply(folds, sort)
  names(folds) <- paste0("Fold", seq_len(k))
  folds
}

make_train_validation_indices <- function(y, p = 0.8) {
  y <- as.numeric(y)
  n <- length(y)
  n_train_target <- max(2L, min(n - 1L, floor(n * p)))

  probs <- seq(0, 1, length.out = min(6, n) + 1)
  breaks <- unique(as.numeric(stats::quantile(y, probs = probs, na.rm = TRUE, type = 7)))
  if (length(breaks) >= 3) {
    bins <- cut(y, breaks = breaks, include.lowest = TRUE, labels = FALSE)
    bins[is.na(bins)] <- 1L
  } else {
    bins <- rep(1L, n)
  }

  train_idx <- integer(0)
  for (b in sort(unique(bins))) {
    idx <- sample(which(bins == b))
    n_b <- if (length(idx) <= 1) length(idx) else max(1L, floor(length(idx) * p))
    train_idx <- c(train_idx, idx[seq_len(n_b)])
  }

  if (length(train_idx) > n_train_target) {
    train_idx <- sample(train_idx, n_train_target)
  }
  if (length(train_idx) < n_train_target) {
    add <- setdiff(seq_len(n), train_idx)
    train_idx <- c(train_idx, sample(add, n_train_target - length(train_idx)))
  }

  sort(unique(train_idx))
}

# ----------------------------- Evaluation functions ----------------------------------
run_full_population <- function(geno, y, trait, snp_set, params) {
  res <- list()
  hist <- list()
  n <- length(y)

  for (rep_id in seq_len(NUM_REPEATS_FULL)) {
    set.seed(42000 + rep_id)
    folds <- make_regression_folds(y, NUM_FOLDS)

    for (fold_id in seq_along(folds)) {
      test_idx <- folds[[fold_id]]
      train_idx <- setdiff(seq_len(n), test_idx)

      sc <- scale_split(geno[train_idx, , drop = FALSE], geno[test_idx, , drop = FALSE])
      x_train <- sc$train
      x_test <- sc$test
      y_train <- y[train_idx]
      y_test <- y[test_idx]

      for (model_name in MODELS_TO_RUN) {
        log_message("FULL: ", trait, " | ", snp_set, " | rep ", rep_id, " fold ", fold_id, " | ", model_name)
        out <- tryCatch(
          fit_predict_model(model_name, x_train, y_train, x_test, y_test,
                            params, trait, snp_set, "FullPopulation",
                            rep_id, fold_id),
          error = function(e) {
            log_message("ERROR FULL: ", trait, " | ", snp_set, " | ", model_name, " | ", e$message)
            list(cor = NA_real_, history = NULL, n_epochs = NA_integer_)
          }
        )

        res[[length(res) + 1]] <- tibble::tibble(
          Analysis = "FullPopulation",
          Trait = trait,
          SNP_Set = snp_set,
          n_markers = ncol(x_train),
          Repetition = rep_id,
          Fold = fold_id,
          Model = model_name,
          Cor = out$cor,
          n_train = length(train_idx),
          n_test = length(test_idx),
          Training_fraction = NA_real_,
          Direction = NA_character_,
          n_epochs = out$n_epochs
        )
        if (!is.null(out$history)) hist[[length(hist) + 1]] <- out$history
      }

      append_csv(dplyr::bind_rows(res), file.path(OUT_DIR, "results", "DL_full_population_feature_results.csv"))
      if (length(hist) > 0) append_csv(dplyr::bind_rows(hist), file.path(OUT_DIR, "results", "DL_training_history.csv"))
      res <- list()
      hist <- list()

      # Release fold-level matrices before moving to the next fold.
      safe_rm(sc, x_train, x_test, y_train, y_test, out, envir = environment())
      clear_keras_session(verbose = FALSE)
    }
  }

  invisible(TRUE)
}

run_cross_population <- function(geno, y, trait, snp_set, params, ids) {
  pop1 <- ids[startsWith(ids, "NH") | startsWith(ids, "HN")]
  pop2 <- ids[startsWith(ids, "HP")]

  if (length(pop1) < 10 || length(pop2) < 10) {
    log_message("CROSS: Skipping ", trait, " | ", snp_set, " because one population has <10 individuals.")
    return(invisible(FALSE))
  }

  directions <- list(
    NH_HN_to_HP = list(train = pop1, test = pop2),
    HP_to_NH_HN = list(train = pop2, test = pop1)
  )

  res <- list()
  hist <- list()

  for (direction_name in names(directions)) {
    train_ids <- directions[[direction_name]]$train
    test_ids <- directions[[direction_name]]$test

    # Repeat DL training with different random seeds to capture stochastic variability.
    for (rep_id in seq_len(NUM_REPEATS_CROSS)) {
      set.seed(52000 + rep_id)

      sc <- scale_split(geno[train_ids, , drop = FALSE], geno[test_ids, , drop = FALSE])
      x_train <- sc$train
      x_test <- sc$test
      y_train <- y[train_ids]
      y_test <- y[test_ids]

      for (model_name in MODELS_TO_RUN) {
        log_message("CROSS: ", trait, " | ", snp_set, " | ", direction_name, " | rep ", rep_id, " | ", model_name)
        out <- tryCatch(
          fit_predict_model(model_name, x_train, y_train, x_test, y_test,
                            params, trait, snp_set, "CrossPopulation",
                            rep_id, NA_integer_, direction = direction_name),
          error = function(e) {
            log_message("ERROR CROSS: ", trait, " | ", snp_set, " | ", model_name, " | ", e$message)
            list(cor = NA_real_, history = NULL, n_epochs = NA_integer_)
          }
        )

        res[[length(res) + 1]] <- tibble::tibble(
          Analysis = "CrossPopulation",
          Trait = trait,
          SNP_Set = snp_set,
          n_markers = ncol(x_train),
          Repetition = rep_id,
          Fold = NA_integer_,
          Model = model_name,
          Cor = out$cor,
          n_train = length(train_ids),
          n_test = length(test_ids),
          Training_fraction = NA_real_,
          Direction = direction_name,
          n_epochs = out$n_epochs
        )
        if (!is.null(out$history)) hist[[length(hist) + 1]] <- out$history
      }

      append_csv(dplyr::bind_rows(res), file.path(OUT_DIR, "results", "DL_cross_population_feature_results.csv"))
      if (length(hist) > 0) append_csv(dplyr::bind_rows(hist), file.path(OUT_DIR, "results", "DL_training_history.csv"))
      res <- list()
      hist <- list()

      # Release repetition-level matrices before moving to the next stochastic repeat.
      safe_rm(sc, x_train, x_test, y_train, y_test, out, envir = environment())
      clear_keras_session(verbose = FALSE)
    }
  }

  invisible(TRUE)
}

run_saturation <- function(geno, y, trait, snp_set, params) {
  res <- list()
  hist <- list()
  n <- length(y)

  for (rep_id in seq_len(NUM_REPEATS_SAT)) {
    set.seed(62000 + rep_id)
    folds <- make_regression_folds(y, NUM_FOLDS)

    for (fold_id in seq_along(folds)) {
      test_idx <- folds[[fold_id]]
      train_pool <- setdiff(seq_len(n), test_idx)

      for (frac in SAT_FRACTIONS) {
        set.seed(70000 + rep_id * 1000 + fold_id * 10 + round(frac * 100))
        n_sub <- max(20, floor(length(train_pool) * frac))
        n_sub <- min(n_sub, length(train_pool))
        train_idx <- sample(train_pool, n_sub)

        sc <- scale_split(geno[train_idx, , drop = FALSE], geno[test_idx, , drop = FALSE])
        x_train <- sc$train
        x_test <- sc$test
        y_train <- y[train_idx]
        y_test <- y[test_idx]

        for (model_name in MODELS_TO_RUN) {
          log_message("SAT: ", trait, " | ", snp_set, " | rep ", rep_id, " fold ", fold_id,
                      " | frac ", frac, " | ", model_name)
          out <- tryCatch(
            fit_predict_model(model_name, x_train, y_train, x_test, y_test,
                              params, trait, snp_set, "Saturation",
                              rep_id, fold_id, training_fraction = frac),
            error = function(e) {
              log_message("ERROR SAT: ", trait, " | ", snp_set, " | ", model_name, " | ", e$message)
              list(cor = NA_real_, history = NULL, n_epochs = NA_integer_)
            }
          )

          res[[length(res) + 1]] <- tibble::tibble(
            Analysis = "Saturation",
            Trait = trait,
            SNP_Set = snp_set,
            n_markers = ncol(x_train),
            Repetition = rep_id,
            Fold = fold_id,
            Model = model_name,
            Cor = out$cor,
            n_train = length(train_idx),
            n_test = length(test_idx),
            Training_fraction = frac,
            Direction = NA_character_,
            n_epochs = out$n_epochs
          )
          if (!is.null(out$history)) hist[[length(hist) + 1]] <- out$history
        }

        append_csv(dplyr::bind_rows(res), file.path(OUT_DIR, "results", "DL_saturation_feature_results.csv"))
        if (length(hist) > 0) append_csv(dplyr::bind_rows(hist), file.path(OUT_DIR, "results", "DL_training_history.csv"))
        res <- list()
        hist <- list()

        # Release saturation-subset matrices before the next fraction/fold.
        safe_rm(sc, x_train, x_test, y_train, y_test, out, train_idx, envir = environment())
        clear_keras_session(verbose = FALSE)
      }
    }
  }

  invisible(TRUE)
}

# ----------------------------- Plot functions ----------------------------------------
plot_results <- function() {
  full_file <- file.path(OUT_DIR, "results", "DL_full_population_feature_results.csv")
  cross_file <- file.path(OUT_DIR, "results", "DL_cross_population_feature_results.csv")
  sat_file <- file.path(OUT_DIR, "results", "DL_saturation_feature_results.csv")
  hist_file <- file.path(OUT_DIR, "results", "DL_training_history.csv")

  if (file.exists(full_file)) {
    full <- data.table::fread(full_file, data.table = FALSE)
    full_summary <- full %>%
      dplyr::group_by(Trait, SNP_Set, Model) %>%
      dplyr::summarise(
        mean_Cor = mean(Cor, na.rm = TRUE),
        sd_Cor = sd(Cor, na.rm = TRUE),
        se_Cor = sd_Cor / sqrt(sum(!is.na(Cor))),
        n_runs = sum(!is.na(Cor)),
        .groups = "drop"
      )
    data.table::fwrite(full_summary, file.path(OUT_DIR, "results", "DL_full_population_feature_summary.csv"))

    p <- ggplot(full_summary, aes(x = SNP_Set, y = mean_Cor, color = Model, group = Model)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2) +
      geom_errorbar(aes(ymin = mean_Cor - se_Cor, ymax = mean_Cor + se_Cor), width = 0.15) +
      facet_wrap(~ Trait, scales = "free_y") +
      labs(
        title = "Feature-reduced Full-population Prediction Accuracy",
        x = "SNP feature set",
        y = "Prediction accuracy (Pearson correlation)"
      ) +
      theme_classic(base_size = 11) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ggsave(file.path(OUT_DIR, "plots", "DL_full_population_feature_accuracy.pdf"), p,
           width = 14, height = 9, device = cairo_pdf)
  }

  if (file.exists(cross_file)) {
    cross <- data.table::fread(cross_file, data.table = FALSE)
    cross_summary <- cross %>%
      dplyr::group_by(Trait, SNP_Set, Direction, Model) %>%
      dplyr::summarise(
        mean_Cor = mean(Cor, na.rm = TRUE),
        sd_Cor = sd(Cor, na.rm = TRUE),
        se_Cor = sd_Cor / sqrt(sum(!is.na(Cor))),
        n_runs = sum(!is.na(Cor)),
        .groups = "drop"
      )
    data.table::fwrite(cross_summary, file.path(OUT_DIR, "results", "DL_cross_population_feature_summary.csv"))

    p <- ggplot(cross_summary, aes(x = SNP_Set, y = mean_Cor, color = Model, group = Model)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2) +
      geom_errorbar(aes(ymin = mean_Cor - se_Cor, ymax = mean_Cor + se_Cor), width = 0.15) +
      facet_grid(Direction ~ Trait, scales = "free_y") +
      labs(
        title = "Feature-reduced Cross-population Prediction Accuracy",
        x = "SNP feature set",
        y = "Prediction accuracy (Pearson correlation)"
      ) +
      theme_classic(base_size = 10) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ggsave(file.path(OUT_DIR, "plots", "DL_cross_population_feature_accuracy.pdf"), p,
           width = 16, height = 10, device = cairo_pdf)
  }

  if (file.exists(sat_file)) {
    sat <- data.table::fread(sat_file, data.table = FALSE)
    sat_summary <- sat %>%
      dplyr::group_by(Trait, SNP_Set, Training_fraction, n_train, Model) %>%
      dplyr::summarise(
        mean_Cor = mean(Cor, na.rm = TRUE),
        sd_Cor = sd(Cor, na.rm = TRUE),
        se_Cor = sd_Cor / sqrt(sum(!is.na(Cor))),
        n_runs = sum(!is.na(Cor)),
        .groups = "drop"
      )
    data.table::fwrite(sat_summary, file.path(OUT_DIR, "results", "DL_saturation_feature_summary.csv"))

    p <- ggplot(sat_summary, aes(x = n_train, y = mean_Cor, color = Model, group = Model)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2) +
      geom_errorbar(aes(ymin = mean_Cor - se_Cor, ymax = mean_Cor + se_Cor), width = 5) +
      facet_grid(SNP_Set ~ Trait, scales = "free_y") +
      labs(
        title = "Training-size Saturation Analysis",
        x = "Training population size",
        y = "Prediction accuracy (Pearson correlation)"
      ) +
      theme_classic(base_size = 9)
    ggsave(file.path(OUT_DIR, "plots", "DL_training_size_saturation.pdf"), p,
           width = 16, height = 14, device = cairo_pdf)
  }

  if (file.exists(hist_file)) {
    hist <- data.table::fread(hist_file, data.table = FALSE)
    hist_summary <- hist %>%
      dplyr::group_by(Trait, SNP_Set, Model, Analysis, Epoch) %>%
      dplyr::summarise(
        mean_loss = mean(loss, na.rm = TRUE),
        mean_val_loss = mean(val_loss, na.rm = TRUE),
        n_runs = dplyr::n(),
        .groups = "drop"
      )

    data.table::fwrite(hist_summary, file.path(OUT_DIR, "results", "DL_training_history_summary.csv"))

    p <- ggplot(hist_summary %>% dplyr::filter(Analysis == "FullPopulation"),
                aes(x = Epoch, color = Model)) +
      geom_line(aes(y = mean_loss, linetype = "Training"), linewidth = 0.65) +
      geom_line(aes(y = mean_val_loss, linetype = "Inner validation"), linewidth = 0.65) +
      facet_grid(SNP_Set ~ Trait, scales = "free_y") +
      labs(
        title = "MLP/CNN Training and Inner-validation Loss Curves",
        x = "Epoch",
        y = "Mean squared error loss",
        linetype = "Loss"
      ) +
      theme_classic(base_size = 9)
    ggsave(file.path(OUT_DIR, "plots", "DL_loss_curves_full_population.pdf"), p,
           width = 16, height = 14, device = cairo_pdf)
  }
}

# ----------------------------- Main workflow -----------------------------------------
output_files_to_reset <- c(
  "DL_full_population_feature_results.csv",
  "DL_cross_population_feature_results.csv",
  "DL_saturation_feature_results.csv",
  "DL_tuning_summary.csv",
  "DL_training_history.csv",
  "DL_full_population_feature_summary.csv",
  "DL_cross_population_feature_summary.csv",
  "DL_saturation_feature_summary.csv",
  "DL_training_history_summary.csv"
)

reset_output_files <- function(out_dir = OUT_DIR) {
  for (ff in output_files_to_reset) {
    path <- file.path(out_dir, "results", ff)
    if (file.exists(path)) file.remove(path)
  }
}

aggregate_worker_outputs <- function(worker_root, final_out_dir = OUT_DIR) {
  dir.create(file.path(final_out_dir, "results"), recursive = TRUE, showWarnings = FALSE)

  raw_files <- c(
    "DL_full_population_feature_results.csv",
    "DL_cross_population_feature_results.csv",
    "DL_saturation_feature_results.csv",
    "DL_tuning_summary.csv",
    "DL_training_history.csv"
  )

  for (ff in raw_files) {
    paths <- list.files(worker_root, pattern = paste0("^", ff, "$"),
                        recursive = TRUE, full.names = TRUE)
    if (length(paths) == 0) next

    combined <- data.table::rbindlist(
      lapply(paths, function(x) data.table::fread(x, data.table = FALSE)),
      fill = TRUE
    )
    data.table::fwrite(combined, file.path(final_out_dir, "results", ff))
  }
}

run_one_trait_snp_set <- function(trait, snp_set, worker_out_dir = NA_character_,
                                  make_plots_during_run = TRUE) {
  if (!is.na(worker_out_dir) && nzchar(worker_out_dir)) {
    OUT_DIR <<- worker_out_dir
    dir.create(file.path(OUT_DIR, "results"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(OUT_DIR, "plots"), recursive = TRUE, showWarnings = FALSE)
    log_file <<- file.path(OUT_DIR, "DL_feature_analysis_log.txt")
    if (file.exists(log_file)) file.remove(log_file)
  }

  if (is.null(feature_mats[[snp_set]]) || is.null(feature_mats[[snp_set]][[trait]])) {
    log_message("MAIN: Missing feature matrix for Trait=", trait, " SNP_Set=", snp_set, ". Skipping.")
    return(invisible(FALSE))
  }

  log_message("====================================================================")
  log_message("MAIN: Trait=", trait, " | SNP_Set=", snp_set)
  log_message("====================================================================")

  aligned <- align_and_impute_global(feature_mats[[snp_set]][[trait]], pheno_df[, trait, drop = FALSE])
  geno <- aligned$geno
  y_df <- aligned$pheno
  y <- as.numeric(y_df[[trait]])
  names(y) <- rownames(y_df)
  geno <- geno[names(y), , drop = FALSE]

  finite_y <- is.finite(y)
  if (!all(finite_y)) {
    log_message("MAIN: Removing ", sum(!finite_y),
                " individuals with missing/non-finite phenotype for Trait=", trait)
    y <- y[finite_y]
    geno <- geno[names(y), , drop = FALSE]
  }

  log_message("MAIN: Prepared ", nrow(geno), " individuals and ", ncol(geno), " markers.")

  if (nrow(geno) < 50 || ncol(geno) < 2) {
    log_message("MAIN: Too few individuals or markers. Skipping Trait=", trait, " SNP_Set=", snp_set)
    return(invisible(FALSE))
  }

  # Tune hyperparameters once per trait x SNP set, then reuse across analyses.
  params <- tune_params_once(geno, y, trait, snp_set)
  tuning_rec <- attr(params, "tuning_records")
  if (!is.null(tuning_rec) && nrow(tuning_rec) > 0) {
    append_csv(tuning_rec, file.path(OUT_DIR, "results", "DL_tuning_summary.csv"))
  }

  log_message("MAIN: Params | MLP neurons=", params$mlp$neurons,
              ", dropout=", params$mlp$dropout_rate,
              ", lr=", params$mlp$learning_rate,
              " | CNN filters=", params$cnn$filters,
              ", kernel=", params$cnn$kernel_size,
              ", lr=", params$cnn$learning_rate)

  if (RUN_FULL_POPULATION) {
    run_full_population(geno, y, trait, snp_set, params)
  }

  if (RUN_CROSS_POPULATION) {
    run_cross_population(geno, y, trait, snp_set, params, rownames(geno))
  }

  if (RUN_SATURATION) {
    run_saturation(geno, y, trait, snp_set, params)
  }

  if (isTRUE(make_plots_during_run)) {
    plot_results()
  }

  # Release trait x SNP-set level objects before the next combination.
  safe_rm(aligned, geno, y_df, y, finite_y, params, tuning_rec, envir = environment())
  clear_keras_session(verbose = TRUE)
  invisible(TRUE)
}

log_message("DATA: Loading phenotype file: ", PHENOTYPE_FILE)
pheno_df <- read.csv(PHENOTYPE_FILE, row.names = 1, check.names = FALSE, sep = ",", header = TRUE)
pheno_df <- as.data.frame(pheno_df)
pheno_df <- type.convert(pheno_df, as.is = TRUE)

traits_to_run <- TRAITS_TO_RUN_FIXED
missing_traits <- setdiff(traits_to_run, colnames(pheno_df))
if (length(missing_traits) > 0) {
  stop("Traits not found in phenotype file: ", paste(missing_traits, collapse = ", "))
}

log_message("MAIN: Traits to run: ", paste(traits_to_run, collapse = ", "))
log_message("MAIN: Requested SNP sets: ", paste(REQUESTED_SNP_SETS, collapse = ", "))
log_message("MAIN: Models: ", paste(MODELS_TO_RUN, collapse = ", "))
log_message("MAIN: Epochs=", TRAINING_EPOCHS,
            " | Batch size=", BATCH_SIZE,
            " | Full repeats=", NUM_REPEATS_FULL,
            " | Cross repeats=", NUM_REPEATS_CROSS,
            " | Saturation repeats=", NUM_REPEATS_SAT)
log_memory_snapshot("MEMORY START")

reset_output_files(OUT_DIR)
worker_root <- file.path(OUT_DIR, "worker_runs")
if (dir.exists(worker_root)) unlink(worker_root, recursive = TRUE, force = TRUE)

feature_mats <- load_all_feature_matrices(rownames(pheno_df))

task_grid <- expand.grid(
  Trait = traits_to_run,
  SNP_Set = REQUESTED_SNP_SETS,
  stringsAsFactors = FALSE
)

# Keep only trait x SNP-set combinations that actually have a matrix.
task_grid <- task_grid[vapply(seq_len(nrow(task_grid)), function(i) {
  tr <- task_grid$Trait[i]
  ss <- task_grid$SNP_Set[i]
  !is.null(feature_mats[[ss]]) && !is.null(feature_mats[[ss]][[tr]])
}, logical(1)), , drop = FALSE]

if (nrow(task_grid) == 0) {
  stop("No valid trait x SNP-set tasks found. Check REQUESTED_SNP_SETS and DL_feature_sets.")
}

if (N_WORKERS <= 1L) {
  log_message("PARALLEL: N_WORKERS <= 1, running trait x SNP-set tasks sequentially.")
  for (i in seq_len(nrow(task_grid))) {
    run_one_trait_snp_set(task_grid$Trait[i], task_grid$SNP_Set[i],
                          make_plots_during_run = TRUE)
  }
} else {
  log_message("PARALLEL: Starting PSOCK cluster with ", N_WORKERS,
              " workers for CPU-style trait x SNP-set parallelism.")
  dir.create(worker_root, recursive = TRUE, showWarnings = FALSE)

  cl <- parallel::makeCluster(N_WORKERS, type = "PSOCK",
                              outfile = file.path(OUT_DIR, "results", "parallel_workers.log"))
  on.exit(parallel::stopCluster(cl), add = TRUE)

  parallel::clusterExport(
    cl,
    varlist = c(
      "python_bin", "required_packages",
      "BLAS_THREADS_PER_WORKER", "TF_INTRAOP_THREADS_PER_WORKER",
      "TF_INTEROP_THREADS_PER_WORKER", "TF_FORCE_GPU_ALLOW_GROWTH_VALUE",
      "TF_CPP_MIN_LOG_LEVEL_VALUE", "HIDE_CUDA_FOR_CPU_PARALLEL"
    ),
    envir = environment()
  )

  parallel::clusterEvalQ(cl, {
    Sys.setenv(
      OPENBLAS_NUM_THREADS = BLAS_THREADS_PER_WORKER,
      MKL_NUM_THREADS = BLAS_THREADS_PER_WORKER,
      OMP_NUM_THREADS = BLAS_THREADS_PER_WORKER,
      TF_NUM_INTRAOP_THREADS = TF_INTRAOP_THREADS_PER_WORKER,
      TF_NUM_INTEROP_THREADS = TF_INTEROP_THREADS_PER_WORKER,
      TF_FORCE_GPU_ALLOW_GROWTH = TF_FORCE_GPU_ALLOW_GROWTH_VALUE,
      TF_CPP_MIN_LOG_LEVEL = TF_CPP_MIN_LOG_LEVEL_VALUE
    )
    if (isTRUE(HIDE_CUDA_FOR_CPU_PARALLEL)) {
      Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
    }
    Sys.setenv(RETICULATE_PYTHON = python_bin)
    suppressPackageStartupMessages({
      library(reticulate)
      reticulate::use_python(python_bin, required = TRUE)
      invisible(lapply(required_packages, library, character.only = TRUE))
    })
    NULL
  })

  export_names <- setdiff(ls(envir = environment()), c("cl"))
  parallel::clusterExport(cl, varlist = export_names, envir = environment())

  task_indices <- seq_len(nrow(task_grid))
  worker_results <- parallel::parLapplyLB(cl, task_indices, function(i) {
    tr <- task_grid$Trait[i]
    ss <- task_grid$SNP_Set[i]
    task_out <- file.path(worker_root, paste0(sprintf("%03d", i), "_", safe_name(tr), "_", safe_name(ss)))
    set.seed(900000L + i)
    run_one_trait_snp_set(tr, ss, worker_out_dir = task_out, make_plots_during_run = FALSE)
    TRUE
  })

  aggregate_worker_outputs(worker_root, OUT_DIR)
  log_message("PARALLEL: Worker outputs aggregated into final results directory.")
}

plot_results()
log_message("FINISHED: Feature-reduced DL analysis complete.")
log_message("FINISHED: Outputs written to ", normalizePath(OUT_DIR, winslash = "/"))

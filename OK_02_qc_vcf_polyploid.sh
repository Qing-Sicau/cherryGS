#!/bin/bash

#================================================================================================
#
# VCF Quality Control Pipeline (严格过滤使用)
#
# Description:
#   A flexible, command-line tool for rigorous, multi-stage QC of polyploid VCF data.
#   This version includes checkpoint logic to skip already completed stages, allowing
#   添加中断检测继续运行逻辑
#
# Usage:
#   bash advanced_qc_pipeline.sh <input.vcf.gz> <output_prefix> [OPTIONS]
#   bash advanced_qc_pipeline.sh --help  (for a full list of options)
#
#================================================================================================

# --- Script Setup ---
set -e
set -u
set -o pipefail

# --- Default QC Thresholds (lenient, for pre-imputation QC) ---
SAMPLES_TO_REMOVE_FILE=""
# Genotype-level
MIN_DP_PER_GT=20
MIN_GQ_PER_GT=10
# Site-level
MIN_MEAN_DP_SITE=20
MAX_MEAN_DP_SITE=600
MAX_MISSING_RATE_SITE=0.25
MIN_AF_SITE=0.05
# INFO-level and QUAL
QUAL_THRESH=30.0
MQ_THRESH=40.0
QD_THRESH=2.0
FS_THRESH=100.0
SOR_THRESH=4.0
MQRankSum_THRESH=-12.5
ReadPosRankSum_THRESH=-8.0
# Other options
EXTRACT_INDELS="no"
N_THREADS=$(( $(nproc) / 2 ))

# --- Help Message Function ---
show_help() {
cat << EOF
Usage: ${0##*/} <input.vcf.gz> <output_prefix> [OPTIONS]
Performs a multi-stage QC pipeline on a VCF file. Includes checkpoint logic to skip
already completed steps on re-runs.

Required Arguments:
  input.vcf.gz        Path to the input VCF file.
  output_prefix       A prefix for all output files.

OPTIONS (Defaults are lenient, suitable for pre-imputation filtering):
  --- Stage 0: Sample Filtering ---
  --samples-to-remove FILE  Path to a file with one sample ID per line to remove. [Default: none]
  --- Stage 1: Per-Genotype Filtering (applied to each sample's GT call) ---
  --geno-dp           [Integer] Minimum read depth for a single genotype (FMT/DP). [Default: ${MIN_DP_PER_GT}]
  --geno-gq           [Integer] Minimum quality for a single genotype (FMT/GQ). [Default: ${MIN_GQ_PER_GT}]
  --- Stage 2: Per-Site Mean Depth Filtering (applied to the entire site) ---
  --site-min-mean-dp  [Integer] Minimum average of all samples' FMT/DP for a site. [Default: ${MIN_MEAN_DP_SITE}]
  --site-max-mean-dp  [Integer] Maximum average of all samples' FMT/DP for a site. [Default: ${MAX_MEAN_DP_SITE}]
  --- Stage 3: Other Site-Level Filtering ---
  --site-max-missing  [Float]   Maximum missing rate for a site (F_MISSING). [Default: ${MAX_MISSING_RATE_SITE}]
  --site-min-af       [Float]   Minimum allele frequency for a site (AC/AN). [Default: ${MIN_AF_SITE}]
  --site-qual         [Float]   Minimum QUAL score for a site. [Default: ${QUAL_THRESH}]
  --- Stage 3: INFO Field Filtering (applied to the entire site) ---
  --info-mq-min       [Float]   Minimum Mapping Quality (INFO/MQ). [Default: ${MQ_THRESH}]
  --info-qd-min       [Float]   Minimum Quality by Depth (INFO/QD). [Default: ${QD_THRESH}]
  --info-fs-max       [Float]   Maximum Fisher Strand Bias (INFO/FS). [Default: ${FS_THRESH}]
  --info-sor-max      [Float]   Maximum Strand Odds Ratio (INFO/SOR). [Default: ${SOR_THRESH}]
  --info-mqrs-min     [Float]   Minimum MQRankSum (INFO/MQRankSum). [Default: ${MQRankSum_THRESH}]
  --info-rprs-min     [Float]   Minimum ReadPosRankSum (INFO/ReadPosRankSum). [Default: ${ReadPosRankSum_THRESH}]
  --- Other Options ---
  --extract-indels    [yes|no]  Set to 'yes' to also create a clean Indel file. [Default: ${EXTRACT_INDELS}]
  --threads           [Integer] Number of threads to use. [Default: half of system cores]
  -h, --help                  Display this help message and exit.
EOF
}

# --- Command-Line Argument Parsing ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [ "$#" -lt 2 ]; then echo "ERROR: Missing required arguments." >&2; show_help; exit 1; fi

INPUT_VCF="$1"
OUTPUT_PREFIX="$2"
shift 2

while [ "$#" -gt 0 ]; do
    case "$1" in
        --samples-to-remove) SAMPLES_TO_REMOVE_FILE="$2"; shift 2;;
        --geno-dp)           MIN_DP_PER_GT="$2"; shift 2;;
        --geno-gq)           MIN_GQ_PER_GT="$2"; shift 2;;
        --site-min-mean-dp)  MIN_MEAN_DP_SITE="$2"; shift 2;;
        --site-max-mean-dp)  MAX_MEAN_DP_SITE="$2"; shift 2;;
        --site-max-missing)  MAX_MISSING_RATE_SITE="$2"; shift 2;;
        --site-min-af)       MIN_AF_SITE="$2"; shift 2;;
        --site-qual)         QUAL_THRESH="$2"; shift 2;;
        --info-mq-min)       MQ_THRESH="$2"; shift 2;;
        --info-qd-min)       QD_THRESH="$2"; shift 2;;
        --info-fs-max)       FS_THRESH="$2"; shift 2;;
        --info-sor-max)      SOR_THRESH="$2"; shift 2;;
        --info-mqrs-min)     MQRankSum_THRESH="$2"; shift 2;;
        --info-rprs-min)     ReadPosRankSum_THRESH="$2"; shift 2;;
        --extract-indels)    EXTRACT_INDELS="$2"; shift 2;;
        --threads)           N_THREADS="$2"; shift 2;;
        -h|--help)           show_help; exit 0;;
        *) echo "Unknown option: $1" >&2; show_help; exit 1;;
    esac
done

# --- Post-parsing calculations ---
MAX_AF_SITE=$(echo "1 - ${MIN_AF_SITE}" | bc)

# --- Define Intermediate and Final Filenames ---
STAGE0_VCF="${OUTPUT_PREFIX}.00_samples_removed.vcf.gz"
STAGE1_VCF="${OUTPUT_PREFIX}.01_gt_filtered.vcf.gz"
STAGE2_VCF="${OUTPUT_PREFIX}.02_depth_filtered.vcf.gz"
STAGE3_VCF="${OUTPUT_PREFIX}.03_fully_filtered.vcf.gz"
FINAL_SNP_VCF="${OUTPUT_PREFIX}.final_snps.vcf.gz"
FINAL_INDEL_VCF="${OUTPUT_PREFIX}.final_indels.vcf.gz"

# --- Pipeline Execution ---
echo "======================================================================"
echo "Starting Advanced VCF QC Pipeline (v6 - Checkpoints)"
echo "======================================================================"
echo "Input VCF:                   ${INPUT_VCF}"
echo "Output Prefix:               ${OUTPUT_PREFIX}"
echo "--- Using a maximum of ${N_THREADS} threads ---"
echo "--- Key Thresholds for this run ---"
echo "  Stage 0: Sample Removal File = ${SAMPLES_TO_REMOVE_FILE:-'Not provided'}"
echo "  Stage 1: --geno-dp >= ${MIN_DP_PER_GT}, --geno-gq >= ${MIN_GQ_PER_GT}"
echo "  Stage 2: ${MIN_MEAN_DP_SITE} <= --site-mean-dp <= ${MAX_MEAN_DP_SITE}"
echo "  Stage 3: --site-max-missing <= ${MAX_MISSING_RATE_SITE}, ${MIN_AF_SITE} <= --site-af <= ${MAX_AF_SITE}"
echo "----------------------------------------------------------------------"

# --- Stage 0: Remove specified low-quality samples ---
echo -e "\n>>> Stage 0/5: Removing low-quality samples..."
INPUT_FOR_STAGE1=""
if [[ -z "${SAMPLES_TO_REMOVE_FILE}" ]]; then
    echo "[SKIP] No sample removal file provided. Starting with all samples."
    INPUT_FOR_STAGE1="${INPUT_VCF}"
else
    if [[ -f "${STAGE0_VCF}" && (-f "${STAGE0_VCF}.csi" || -f "${STAGE0_VCF}.tbi") ]]; then
        echo "[SKIP] Found existing file ${STAGE0_VCF}. Using it for the next stage."
    else
        echo "Found sample removal file: ${SAMPLES_TO_REMOVE_FILE}. Removing samples..."
        bcftools view --threads "${N_THREADS}" --samples-file "^${SAMPLES_TO_REMOVE_FILE}" -o "${STAGE0_VCF}" -Oz "${INPUT_VCF}"
        bcftools index --threads "${N_THREADS}" -f "${STAGE0_VCF}"
        echo "Stage 0 Complete. Output: ${STAGE0_VCF}"
    fi
    INPUT_FOR_STAGE1="${STAGE0_VCF}"
fi
echo "----------------------------------------------------------------------"

# --- Stage 1: Per-Genotype Filtering ---
echo -e "\n>>> Stage 1/5: Setting low-quality individual genotypes to missing..."
if [[ -f "${STAGE1_VCF}" && (-f "${STAGE1_VCF}.csi" || -f "${STAGE1_VCF}.tbi") ]]; then
    echo "[SKIP] Found existing file ${STAGE1_VCF}. Using it for the next stage."
else
    bcftools filter --threads "${N_THREADS}" -S . -e "FMT/DP < ${MIN_DP_PER_GT} || FMT/GQ < ${MIN_GQ_PER_GT}" --set-GTs . -o "${STAGE1_VCF}" -Oz "${INPUT_FOR_STAGE1}"
    bcftools index --threads "${N_THREADS}" -f "${STAGE1_VCF}"
    echo "Stage 1 Complete. Output: ${STAGE1_VCF}"
fi
echo "----------------------------------------------------------------------"

# --- Stage 2: Per-Site Mean Depth Filtering (Parallelized) ---
echo -e "\n>>> Stage 2/5: Filtering sites based on mean sample depth (Parallelized)..."
if [[ -f "${STAGE2_VCF}" && (-f "${STAGE2_VCF}.csi" || -f "${STAGE2_VCF}.tbi") ]]; then
    echo "[SKIP] Found existing file ${STAGE2_VCF}. Using it for the next stage."
else
    STAGE2_MEAN_DEPTH_BED="${OUTPUT_PREFIX}.02_mean_depth.bed"
    STAGE2_SITES_TO_KEEP_BED="${OUTPUT_PREFIX}.02_sites_to_keep.bed"
    if [[ -f "${STAGE2_MEAN_DEPTH_BED}" && -f "${STAGE2_SITES_TO_KEEP_BED}" ]]; then
        echo "[SKIP] Found existing file ${STAGE2_MEAN_DEPTH_BED}. Using it for the next step."
    else
        echo "    (2a) Calculating mean depth..."
        bcftools query -f '%CHROM\t%POS[\t%DP]\n' "${STAGE1_VCF}" | awk 'BEGIN{OFS="\t"} {sum=0; count=0; for (i=3; i<=NF; i++) { if ($i != "." && $i > 0) { sum += $i; count++; } } if (count > 0) { avg = sum / count; print $1, $2-1, $2, avg; }}' > "${STAGE2_MEAN_DEPTH_BED}"
        echo "    (2b) Identifying sites within range..."
        awk -v min="${MIN_MEAN_DP_SITE}" -v max="${MAX_MEAN_DP_SITE}" '$4 >= min && $4 <= max' "${STAGE2_MEAN_DEPTH_BED}" > "${STAGE2_SITES_TO_KEEP_BED}"
        if [ ! -s "${STAGE2_SITES_TO_KEEP_BED}" ]; then echo "ERROR: No sites passed Stage 2. Check thresholds." >&2; exit 1; fi
    fi
    echo "    (2c) Splitting site list for parallel processing..."
    TMP_DIR="${OUTPUT_PREFIX}.tmp_stage2_parallel"
    mkdir -p "${TMP_DIR}"
    TOTAL_SITES=$(wc -l < "${STAGE2_SITES_TO_KEEP_BED}")
    LINES_PER_SPLIT=$(( (TOTAL_SITES + N_THREADS - 1) / N_THREADS ))
    split -l "${LINES_PER_SPLIT}" "${STAGE2_SITES_TO_KEEP_BED}" "${TMP_DIR}/sites_split."
    echo "     -> Split into $(ls -1q "${TMP_DIR}"/sites_split.* | wc -l) files."
    echo "    (2d) Filtering VCF in parallel across ${N_THREADS} jobs..."
    VCF_CHUNK_LIST="${TMP_DIR}/vcf_chunks.txt"
    > "${VCF_CHUNK_LIST}"
    for split_bed_chunk in "${TMP_DIR}"/sites_split.*; do
        (
            chunk_name=$(basename "${split_bed_chunk}")
            output_vcf_chunk="${TMP_DIR}/${chunk_name}.vcf.gz"
            bcftools view --threads 2 -R "${split_bed_chunk}" -o "${output_vcf_chunk}" -Oz "${STAGE1_VCF}"
            bcftools index --threads 2 "${output_vcf_chunk}"
            echo "${output_vcf_chunk}" >> "${VCF_CHUNK_LIST}"
        ) &
    done
    wait
    echo "    (2e) Merging parallel VCF chunks..."
    sort -o "${VCF_CHUNK_LIST}" "${VCF_CHUNK_LIST}"
    bcftools concat -a --threads "${N_THREADS}" -f "${VCF_CHUNK_LIST}" -o "${STAGE2_VCF}" -Oz
    bcftools index --threads "${N_THREADS}" -f "${STAGE2_VCF}"
    echo "    (2f) Cleaning up temporary files..."
    rm -rf "${TMP_DIR}"
    rm -f "${STAGE2_MEAN_DEPTH_BED}" "${STAGE2_SITES_TO_KEEP_BED}"
    echo "Stage 2 Complete. Output: ${STAGE2_VCF}"
fi
echo "----------------------------------------------------------------------"

# --- Stage 3: Final Site-Level Filtering ---
echo -e "\n>>> Stage 3/5: Applying final site-level filters..."
if [[ -f "${STAGE3_VCF}" && (-f "${STAGE3_VCF}.csi" || -f "${STAGE3_VCF}.tbi") ]]; then
    echo "[SKIP] Found existing file ${STAGE3_VCF}. Using it for the next stage."
else
    FILTER_EXPRESSION="F_MISSING > ${MAX_MISSING_RATE_SITE} || (AC/AN < ${MIN_AF_SITE} || AC/AN > ${MAX_AF_SITE}) || QUAL < ${QUAL_THRESH} || INFO/MQ < ${MQ_THRESH} || INFO/QD < ${QD_THRESH} || INFO/FS > ${FS_THRESH} || INFO/SOR > ${SOR_THRESH} || INFO/MQRankSum < ${MQRankSum_THRESH} || INFO/ReadPosRankSum < ${ReadPosRankSum_THRESH}"
    bcftools filter --threads "${N_THREADS}" --exclude "${FILTER_EXPRESSION}" -o "${STAGE3_VCF}" -Oz "${STAGE2_VCF}"
    bcftools index --threads "${N_THREADS}" -f "${STAGE3_VCF}"
    echo "Stage 3 Complete. High-quality unified VCF: ${STAGE3_VCF}"
fi
echo "----------------------------------------------------------------------"

# --- Stage 4: Final Biallelic SNP Extraction ---
echo -e "\n>>> Stage 4/5: Extracting final analysis-ready variant files..."
if [[ -f "${FINAL_SNP_VCF}" && (-f "${FINAL_SNP_VCF}.csi" || -f "${FINAL_SNP_VCF}.tbi") ]]; then
    echo "[SKIP] Found existing file ${FINAL_SNP_VCF}."
else
    echo "    (4a) Extracting biallelic SNPs..."
    bcftools view --threads "${N_THREADS}" -v snps -m2 -M2 -o "${FINAL_SNP_VCF}" -Oz "${STAGE3_VCF}"
    bcftools index --threads "${N_THREADS}" -f "${FINAL_SNP_VCF}"
    echo "     -> Analysis-ready SNP file created: ${FINAL_SNP_VCF}"
fi

# --- Stage 5: (Optional) Indel Extraction ---
if [[ "${EXTRACT_INDELS}" == "yes" ]]; then
    echo -e "\n>>> Stage 5/5: Extracting Indels as requested..."
    if [[ -f "${FINAL_INDEL_VCF}" && (-f "${FINAL_INDEL_VCF}.csi" || -f "${FINAL_INDEL_VCF}.tbi") ]]; then
        echo "[SKIP] Found existing file ${FINAL_INDEL_VCF}."
    else
        bcftools view --threads "${N_THREADS}" -v indels -o "${FINAL_INDEL_VCF}" -Oz "${STAGE3_VCF}"
        bcftools index --threads "${N_THREADS}" -f "${FINAL_INDEL_VCF}"
        echo "     -> Filtered Indel file created: ${FINAL_INDEL_VCF}"
    fi
else
    echo -e "\n[SKIP] Stage 5: Skipping Indel extraction as per configuration."
fi
echo ""
echo "======================================================================"
echo "PIPELINE FINISHED SUCCESSFULLY!"
echo ""
echo "Your final, analysis-ready file for downstream analysis is:"
echo "==> ${FINAL_SNP_VCF}"
echo ""
echo "All intermediate files have been kept in the '${OUTPUT_PREFIX}.*' pattern."
echo "======================================================================"
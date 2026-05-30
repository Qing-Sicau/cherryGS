#!/bin/bash
# Set parameters
Project="gatc"
dataDir="./reseq.data"
config="$dataDir/data_info/sample.list"
refDir="$dataDir/reference/"
ref="$dataDir/reference/genome.fasta"
NCPUS=20
export start_step=1
start_sample=1
stop_sample=399
max_jobs=20
job_count=0
log_file="time_${start_sample}_${stop_sample}_log.txt"
proc_file="proc_${start_sample}_${stop_sample}_log.txt"
echo "Process Log" > "$log_file"

# samtools faidx $ref
# picard CreateSequenceDictionary \
# 	R="$ref" \
#   O="$refDir/genome.dict"

# Function to process each sample
process_sample() {
    echo "Processing sample: $1, starting from step: $start_step" >> "$log_file"
    local index=$1
    group=$(awk -v taskID="$index" '$1==taskID {print $2}' "$config")
    sample=$(awk -v taskID="$index" '$1==taskID {print $3}' "$config")
    platform=$(awk -v taskID="$index" '$1==taskID {print $4}' "$config")
    seq_centre=$(awk -v taskID="$index" '$1==taskID {print $5}' "$config")
    fq1=$(awk -v taskID="$index" '$1==taskID {print $6}' "$config")
    fq2=$(awk -v taskID="$index" '$1==taskID {print $7}' "$config")

    if [ -z "$sample" ] || [ -z "$fq1" ] || [ -z "$fq2" ]; then
        echo "No data for sample $sample. Skipping..." >> "$log_file"
        return
    fi

    flowcell=$(zcat "$dataDir/samples/$sample/$fq1" | head -1 | cut -c 2-11)
    lane=$(zcat "$dataDir/samples/$sample/$fq1" | head -1 | cut -c 13)
    bams="$dataDir/bams/$group"
    fastq="$dataDir/samples/$sample"
    library=1

    # Check if the directory exists
    if [ ! -d "$bams" ]; then
        mkdir -p "$bams"
        echo "Created directory $bams" >> "$log_file"
    else
        echo "Directory $bams already exists" >> "$log_file"
    fi

    if [ "$start_step" -le 1 ]; then
        # Step 1: Alignment
        start_time=$(date +%s)
        echo "Starting alignment for $sample" >> "$log_file"
        
	bwa mem -M -t $NCPUS "$ref" \
        -R "@RG\tID:${flowcell}.${lane}_${sample}_${library}\tPL:${platform}\tPU:${flowcell}.${lane}\tSM:${sample}\tLB:${sample}_${library}\tCN:${seq_centre}" \
        "$fastq/$fq1" "$fastq/$fq2" \
        | samblaster -M -e --addMateTags \
        -d "${bams}/${sample}_disc.sam" \
        -s "${bams}/${sample}_split.sam" \
        | samtools view -@ ${NCPUS} -bSho "${bams}/${sample}_dedup.bam"
        if [ $? -ne 0 ]; then
            echo "Alignment failed for $sample" 
            return
        fi
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Alignment for $sample completed in $duration seconds" 

        # Step 2: Sort and index bam
        start_time=$(date +%s)
        echo "Starting sort and indexing for $sample" 
        samtools view -@ ${NCPUS} -h "${bams}/${sample}_dedup.bam" \
        | samtools sort -@ ${NCPUS} -T "${bams}/temp" -m 4G -o "${bams}/${sample}_dedup_sorted.bam"
        if [ $? -ne 0 ]; then
            echo "Sorting failed for $sample"
            return
        fi

        if ! samtools quickcheck "${bams}/${sample}_dedup_sorted.bam"; then
            echo "Corrupted or missing ${bams}/${sample}_dedup_sorted.bam"
            return
        fi

        samtools index -@ ${NCPUS} "${bams}/${sample}_dedup_sorted.bam" "${bams}/${sample}_dedup_sorted.bai" 
        
        if [ $? -eq 0 ]; then
             rm "${bams}/${sample}_dedup.bam" "${bams}/${sample}_disc.sam" "${bams}/${sample}_split.sam"
        else
             echo "Samtools index failed for ${sample}, keeping intermediate files." >> "$log_file"
        fi
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Sort and Index for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 3 ]; then
        # Step 3: Collect alignment and insert size metrics
        start_time=$(date +%s)
        echo "Starting metrics collection for $sample" >> "$log_file"
        
        picard CollectAlignmentSummaryMetrics \
        R="$ref" \
        I="${bams}/${sample}_dedup_sorted.bam" \
        O="${bams}/${sample}_alignment_metrics.txt"
        
        picard CollectInsertSizeMetrics \
        I="${bams}/${sample}_dedup_sorted.bam" \
        O="${bams}/${sample}_insert_metrics.txt" \
        H="${bams}/${sample}_insert_size_histogram.pdf"
        
        samtools depth -a "${bams}/${sample}_dedup_sorted.bam" > "${bams}/${sample}_depth_out.txt"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Metrics collection for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 4 ]; then
        # Step 4: Call variants
        start_time=$(date +%s)
        echo "Starting variant calling for $sample" >> "$log_file"
        gatk HaplotypeCaller \
        -R "$ref" \
        -ploidy 4 \
        -I "${bams}/${sample}_dedup_sorted.bam" \
        -O "${bams}/${sample}_raw_variants.vcf"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Variant calling for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 5 ]; then
        # Step 5: Extract SNPs and INDELs
        start_time=$(date +%s)
        echo "Starting SNPs and INDELs extraction for $sample" >> "$log_file"
        gatk SelectVariants \
        -R "$ref" \
        -V "${bams}/${sample}_raw_variants.vcf" \
        -select-type SNP \
        -O "${bams}/${sample}_raw_snps.vcf"
        
        gatk SelectVariants \
        -R "$ref" \
        -V "${bams}/${sample}_raw_variants.vcf" \
        -select-type INDEL \
        -O "${bams}/${sample}_raw_indels.vcf"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "SNPs and INDELs extraction for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 6 ]; then
        # Step 6: Filter SNPs
        start_time=$(date +%s)
        echo "Starting SNP filtering for $sample" >> "$log_file"
        gatk VariantFiltration \
        -R "$ref" \
        -V "${bams}/${sample}_raw_snps.vcf" \
        -O "${bams}/${sample}_filtered_snps.vcf" \
        -filter-name "QD_filter" -filter "QD < 2.0" \
        -filter-name "FS_filter" -filter "FS > 60.0" \
        -filter-name "MQ_filter" -filter "MQ < 40.0" \
        -filter-name "SOR_filter" -filter "SOR > 4.0" \
        -filter-name "MQRankSum_filter" -filter "MQRankSum < -12.5" \
        -filter-name "ReadPosRankSum_filter" -filter "ReadPosRankSum < -8.0"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "SNP filtering for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 7 ]; then
        # Step 7: Filter INDELs
        start_time=$(date +%s)
        echo "Starting INDEL filtering for $sample" >> "$log_file"
        gatk VariantFiltration \
        -R "$ref" \
        -V "${bams}/${sample}_raw_indels.vcf" \
        -O "${bams}/${sample}_filtered_indels.vcf" \
        -filter-name "QD_filter" -filter "QD < 2.0" \
        -filter-name "FS_filter" -filter "FS > 200.0" \
        -filter-name "SOR_filter" -filter "SOR > 10.0"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "INDEL filtering for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 8 ]; then
        # Step 8: Filter variants
        start_time=$(date +%s)
        echo "Starting variant selection for $sample" >> "$log_file"
        
        # Select filtered SNPs and INDELs
        gatk SelectVariants \
        --exclude-filtered \
        -V "${bams}/${sample}_filtered_snps.vcf" \
        -O "${bams}/${sample}_bqsr_snps.vcf"
        
        gatk SelectVariants \
        --exclude-filtered \
        -V "${bams}/${sample}_filtered_indels.vcf" \
        -O "${bams}/${sample}_bqsr_indels.vcf"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Variant selection for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 9 ]; then
        # Step 9: Base Quality Score Recalibration (BQSR)
        start_time=$(date +%s)
        echo "Starting BQSR for $sample" >> "$log_file"
        
        gatk BaseRecalibrator \
        -R "$ref" \
        -I "${bams}/${sample}_dedup_sorted.bam" \
        --known-sites "${bams}/${sample}_bqsr_snps.vcf" \
        --known-sites "${bams}/${sample}_bqsr_indels.vcf" \
        -O "${bams}/${sample}_recal_data.table"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "BQSR for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 10 ]; then
        # Step 10: Apply BQSR
        start_time=$(date +%s)
        echo "Starting ApplyBQSR for $sample" >> "$log_file"
        
        gatk ApplyBQSR \
        	-R "$ref" \
        	-I "${bams}/${sample}_dedup_sorted.bam" \
        	-bqsr "${bams}/${sample}_recal_data.table" \
        	-O "${bams}/${sample}_recal_reads.bam"
        
        samtools index -@ $NCPUS "${bams}/${sample}_recal_reads.bam"
        if [ $? -eq 0 ]; then
        	echo "Samtools index OK for ${sample}_recal_reads.bam" >> "$log_file"
        else
		echo "samtools index failed for ${sample}_recal_reads.bam" >> "$log_file"
	fi
	
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "ApplyBQSR for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 11 ]; then
        # Step 11: Base Quality Score Recalibration (BQSR) 2nd time
        start_time=$(date +%s)
        echo "Starting second BQSR for $sample" >> "$log_file"
        
        gatk BaseRecalibrator \
        -R "$ref" \
        -I "${bams}/${sample}_recal_reads.bam" \
        --known-sites "${bams}/${sample}_bqsr_snps.vcf" \
        --known-sites "${bams}/${sample}_bqsr_indels.vcf" \
        -O "${bams}/${sample}_post_recal_data.table"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Second BQSR for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 12 ]; then
        # Step 12: Analyze covariates
        start_time=$(date +%s)
        echo "Starting AnalyzeCovariates for $sample" >> "$log_file"
        
        gatk AnalyzeCovariates \
        -before "${bams}/${sample}_recal_data.table" \
        -after "${bams}/${sample}_post_recal_data.table" \
        -plots "${bams}/${sample}_recalibration_plots.pdf"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "AnalyzeCovariates for $sample completed in $duration seconds" >> "$log_file"
    fi

    if [ "$start_step" -le 13 ]; then
        # Step 13: Call variants with recalibrated BAM
        start_time=$(date +%s)
        echo "Starting final variant calling for $sample" >> "$log_file"
        
        gatk HaplotypeCaller \
        -R "$ref" \
        -I "${bams}/${sample}_recal_reads.bam" \
        -ERC GVCF \
        -ploidy 4 \
        -O "${bams}/${sample}_raw_variants_recal.g.vcf"
        end_time=$(date +%s)
		duration=$((end_time - start_time))
        echo "Final variant calling for $sample completed in $duration seconds" >> "$log_file"
    fi
}

# Main loop to process each sample
export -f process_sample
for i in $(seq $start_sample $stop_sample); do
    process_sample $i 2>> $proc_file &
    ((job_count++))
    # Check if the max jobs are running
    if [[ $job_count -ge $max_jobs ]]; then
        wait -n  # Wait for any job to finish
        ((job_count--))
    fi
done

echo "All samples processed."

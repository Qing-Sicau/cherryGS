# ## 简介
# 本文档详细记录了使用R包 `polymapR` 进行四倍体连锁图谱构建的完整流程。流程主要包括：数据导入与质控、利用单剂量标记构建双亲各自的连锁图谱骨架、处理连锁群断裂与合并、利用桥梁标记整合双亲图谱、计算遗传距离并输出结果。
# ## 阶段一：数据加载与预处理 (Data Loading and Pre-processing)
# 在构建图谱之前，必须对原始数据进行严格的质量控制，以剔除低质量的标记和个体，保证后续分析的准确性。


# 加载所需R包
library(polymapR)
library(hexbin)
library(igraph)
library(dplyr)

# 1. ---- 读取数据 ----
# 加载包内自带的四倍体马铃薯F1代群体标记剂量数据
ALL_dosage <- read.csv("hn_hn_gq5dp5.dosage.tsv", header = T, sep = "\t")

ALL_dosages <- ALL_dosage %>%
  dplyr::mutate(snp = paste0(CHROM, ":", POS)) %>%
  dplyr::select(snp, HF1, NZH2, starts_with("HN"), starts_with("NH")) %>%
  dplyr::filter(HF1 != "NA" & NZH2 != "NA") 

#ALL_dosages <- readRDS("hn_nh_input_44384_for_polyrad_polymapR.rds")
  
P1="HF1"
P2="NZH2"

head(ALL_dosages[,1:10])



# 2. ---- 数据初步质控 ----
# 2.1 检查F1代个体
# 目的：通过与亲本比较，检查F1群体中是否存在异常个体（如错误标记、远缘个体或自交个体）。
# polysomic = TRUE 表明这是一个同源四倍体。
F1checked <- checkF1(dosage_matrix = ALL_dosages,
                     parent1 = P1, parent2 = P2,
                     F1 = colnames(ALL_dosages)[3:ncol(ALL_dosages)],
                     polysomic = TRUE,
                     disomic = FALSE, 
                     mixed = FALSE, 
                     ploidy = 4, fracNA_threshold = 0.2)

head(F1checked$checked_F1) # 查看检查后的F1数据

# 2.2 主成分分析(PCA)可视化群体结构
# 目的：通过降维直观地查看群体结构，识别异常或离群的个体。
PCA_progeny(dosage_matrix = ALL_dosages, 
            highlight = list(c(P1, P2)), 
            colors = "red")

# 2.3 生成标记数据的整体摘要
# 目的：全面评估标记质量，包括偏分离检验、亲本剂量、缺失率等。
# progeny_incompat_cutoff: 允许的与亲本基因型不兼容的后代比例阈值。
mds <- marker_data_summary(dosage_matrix = ALL_dosages,
                           ploidy = 4,
                           pairing = "random", # 假设是随机配对
                           parent1 = P1,
                           parent2 = P2,
                           progeny_incompat_cutoff = 0.05)

# 2.4 查看亲本标记剂量分布
# 目的：了解亲本中不同剂量类型标记的数量，对后续连锁分析策略提供参考。
pq_before_convert <- parental_quantities(dosage_matrix = ALL_dosages, las = 2,
                                         parent1 = P1, parent2 = P2)

# 3. ---- 数据转换与过滤 ----
# 3.1 转换标记剂量为分离类型
# 核心步骤：将绝对剂量（如0,1,2,3,4）转换为分离类型（如1_0, 1_1, 2_0等）。
# 这是连锁分析的基础，因为连锁计算是基于特定分离类型的。
segregating_data <- convert_marker_dosages(dosage_matrix = ALL_dosages, ploidy = 4,
                                           parent1 = P1, parent2 = P2)

pq_after_convert <- parental_quantities(dosage_matrix = segregating_data,
                                        parent1 = P1, parent2 = P2) # 转换后再次查看

# 3.2 严格的质量过滤
# 3.2.1 按标记过滤：剔除缺失率过高的标记
screened_data <- screen_for_NA_values(dosage_matrix = segregating_data, parentnames = c(P1, P2),
                                      margin = 1,         # margin=1 表示对行(marker)进行操作
                                      cutoff =  0.10,     # 缺失率超过10%的标记将被移除
                                      print.removed = FALSE)

# 3.2.2 按个体过滤：剔除缺失率过高的个体

screened_data2 <- screen_for_NA_values(dosage_matrix = screened_data, parentnames = c(P1, P2),
                                       cutoff = 0.1, 
                                       margin = 2,         # margin=2 表示对列(individual)进行操作
                                       print.removed = FALSE)

# 3.2.3 过滤重复个体
# 目的：通过计算个体间的相关性，移除遗传上高度相似（可能是克隆或错误取样）的个体。
screened_data3 <- screen_for_duplicate_individuals(dosage_matrix = screened_data2, 
                                                   cutoff = 0.95,      # 相关性大于95%的被认为是重复
                                                   plot_cor = TRUE)

# 3.2.4 过滤重复标记 (Binning)
# 目的：将遗传行为完全相同（共分离）的标记放入同一个"bin"中，并只保留一个代表性标记进行后续计算。
# 这样做可以大大降低计算复杂度和冗余。被移除的标记信息会保存下来，在最后一步加回到图谱中。
screened_data4 <- screen_for_duplicate_markers(dosage_matrix = screened_data3)

# 过滤后的最终高质量数据集
filtered_data <- screened_data4$filtered_dosage_matrix 
pq_screened_data <- parental_quantities(dosage_matrix = filtered_data,
                                        parent1 = P1, parent2 = P2)



## 阶段二：构建双亲连锁图谱骨架 (Building Parental Map Skeletons)

## 核心策略是分别构建P1和P2的图谱，然后再进行整合。构建骨架时，信息量最丰富、最易于分相的 Simplex x ## Nulliplex (1x0) 标记是主要决定因素。

# 3. ---- 标记聚类：构建同源组(Homologues)和连锁群(LG) ----
# 逻辑：连锁分析的聚类是两步过程。
# 第一步：用较高的LOD阈值将紧密连锁的标记聚成小簇，这些簇理论上代表了单条“同源染色单体”(Homologue)。
# 第二步：用较低的LOD阈值将这些“同源染色单体”簇进一步聚合，形成完整的“染色体”(Linkage Group)。
# 对于一个5条染色体的四倍体，理论上应有 5 (LG) x 4 (Ploidy) = 20 个同源染色单体簇。

# 推荐策略：先定义同源组，再定义连锁群 (Vignette推荐方法)
# 3.1. 仅用Coupling相数据，在高LOD下初步聚类，目的是识别同源组。
# 为什么用Coupling相？因为相引连锁的标记对理论上位于同一条同源染色单体上。
# 因此，高LOD下的Coupling连锁是定义同源染色单体的最可靠依据。

SN_SN_P1_coupl <- SN_SN_P1[SN_SN_P1$phase == "coupling",] 

# 通过尝试一系列LOD值，找到一个能产生约20个簇 (5 LG x 4 ploidy) 的最佳LOD。
P1_homologues_1 <- cluster_SN_markers(linkage_df = SN_SN_P1_coupl, 
                                      LOD_sequence = seq(3,20,1), # 探索3-12的LOD值
                                      LG_number = 8, # 预期染色体数量
                                      ploidy = 4,
                                      parentname = P1,
                                      plot_network = F,
                                      plot_clust_size = F, 
                                      min_clust_size = 3) # 可视化不同LOD下的聚类数

# 从图中或结果中可以看到，LOD=5时，可以得到20个左右的cluster，符合预期。
head(P1_homologues_1[["17"]]) # 查看LOD=6时的聚类结果
table(t(P1_homologues_1[["17"]]$cluster))

# 3.2. 利用桥梁标记(Duplex x Nulliplex)将同源组聚合成连锁群。
# 首先计算 SN vs DN 标记对的连锁。DN标记可以同时与多个同源组上的SN标记连锁，起到桥梁作用。
SN_DN_P1 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    markertype2 = c(2,0),
                    which_parent = 1,
                    ploidy = 4,
                    pairing = "random", parent1 = P1, parent2 = P2)

# 使用bridgeHomologues函数，将前面LOD=6得到的20个同源组簇，聚合成5个连锁群(LG)。
LGHomDf_P1 <- bridgeHomologues(cluster_stack = P1_homologues[["17"]], 
                               linkage_df = SN_DN_P1, 
                               LOD_threshold = 5, # 定义桥接的LOD阈值
                               automatic_clustering = T, # 自动聚类
                               LG_number = 8,
                               parentname = P1, 
                               min_links = 2, 
                               min_bridges = 2)

# 查看P1图谱骨架的最终结构：5个连锁群，每个群包含4个同源组。
table(LGHomDf_P1$LG, LGHomDf_P1$homologue)

# 如何修复？
# 方法一：可视化诊断 (overviewSNlinks)
# 目的：绘制出LG3内部，所有同源组两两之间的连锁关系图。
# 我们要寻找的信号是：某两个片段之间存在强烈的、密集的绿色(Coupling)连线，这表明它们本应是一体的。
# 注意软件版本更新，也可能是plot_SNlinks_phasing函数
overviewSNlinks(linkage_df = SN_SN_P1,
                     LG_hom_stack = LGHomDf_P1,
                     LG = 2,
                     LOD_threshold = 0)

# 从图中可以清晰地看到，片段4和片段5之间存在强烈的绿色信号，说明它们应该被合并。
# Vignette中的函数名为 overviewSNlinks，较新版本中可能是 plot_SNlinks_phasing
# 确认函数名并执行后，根据可视化结果进行合并
LGHomDf_P1 <- merge_homologues(LG_hom_stack = LGHomDf_P1,
                                 ploidy = 4,
                                 LG = 2,
                                 mergeList = list(c(3,5))) # 合并homologue 4和5
# 再次查看
table(LGHomDf_P1$LG, LGHomDf_P1$homologue)
# 再解决6号连锁群
overviewSNlinks(linkage_df = SN_SN_P1,
                     LG_hom_stack = LGHomDf_P1,
                     LG = 6,
                     LOD_threshold = 0) 

LGHomDf_P1 <- merge_homologues(LG_hom_stack = LGHomDf_P1,
                                 ploidy = 4,
                                 LG = 6,
                                 mergeList = list(c(4,6))) # 合并homologue 4和6

overviewSNlinks(linkage_df = SN_SN_P1,
                     LG_hom_stack = LGHomDf_P1,
                     LG = 6,
                     LOD_threshold = 0) 

LGHomDf_P1 <- merge_homologues(LG_hom_stack = LGHomDf_P1,
                                 ploidy = 4,
                                 LG = 6,
                                 mergeList = list(c(2,3))) # 合并homologue 2和3

### 2.2 构建亲本P2的图谱骨架
#对亲本P2重复同样的操作。
# 1. ---- 计算P2的 1x0 vs 1x0 连锁 ----

SN_SN_P2 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    parent1 = P1,
                    parent2 = P2,
                    which_parent = 2,
                    ploidy = 4,
                    pairing = "random", 
                    ncores = 1)


# 方法一，直接用snsn建立染色体

P2_homologues <- cluster_SN_markers(linkage_df = SN_SN_P2, 
                                    LOD_sequence = seq(3, 30, 1), 
                                    LG_number = 12,
                                    ploidy = 4,
                                    parentname = "P2",
                                    plot_network = FALSE,
                                    plot_clust_size = T)

P1_hom_LOD3 <- P1_homologues[["3"]]
t <- table(P1_hom_LOD3$cluster)
print(paste("Number of clusters:",length(t)))
t





# 2. ---- 仅用Coupling相数据在高LOD下聚类识别同源组 ----
SN_SN_P2_coupl <- SN_SN_P2[SN_SN_P2$phase == "coupling",]
P2_homologues <- cluster_SN_markers(linkage_df = SN_SN_P2_coupl, 
                                      LOD_sequence = c(3:12), 
                                      LG_number = 12,
                                      ploidy = 4,
                                      parentname = "P2",
                                      plot_network = F,
                                      plot_clust_size = TRUE)
# 从图中可以看到，LOD=4 可以得到约20个cluster。

# 3. ---- 计算 SN vs DN 连锁，用于桥接 ----
SN_DN_P2 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    markertype2 = c(2,0),
                    which_parent = 2,
                    ploidy = 4,
                    pairing = "random")

# 4. ---- 桥接同源组，构建连锁群 ----
# 这里使用LOD=4的结果来构建P2的图谱骨架
LGHomDf_P2 <- bridgeHomologues(cluster_stack = P2_homologues[["4"]], 
                                     linkage_df = SN_DN_P2, 
                                     LOD_threshold = 4, 
                                     automatic_clustering = TRUE, 
                                     LG_number = 5,
                                     parentname = "P2")

table(LGHomDf_P2$LG, LGHomDf_P2$homologue)


## 阶段三：连锁群的修正与优化 (Refining Linkage Groups)

在自动构建过程中，时常会出现连锁群断裂（一个同源组被分成几段）或错误合并。这通常表现为某个连锁群中的同源组数量大于倍性数。`polymapR`提供了强大的工具来诊断和修正这些问题。


# 假设我们用了一个不同的LOD值(e.g., 6)来构建P2图谱，并发现了一个问题。
# 这部分代码是Vignette中的一个演示实例，用于说明如何处理问题。
LGHomDf_P2_problematic <- bridgeHomologues(cluster_stack = P2_homologues[["6"]], # 使用LOD 6的结果
                                     linkage_df = SN_DN_P2, 
                                     LOD_threshold = 4, 
                                     automatic_clustering = TRUE, 
                                     LG_number = 5,
                                     parentname = "P2")

# 检查图谱结构，发现LG3有5个同源组(3, 4, 5, 16, 20)，多于四倍体的4个。
# 这说明LG3中至少有一个同源组是断裂的。
table(LGHomDf_P2_problematic$LG, LGHomDf_P2_problematic$homologue)

# 如何修复？
# 方法一：可视化诊断 (overviewSNlinks)
# 目的：绘制出LG3内部，所有同源组两两之间的连锁关系图。
# 我们要寻找的信号是：某两个片段之间存在强烈的、密集的绿色(Coupling)连线，这表明它们本应是一体的。
# 注意软件版本更新，也可能是plot_SNlinks_phasing函数
overviewSNlinks(linkage_df = SN_SN_P1,
                     LG_hom_stack = LGHomDf_P1,
                     LG = 9,
                     LOD_threshold = 0)

# 从图中可以清晰地看到，片段4和片段5之间存在强烈的绿色信号，说明它们应该被合并。
# Vignette中的函数名为 overviewSNlinks，较新版本中可能是 plot_SNlinks_phasing
# 确认函数名并执行后，根据可视化结果进行合并
LGHomDf_P1 <- merge_homologues(LG_hom_stack = LGHomDf_P1,
                                 ploidy = 4,
                                 LG = 9,
                                 mergeList = list(c(4,5))) # 合并homologue 4和5

# 方法二：算法再聚类 (cluster_per_LG)
# 目的：对有问题的连锁群(LG3)内的所有标记，进行强制性的重新聚类，使其聚成指定的簇数(nclust_out = 4)。
# 使用不同LOD值，判断成簇数量是否符合预期，以确定LOD
cluster_per_LG(LG = 9,
               linkage_df = SN_SN_P1[SN_SN_P1$phase == "coupling",], 
               LG_hom_stack =LGHomDf_P1,  
               LOD_sequence = c(2:10), # The first element is used for network layout
               modify_LG_hom_stack = F, # 确定前，此选项为F，确定后修改为T
               network.layout = "stacked", # circled or stacked
               nclust_out = 4,
               label.offset=1.2)



# 更新连锁群分组信息
LGHomDf_P1 <- cluster_per_LG(LG = 8, 
                                     linkage_df = SN_SN_P1[SN_SN_P1$phase == "coupling",], 
                                     LG_hom_stack = LGHomDf_P1, 
                                     LOD_sequence = 3.2, # 改为筛选好的LOD
                                     modify_LG_hom_stack = T, # 设为TRUE来直接返回修正后的完整数据框
                                     nclust_out = 4,  # 强制聚为4簇
                                     network.layout = "stacked") 


# 检查修正后的结果
table(LGHomDf_P1$LG, LGHomDf_P1$homologue)





## 阶段四：整合双亲图谱 (Integrating Parental Maps)

利用在双亲中都分离的标记（主要是 Simplex-Simplex, SxS, 1x1）作为“锚点”，将P1和P2的图谱对应起来，形成统一的整合图谱。


# 在这部分，我们使用之前没有问题的P2图谱(LGHomDf_P2)进行后续操作。
LGHomDf_P2_1 <- LGHomDf_P2_1_alternative # 为保持变量名一致性，使用修正后的命名

# 1. ---- 将SxS标记分配到P1图谱 ----
# 首先计算 SN vs SS 的连锁
SN_SS_P1 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    markertype2 = c(1,1),
                    which_parent = 1,
                    ploidy = 4,
                    pairing = "random")
# 然后将SS标记分配到P1的连锁群和同源组上
P1_SxS_Assigned <- assign_linkage_group(linkage_df = SN_SS_P1,
                                        LG_hom_stack = LGHomDf_P1,
                                        SN_colname = "marker_a",
                                        unassigned_marker_name = "marker_b",
                                        phase_considered = "coupling",
                                        LG_number = 5,
                                        LOD_threshold = 4,
                                        ploidy = 4)

# 2. ---- 将SxS标记分配到P2图谱 ----
# 同样的操作用于P2
SN_SS_P2 <- linkage(dosage_matrix = filtered_data, 
                    markertype1 = c(1,0),
                    markertype2 = c(1,1),
                    which_parent = 2,
                    ploidy = 4,
                    pairing = "random")

P2_SxS_Assigned <- assign_linkage_group(linkage_df = SN_SS_P2,
                                        LG_hom_stack = LGHomDf_P2_1,
                                        SN_colname = "marker_a",
                                        unassigned_marker_name = "marker_b",
                                        phase_considered = "coupling",
                                        LG_number = 5,
                                        LOD_threshold = 4,
                                        ploidy = 4)

# 3. ---- 统一双亲的连锁群命名 ----
# 核心步骤：SxS标记是连接P1和P2图谱的桥梁。此函数通过比较SxS标记在两个亲本图谱中的位置，
# 来确保P1的LG1对应P2的LG1，P2的LG2对应P2的LG2...
# 它会以P1为模板，重命名P2的连锁群编号。
LGHomDf_P2_2 <- consensus_LG_names(modify_LG = LGHomDf_P2_1, 
                                   template_SxS = P1_SxS_Assigned, 
                                   modify_SxS = P2_SxS_Assigned)



## 阶段五：构建最终图谱与输出 (Final Map Construction and Export)

最后一步是计算标记间的遗传距离（单位：cM），构建完整的遗传图谱，并输出可用于QTL定位等下游分析的文件。


# ==============================================================================
# 步骤 5.1: 将所有剩余的标记类型分配到图谱骨架上
#
# homologue_lg_assignment 是一个综合性的函数，它的核心任务是“查缺补漏”。
# 它会遍历filtered_data中所有还未被分配的标记，计算它们与图谱骨架上所有标记的连锁关系，
# 并根据最强的连锁信号（最高的LOD值）将其“锚定”到最合适的同源组上。
#
# 参数说明:
# - assigned_list & assigned_markertypes: 告诉函数哪些标记（和它们的类型）我们已经手动分配过了
#   （如此前的SxS和DxN），以避免重复计算，提高效率。
# - LG_hom_stack: 我们已经构建好的、可靠的图谱骨架（来自1x0标记）。
# ==============================================================================

# 修正原函数bug
assign_linkage_group_FIXED <- function (linkage_df, LG_hom_stack, SN_colname = "marker_a", 
    unassigned_marker_name = "marker_b", phase_considered = "coupling", 
    LG_number, LOD_threshold = 3, ploidy, assign_homologue = T, 
    log = NULL) 
{
    # 修正：使用 ::: 访问内部函数
    LG_hom_stack <- polymapR:::test_LG_hom_stack(LG_hom_stack)
    
    if (length(levels(factor(LG_hom_stack$homologue))) > ploidy) {
        stop("The number of homologues per chromosome should not exceed ploidy.")
    }
    if (length(unique(LG_hom_stack$LG)) != LG_number) {
        stop(paste("Only", length(unique(LG_hom_stack$LG)), 
            "linkage groups were identified in LG_hom_stack. Please revise LG_number accordingly."))
    }
    linkage_df <- linkage_df[linkage_df$LOD > LOD_threshold & 
        linkage_df$phase == phase_considered, , drop = FALSE]
    if (is.null(linkage_df)) {
        message("There were no linkage groups the marker could be assigned to")
        return(NULL)
    }
    if (nrow(linkage_df) == 0) {
        message("There were no linkage groups the marker could be assigned to")
        return(NULL)
    }
    SN_markers <- levels(as.factor(linkage_df[, SN_colname]))
    unassigned_markers <- levels(as.factor(as.character((linkage_df[, 
        unassigned_marker_name]))))
    LG_hom_stack$LG <- as.factor(LG_hom_stack$LG)
    LG_hom_stack$homologue <- as.factor(LG_hom_stack$homologue)
    comb_df <- merge(linkage_df[, c(SN_colname, unassigned_marker_name)], 
        LG_hom_stack, by.x = SN_colname, by.y = "SxN_Marker", 
        all.x = T)
    count_tables <- tapply(1:nrow(comb_df), as.character(comb_df[, 
        unassigned_marker_name]), function(x) {
        table(comb_df[x, "LG"], comb_df[x, "homologue"])
    })
    if (LG_number > 1) {
        chm.counts <- sapply(count_tables, rowSums)
    }
    else {
        chm.counts <- matrix(sapply(count_tables, rowSums), 
            nrow = 1, dimnames = list(1, names(count_tables)))
    }
    counts_chm <- t(matrix(chm.counts, nrow = nrow(chm.counts), 
        dimnames = list(paste0("LG", rownames(chm.counts)), 
            names(count_tables))))
    unlinked_markers <- rownames(counts_chm)[rowSums(counts_chm) == 
        0]
    if (length(setdiff(unassigned_markers, unlinked_markers)) == 
        0) {
        message("There were no linkage groups the marker could be assigned to")
        return(NULL)
    }
    counts_chm <- counts_chm[!rownames(counts_chm) %in% unlinked_markers, 
        , drop = FALSE]
    count_tables <- count_tables[rownames(counts_chm)]
    warn_lg <- apply(counts_chm, 1, function(x) {
        m <- max(x, na.rm = T)
        a <- m/x < 2
        return(sum(a, na.rm = T) > 1)
    })
    if (is.null(log)) {
        log.conn <- stdout()
    }
    else {
        matc <- match.call()
        # 修正：使用 ::: 访问内部函数
        polymapR:::write.logheader(matc, log)
        log.conn <- file(log, "a")
    }
    if (sum(warn_lg) > 0) {
        write("\n#### Marker(s) showing ambiguous linkage to more than one LG:\n", 
            log.conn)
        # 修正：使用 ::: 访问内部函数
        amb.m <- polymapR:::vector.to.matrix(unassigned_markers[warn_lg], 
            4)
        write(knitr::kable(amb.m), log.conn)
    }
    Assigned_LG <- as.numeric(rownames(chm.counts))[apply(counts_chm, 
        1, which.max)]
    counts_hom <- t(sapply(count_tables, function(x) {
        x[which.max(rowSums(x)), ]
    }))
    if (ncol(counts_hom) != ploidy) {
        counts_hom <- cbind(counts_hom, matrix(0, ncol = ploidy - 
            ncol(counts_hom), nrow = nrow(counts_hom)))
    }
    colnames(counts_hom) <- paste0("Hom", 1:ploidy)
    if (assign_homologue) {
        assigned_hom <- t(apply(counts_hom, 1, function(x) {
            s <- sum(x < 1)
            o <- order(x)
            o[0:s] <- NA
            return(rev(o))
        }))
        if (phase_considered == "repulsion") {
            assigned_hom <- t(apply(assigned_hom, 1, function(x) {
                d <- setdiff(1:ploidy, x)
                return(c(d, rep(NA, sum(!is.na(x)))))
            }))
        }
    }
    else {
        nmark <- length(Assigned_LG)
        assigned_hom <- matrix(rep(1:ploidy, nmark), ncol = ploidy, 
            byrow = T)
    }
    colnames(assigned_hom) <- paste0("Assigned_hom", 1:ploidy)
    output <- cbind(Assigned_LG, counts_chm, counts_hom, assigned_hom)
    write(paste("\n In total,", length(Assigned_LG), "out of", 
        length(unassigned_markers), "markers were assigned."), 
        log.conn)
    if (length(unlinked_markers) > 0) {
        write("\n#### Marker(s) not assigned:\n", log.conn)
        # 修正：使用 ::: 访问内部函数
        unl.m <- polymapR:::vector.to.matrix(unlinked_markers, 4)
        write(knitr::kable(unl.m), log.conn)
    }
    if (!is.null(log)) 
        close(log.conn)
    return(output)
}

homologue_lg_assignment_FIXED <- function (input_type = "discrete", dosage_matrix, probgeno_df, 
    chk, assigned_list, assigned_markertypes, SN_functions = NULL, 
    LG_hom_stack, parent1 = "P1", parent2 = "P2", which_parent = 1, 
    ploidy, ploidy2 = NULL, convert_palindrome_markers = TRUE, 
    pairing = "random", LG_number, LOD_threshold = 3, write_intermediate_files = TRUE, 
    log = NULL, ...) 
{
    input_type <- match.arg(input_type, choices = c("discrete", 
        "probabilistic"))
    # 修正：使用 ::: 访问内部函数
    LG_hom_stack <- polymapR:::test_LG_hom_stack(LG_hom_stack)
    if (!which_parent %in% 1:2) 
        stop("which_parent must be either 1 or 2!")
    if (which_parent == 1) {
        target_parent <- parent1
        other_parent <- parent2
    }
    else {
        target_parent <- parent2
        other_parent <- parent1
    }
    if (input_type == "discrete") {
        # 修正：使用 ::: 访问内部函数
        dosage_matrix <- polymapR:::test_dosage_matrix(dosage_matrix)
        if (!parent1 %in% colnames(dosage_matrix) | !parent2 %in% 
            colnames(dosage_matrix)) 
            stop("Incorrect column name identifiers supplied for parent(s). Please check!")
    }
    else {
        probgeno_df <- polymapR:::test_probgeno_df(probgeno_df)
        pardose <- polymapR:::assign_parental_dosage(chk = chk, probgeno_df = probgeno_df)
    }
    assigned_markers <- lapply(assigned_list, rownames)
    assigned_markers <- do.call("c", assigned_markers)
    if (input_type == "discrete") {
        filt_dosdat <- dosage_matrix[!rownames(dosage_matrix) %in% 
            assigned_markers, ]
    }
    else {
        filt_score <- probgeno_df[!probgeno_df$MarkerName %in% 
            assigned_markers, ]
    }
    if (pairing == "random") {
        pairing_abbr <- "r"
    }
    else if (pairing == "preferential") {
        pairing_abbr <- "p"
    }
    sn.grep1 <- "_1.0_"
    sn.grep2 <- "_1.0_1.0"
    p1_ploidy <- ploidy
    p2_ploidy <- if (!is.null(ploidy2)) ploidy2 else ploidy
    if (which_parent == 1) {
        target.ploidy <- p1_ploidy
        other.ploidy <- p2_ploidy
    }
    else {
        target.ploidy <- p2_ploidy
        other.ploidy <- p1_ploidy
    }
    ploidy.F1 <- (p1_ploidy + p2_ploidy)/2
    if (!is.null(ploidy2)) {
        if (target.ploidy < other.ploidy) {
            sn.grep1 <- "_2_1.0_"
            sn.grep2 <- "_2_1.0_1.0"
        }
        else {
            sn.grep1 <- "_4_1.0_"
            sn.grep2 <- "_4_1.0_1.0"
        }
    }
    avail_funs <- ls(getNamespace("polymapR"))
    linkage_functions <- avail_funs[grep(paste0(pairing_abbr, 
        ploidy.F1, "_"), avail_funs)]
    already_assigned_functions <- sapply(assigned_markertypes, 
        function(x) {
            paste0(pairing_abbr, ploidy.F1, sn.grep1, paste(x, 
                collapse = "."))
        })
    linkage_functions <- linkage_functions[!linkage_functions %in% 
        c(paste0(pairing_abbr, ploidy.F1, sn.grep2), already_assigned_functions)]
    if (is.null(SN_functions)) {
        SN_functions <- linkage_functions[grep(paste0(pairing_abbr, 
            ploidy.F1, sn.grep1), linkage_functions)]
    }
    marker_combinations <- do.call(rbind, strsplit(SN_functions, 
        "[_.]"))
    if (ploidy.F1 != 3) {
        marker_combinations <- marker_combinations[, -1, drop = FALSE]
    }
    else {
        marker_combinations <- marker_combinations[, -c(1, 2), 
            drop = FALSE]
    }
    class(marker_combinations) <- "integer"
    if (is.null(log)) {
        log.conn <- stdout()
    }
    else {
        matc <- match.call()
        polymapR:::write.logheader(matc, log)
        log.conn <- file(log, "a")
    }
    if (!is.null(log) & length(marker_combinations) > 0) {
        pb <- txtProgressBar(min = 0, max = nrow(marker_combinations), 
            style = 3)
    }
    for (i in seq(nrow(marker_combinations))) {
        if (!is.null(log)) 
            sink(log.conn, append = TRUE)
        write(paste0("Calculating r and LOD between ", paste(marker_combinations[i, 
            1:2], collapse = "."), " and ", paste(marker_combinations[i, 
            3:4], collapse = "."), " markers..."), stdout())
        mtype1 <- marker_combinations[i, 1:2]
        mtype2 <- marker_combinations[i, 3:4]
        if (input_type == "discrete") {
            linkage_df <- linkage(dosage_matrix = filt_dosdat, 
                markertype1 = mtype1, markertype2 = mtype2, 
                parent1 = parent1, parent2 = parent2, which_parent = which_parent, 
                convert_palindrome_markers = convert_palindrome_markers, 
                LOD_threshold = 0, ploidy = ploidy, ploidy2 = ploidy2, 
                pairing = pairing, verbose = FALSE, ...)
        }
        else {
            linkage_df <- linkage.gp(probgeno_df = filt_score, 
                chk = chk, pardose = pardose, markertype1 = mtype1, 
                markertype2 = mtype2, target_parent = c(parent1, 
                  parent2)[which_parent], LOD_threshold = 0, 
                verbose = FALSE, ...)
        }
        if (write_intermediate_files) {
            mname1 <- paste(marker_combinations[i, 1:2], collapse = "x")
            mname2 <- paste(marker_combinations[i, 3:4], collapse = "x")
            saveRDS(linkage_df, paste0(target_parent, "_", mname1, 
                "_", mname2, ".RDS"))
        }
        
        # 修正：调用我们第一步修正好的 assign_linkage_group_FIXED 函数
        assignedData <- assign_linkage_group_FIXED(linkage_df = linkage_df, 
            LG_hom_stack = LG_hom_stack, phase_considered = "coupling", 
            LG_number = LG_number, LOD_threshold = LOD_threshold, 
            ploidy = target.ploidy, assign_homologue = TRUE)
            
        if (write_intermediate_files) {
            write.table(assignedData, paste0(target_parent, 
                "_", mname2, "_Assigned.txt"), sep = "\t")
        }
        assigned_name <- paste0("assigned_", paste(mtype1, collapse = "."), 
            "_", paste(mtype2, collapse = "."))
        assign(assigned_name, get("assignedData"))
        assigned_list[[assigned_name]] <- get(assigned_name)
        write("\n________________________________________\n", 
            stdout())
        if (!is.null(log)) 
            sink()
        if (!is.null(log)) 
            setTxtProgressBar(pb, i)
    }
    if (!is.null(log)) 
        sink(log.conn, append = TRUE)
    if (input_type == "discrete") {
        # 修正：使用 ::: 访问内部函数
        marker_assignments <- polymapR:::merge_marker_assignments(dosage_matrix = dosage_matrix, 
            target_parent = target_parent, other_parent = other_parent, 
            LG_hom_stack = LG_hom_stack, SN_linked_markers = assigned_list, 
            ploidy = target.ploidy, LG_number = LG_number)
    }
    else {
        marker_assignments <- polymapR:::merge_marker_assignments.gp(MarkerType = pardose, 
            target_parent = target_parent, other_parent = other_parent, 
            LG_hom_stack = LG_hom_stack, SN_linked_markers = assigned_list, 
            ploidy = target.ploidy, LG_number = LG_number)
    }
    if (!is.null(log)) 
        sink()
    if (!is.null(log)) 
        close(log.conn)
    return(marker_assignments)
}



# 为亲本P1分配所有剩余标记
marker_assignments_P1 <- homologue_lg_assignment_FIXED(dosage_matrix = filtered_data,
                                                 assigned_list = list(P1_SxS_Assigned, 
                                                                        P1_DxN_Assigned), 
                                                 assigned_markertypes = list(c(1,1), c(2,0)),
                                                 LG_hom_stack = LGHomDf_P1, 
                                                 which_parent = 1,
                                                 ploidy = 4,
                                                 pairing = "random",
                                                 convert_palindrome_markers = FALSE,
                                                 LG_number = 5,
                                                 LOD_threshold = 4,
                                                 write_intermediate_files = FALSE)





# 为亲本P2分配所有剩余标记 (使用已经和P1统一了LG名称的 LGHomDf_P2_2)
marker_assignments_P2 <- homologue_lg_assignment_FIXED(dosage_matrix = filtered_data,
                                                 assigned_list = list(P2_SxS_Assigned, 
                                                                        P2_DxN_Assigned),
                                                 assigned_markertypes = list(c(1,1), c(2,0)),
                                                 LG_hom_stack = LGHomDf_P2,
                                                 which_parent = 2,
                                                 ploidy = 4,
                                                 pairing = "random",
                                                 convert_palindrome_markers = TRUE, # P2需要转换回文标记
                                                 LG_number = 5,
                                                 LOD_threshold = 4,
                                                 write_intermediate_files = FALSE)
head(marker_assignments_P2)

# ==============================================================================
# 步骤 5.2: 检查双亲标记分配的一致性
#
# 目的：这是一个非常重要的质控步骤。它会检查同一个标记是否在P1和P2的图谱中被分配到了
# 不同的连锁群上。例如，如果标记M1在P1中被定到LG1，但在P2中被定到LG2，
# 这通常暗示着上游的连锁群构建或统一命名步骤中可能存在问题。
# ==============================================================================
marker_assignments <- check_marker_assignment(marker_assignments_P1, marker_assignments_P2 )


# ==============================================================================
# 步骤 5.3: 完成连锁分析，准备进行标记排序
#
# 目的：到目前为止，我们只是把标记“分配”到了某个同源组，但还不知道它们在
# 染色单体上的具体顺序和距离。
# finish_linkage_analysis 会为每个同源组内的所有标记（骨架+新分配的）计算
# 它们两两之间的重组率(r)和LOD值，为下一步的排序和距离计算提供输入数据。
# ==============================================================================
all_linkages_list_P1 <- finish_linkage_analysis(marker_assignment = marker_assignments$P1,
                                                dosage_matrix = filtered_data,
                                                which_parent = 1,
                                                convert_palindrome_markers = FALSE,
                                                ploidy = 4,
                                                pairing = "random",
                                                LG_number = 5)

all_linkages_list_P2 <- finish_linkage_analysis(marker_assignment = marker_assignments$P2,
                                                dosage_matrix = filtered_data,
                                                which_parent = 2,
                                                convert_palindrome_markers = TRUE,
                                                ploidy = 4,
                                                pairing = "random",
                                                LG_number = 5)
# str(all_linkages_list_P1) # 可以查看生成的数据结构

# ==============================================================================
# 步骤 5.4: 整合图谱并计算遗传距离
#
# 目的：这是构建图谱的核心步骤。
# 1. 首先将P1和P2的连锁数据合并成MDSMap包要求的格式。
# 2. 调用 MDSMap_from_list 函数，它使用多维尺度分析（MDS）算法来推断每个
#    同源组内标记的最佳顺序，并计算出它们之间的遗传距离（单位：厘摩 cM）。
# ==============================================================================
# 1. 合并P1和P2的连锁数据

linkages <- list()
for(lg in names(all_linkages_list_P1)){
  linkages[[lg]] <- rbind(all_linkages_list_P1[[lg]], all_linkages_list_P2[[lg]])
}

# 2. 使用MDS算法构建整合图谱
integrated.maplist <- MDSMap_from_list(linkages)

# ==============================================================================
# 步骤 5.5: 将被bin的重复标记添加回图谱
#
# 目的：在分析初期，为了减少计算量，我们将遗传上完全相同的标记（共分离标记）
# 打包(binning)并只用一个代表。现在图谱顺序已经确定，我们需要把那些被
# “收起来”的标记重新放回到它们代表标记的相同位置上，使图谱更完整。
# ==============================================================================
complete_mapdata <- add_dup_markers(maplist = integrated.maplist,
                                    bin_list = screened_data4$bin_list,
                                    marker_assignments = marker_assignments)
# 更新图谱列表和标记分配列表
integrated.maplist_complete <- complete_mapdata$maplist
marker_assignments_complete <- complete_mapdata$marker_assignments

# ==============================================================================
# 步骤 5.6: 创建最终的分相图谱 (Phased Map)
#
# 最终目标：这是整个流程最重要的产出之一。
# create_phased_maplist 会确定每个标记的等位基因具体位于P1和P2的哪一条
# 同源染色单体上。对于四倍体，它会给出标记在8条染色单体（P1的4条+P2的4条）
# 上的完整分相信息，这是进行QTL定位的基础。
# ==============================================================================
phased.maplist <- create_phased_maplist(maplist = integrated.maplist, # 可以用 complete 版本
                                          dosage_matrix.conv = filtered_data,
                                          N_linkages = 5,
                                          ploidy = 4,
                                          marker_assignment.1 = marker_assignments$P1,
                                          marker_assignment.2 = marker_assignments$P2)

# ==============================================================================
# 步骤 5.7: 可视化、质控与结果输出
# ==============================================================================
# 1. 绘制整合图谱
plot_map(maplist = integrated.maplist)

# 2. 绘制分相图谱（以第一个连锁群为例）
plot_phased_maplist(phased.maplist = phased.maplist[1], 
                      ploidy = 4,
                      cols = c("black","grey50","grey50"))

# 3. 检查最终图谱质量（如图谱顺序与LOD值的对应关系）
check_map(linkage_list = linkages[1], maplist = integrated.maplist[1])

# 4. 输出用于QTL定位的输入文件（如TetraploidSNPMap格式）
write.TSNPM(phased.maplist = phased.maplist, ploidy = 4)


```

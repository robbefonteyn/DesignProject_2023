library(Signac)
library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)
library(rtracklayer)
library(ggplot2)
library(patchwork)

counts <- Read10X_h5("./10k_PBMC_Multiome_nextgem_Chromium_Controller_filtered_feature_bc_matrix.h5")
fragpath <- "./10k_PBMC_Multiome_nextgem_Chromium_Controller_atac_fragments.tsv.gz"

annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevels(annotation) <- paste0('chr', seqlevels(annotation))

pbmc <- CreateSeuratObject(
  counts = counts$`Gene Expression`,
  assay = "RNA"
)

pbmc[["ATAC"]] <- CreateChromatinAssay(
  counts = counts$Peaks,
  sep = c(":", "-"),
  fragments = fragpath,
  annotation = annotation
)

DefaultAssay(pbmc) <- "ATAC"
pbmc <- NucleosomeSignal(pbmc)
pbmc <- TSSEnrichment(pbmc)

low_ncr <- quantile(pbmc[["nCount_RNA"]]$nCount_RNA, probs = 0.02)
hig_ncr <- quantile(pbmc[["nCount_RNA"]]$nCount_RNA, probs = 0.98)
hig_ns <- quantile(pbmc[["nucleosome_signal"]]$nucleosome_signal, probs = 0.98)
low_ts <- quantile(pbmc[["TSS.enrichment"]]$TSS.enrichment, probs = 0.02)
low_nca <- quantile(pbmc[["nCount_ATAC"]]$nCount_ATAC, probs = 0.02)
high_nca <- quantile(pbmc[["nCount_ATAC"]]$nCount_ATAC, probs = 0.98)

pbmc <- subset(
  x = pbmc,
  subset = nCount_RNA > low_ncr &
    nCount_RNA < hig_ncr &
    nucleosome_signal < hig_ns &
    TSS.enrichment > low_ts &
    nCount_ATAC > low_nca &
    nCount_ATAC < high_nca
)

DefaultAssay(pbmc) <- "RNA"
pbmc <- FindVariableFeatures(pbmc, nfeatures = 3000)
pbmc <- NormalizeData(pbmc)
pbmc <- ScaleData(pbmc)
pbmc <- RunPCA(pbmc, npcs = 50)
pbmc <- RunUMAP(pbmc, dims = 1:50, reduction.name = "umap.rna")
pbmc <- FindNeighbors(pbmc, dims = 1:50)
pbmc <- FindClusters(pbmc, resolution = 0.5, algorithm = 3)


DefaultAssay(pbmc) <- "ATAC"
pbmc <- RunTFIDF(pbmc)
pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
pbmc <- RunSVD(pbmc)
pbmc <- RunUMAP(pbmc, reduction = 'lsi', dims = 2:30, reduction.name = 'umap.atac')


pbmc.ref <- readRDS("./pbmc_multimodal_2023.rds")

DefaultAssay(pbmc) <- "RNA"
anchors <- FindTransferAnchors(
  reference = pbmc.ref,
  query = pbmc,
  reference.reduction = "spca",
  normalization.method = "SCT",
  dims = 1:50
)

predicted.labels <- TransferData(
  anchorset = anchors,
  refdata = pbmc.ref$celltype.l2,
  weight.reduction = pbmc[['pca']],
  dims = 1:50
)

pbmc <- AddMetaData(
  object = pbmc,
  metadata = predicted.labels
)

# visualize label transfer
p1 <- DimPlot(pbmc.ref, reduction = "umap", group.by = "celltype.l2", label = TRUE, label.size = 3, repel = TRUE) +
  NoLegend() +
  ggtitle("Reference PBMC") +
  theme(axis.text.x = element_blank(), axis.text.y = element_blank()) +
  labs(x = "", y = "")
p2 <- DimPlot(pbmc, reduction = "ref.umap", group.by = "predicted.id", label = TRUE, label.size = 3, repel = TRUE) +
  NoLegend() +
  ggtitle("Query PBMC") +
  theme(axis.text.x = element_blank(), axis.text.y = element_blank()) +
  labs(x = "", y = "")
combined_plot <- p1 + p2
print(combined_plot)

DefaultAssay(pbmc) <- "ATAC"
gene.activities <- GeneActivity(pbmc)
pbmc[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)
pbmc <- NormalizeData(
  object = pbmc,
  assay = 'ACTIVITY',
  normalization.method = 'LogNormalize',
  scale.factor = median(pbmc$nCount_ACTIVITY)
)

SaveH5Seurat(pbmc, filename = "pbmc_multiome.h5Seurat",overwrite = TRUE)
Convert("pbmc_multiome.h5Seurat", dest = "h5ad", overwrite = TRUE)


# visualize alignment of random gene (CD8A)
DefaultAssay(pbmc) <- "ATAC"
cov_plot <- CoveragePlot(
  object = pbmc,
  region = "CD8A",
  annotation = TRUE,
  peaks = TRUE
)
print(cov_plot)



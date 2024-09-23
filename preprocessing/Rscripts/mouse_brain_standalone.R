library(Signac)
library(Seurat)
library(EnsDb.Mmusculus.v79)
library(ggplot2)
library(patchwork)
library(SeuratDisk)

counts <- Read10X_h5("./8k_mouse_cortex_ATACv2_nextgem_Chromium_Controller_filtered_peak_bc_matrix.h5")
metadata <- read.csv(
  file = "./8k_mouse_cortex_ATACv2_nextgem_Chromium_Controller_singlecell.csv",
  header = TRUE,
  row.names = 1
)

brain_assay <- CreateChromatinAssay(
  counts = counts,
  sep = c(":", "-"),
  genome = "mm10",
  fragments = './8k_mouse_cortex_ATACv2_nextgem_Chromium_Controller_fragments.tsv.gz',
  min.cells = 1
)

brain <- CreateSeuratObject(
  counts = brain_assay,
  assay = 'peaks',
  project = 'ATAC',
  meta.data = metadata
)

annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Mmusculus.v79)
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- "mm10"
Annotation(brain) <- annotations

brain <- NucleosomeSignal(object = brain)
brain <- TSSEnrichment(brain, fast = TRUE)
brain$pct_reads_in_peaks <- brain$peak_region_fragments / brain$passed_filters * 100
brain$blacklist_ratio <- brain$blacklist_region_fragments / brain$peak_region_fragments

low_prf <- quantile(brain[["peak_region_fragments"]]$peak_region_fragments, probs = 0.02)
hig_prf <- quantile(brain[["peak_region_fragments"]]$peak_region_fragments, probs = 0.98)
low_prp <- quantile(brain[["pct_reads_in_peaks"]]$pct_reads_in_peaks, probs = 0.05)
hig_ns <- quantile(brain[["nucleosome_signal"]]$nucleosome_signal, probs = 0.98)
low_ts <- quantile(brain[["TSS.enrichment"]]$TSS.enrichment, probs = 0.05)
low_ncp <- quantile(brain[["nCount_peaks"]]$nCount_peaks, probs = 0.05)

brain <- subset(
  x = brain,
  subset = peak_region_fragments > low_prf &
    peak_region_fragments < hig_prf &
    pct_reads_in_peaks > low_prp &
    nucleosome_signal < hig_ns &
    TSS.enrichment > low_ts &
    nCount_peaks > low_ncp
  )

brain <- RunTFIDF(brain)
brain <- FindTopFeatures(brain, min.cutoff = 'q0')
brain <- RunSVD(object = brain)

brain <- RunUMAP(
  object = brain,
  reduction = 'lsi',
  dims = 2:30
)
brain <- FindNeighbors(
  object = brain,
  reduction = 'lsi',
  dims = 2:30
)
brain <- FindClusters(
  object = brain,
  algorithm = 3,
  resolution = 1.2,
  verbose = FALSE
)


gene.activities <- GeneActivity(brain)
brain[['ACTIVITY']] <- CreateAssayObject(counts = gene.activities)
brain <- NormalizeData(
  object = brain,
  assay = 'ACTIVITY',
  normalization.method = 'LogNormalize',
  scale.factor = median(brain$nCount_ACTIVITY)
)

allen_rna <- readRDS("./allen_brain.rds")
allen_rna <- UpdateSeuratObject(allen_rna)
allen_rna <- FindVariableFeatures(
  object = allen_rna,
  nfeatures = 5000
)

transfer.anchors <- FindTransferAnchors(
  reference = allen_rna,
  query = brain,
  reduction = 'cca',
  dims = 1:30
)

predicted.labels <- TransferData(
  anchorset = transfer.anchors,
  refdata = allen_rna$subclass,
  weight.reduction = brain[['lsi']],
  dims = 2:30
)

brain <- AddMetaData(object = brain, metadata = predicted.labels)

SaveH5Seurat(brain, filename = "mouse_brain.h5Seurat")
Convert("mouse_brain.h5Seurat", dest = "h5ad")


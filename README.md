<h1 align="center">Regulatory impact of noncoding somatic mutations is associated with cancer progression</h1>

## Overview

Somatic mutations in noncoding regulatory regions are abundant in cancer genomes, yet their contribution to tumor behavior and clinical outcome remains poorly understood as functional impact cannot be inferred from mutation frequency alone. This repository contains code supporting the manuscript analyzing the clinical relevance of regulatory disruption from gene-proximal somatic mutations in cancer.


## Repository Structure

The analysis pipeline consists of four main steps: 

### 1. Variant Distribution Analysis (`step1_variant_distributions.py`)
- Maps variants to genes within ±20 kb of transcription start sites
- Normalizes Sei predictions across 40 sequence classes
- Compares regulatory scores between COSMIC Hallmark genes and non-COSMIC genes

### 2. Survival Analysis (`step2_regulatory_pfi_models.py`)
- Kaplan-Meier curve for samples with high and low regulatory effects 
- Cox proportional hazards regression controlling for mutation burden, FGA, age, and coding mutations
- Stability analyses: gene dropout, sample dropout, leave-one-cancer-out (LOCO)
- Generates forest plots, Kaplan-Meier curves, and recurrence visualizations

### 3. Clonality-Aware Analysis (`step3_regulatory_pfi_with_clonality.py`)
- Integrates PyClone cellular prevalence estimates to separate clonal and subclonal mutations
- Independent Cox models for clonal and subclonal regulatory scores

### 4. Regulatory Clustering (`step4_clustering.py`)
- Clustering of samples by chromatin sequence class-specific regulatory disruption
- Stability analysis to determine optimal cluster number
- Gene-cluster alignment and Jaccard overlap analysis

### Supporting Modules
- `create_clonality_aware_score.py`: Pre-processing pipeline for aggregating variant-level SEI scores to gene x sample level
- `utils.py`: Core functions for survival analysis, normalization, and stability testing
- `visualization_utils.py`: Plotting functions for publication-quality figures

## Data Requirements
Please note that part of this analysis requires access to the TCGA WGS files

## Requirements

See `requirements.txt` for dependencies. Key packages include:
- `lifelines` (Cox regression and Kaplan-Meier analysis)
- `scikit-learn`, `scipy` (clustering and statistical testing)
- `pandas`, `numpy` (data processing)
- `matplotlib`, `seaborn` (visualization)

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:

```
[Citation information will be added upon publication]
```
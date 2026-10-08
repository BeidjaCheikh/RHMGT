# RHMGT

**Relational Hierarchical Molecular Graph Transformer for Interpretable hERG Blocker Prediction**

RHMGT is a multimodal molecular learning framework for hERG blocker prediction and cardiotoxicity assessment. The model combines a relational hierarchical molecular graph with complementary molecular evidence derived from fingerprints and physicochemical descriptors.

The framework represents molecules at multiple levels, including atoms, structural motifs, explicit pharmacophore occurrences, and motif-level relations, and provides interpretation at the atom, motif, pharmacophore, and relation levels.

This repository contains the implementation used for the experiments reported in the associated manuscript.

---

## Overview

RHMGT combines two complementary molecular representation pathways:

1. **Relational hierarchical molecular graph**
   - Atom-level molecular graph
   - BRICS-derived structural motifs
   - Explicit hERG-related pharmacophore occurrences
   - Typed hierarchical relations
   - Topological-distance information
   - Bond-chemistry information
   - Learned global molecular node
   - Relational Graph Transformer

2. **Complementary molecular evidence**
   - Morgan fingerprints
   - MACCS keys
   - AtomPair fingerprints
   - Physicochemical descriptors

The two representations are integrated using a molecule-specific gated fusion mechanism before binary hERG blocker classification.

---

## Repository Structure

```text
RHMGT/
├── configs/                 # Experimental and model configurations
├── data/                    # Dataset-related files and split information
├── final_model/             # Final selected model and associated files
├── hera_hgt/                # Core Python implementation
├── outputs/                 # Experimental and interpretability outputs
├── scripts/                 # Training, evaluation and analysis scripts
├── tests/                   # Tests
├── tuning/                  # Hyperparameter optimization utilities
├── ABALATION_STUDY.md       # Ablation-study documentation
├── G0_MULTIOBJECTIVE_TUNING.md
├── environment.yml          # Conda environment
├── requirements.txt         # Python dependencies
└── README.md
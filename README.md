# GNN AGENT

## Overview

Supply-demand mismatch is a critical challenge in supply chain management, particularly for Small and Medium-sized Enterprises (SMEs) lacking systematic records. 

This repository provides the official implementation of our proposed dual-framework approach. To ensure clarity, the codebase is explicitly divided into two major components:
1. **Graph Inference Component:** A production-function-constrained Graph Machine Learning (GML) model that infers unobserved firm-level inventory changes from supply chain networks.
2. **Econometric Agent Component:** A multi-agent framework that activates diverse econometric models (Spatial, GMM, DID, Cross-lag, etc.) to validate the GML predictions strictly against economic theories.

## Repository Structure

```text
.
├── Graph_Inference_Model/              # Part 1: GNNs
│   ├── best_model_run.py               # Script to execute the best-performing graph model
│   ├── run_code_grid.py                # Script for hyperparameter grid search and training
│   └── data/                           # Desensitized supply chain topology and attributes
│       ├── company_inventory.csv       # (Desensitized sample)
│       ├── GICS_code.csv               # Industry classification codes
│       ├── industry_class.csv          
│       └── supply_chain_trade.csv      # Network transaction linkages
│
├── Econometric_Agent/                  # Part 2: Multi-Agent Validation Framework
│   ├── app/
│   │   ├── main.py                     # Agent initialization and entry point
│   │   ├── schemas.py                  # Data structures and prompt schemas
│   │   ├── skills/                     # Econometric validation skills invoked by the agent
│   │   │   ├── skill_model_cross_lag.py
│   │   │   ├── skill_model_did.py
│   │   │   ├── skill_model_forecast.py
│   │   │   ├── skill_model_gmm.py
│   │   │   ├── skill_model_spatial.py
│   │   │   ├── skill_model_synthesis.py
│   │   │   ├── skill_data_inventory.py
│   │   │   └── skill_data_io.py
│   │   └── utils/                      # Helper functions
│   ├── batch_run.py                    # Script for automated batch validation
│   ├── combine_consensus_plot.py       # Generates agent consensus visualizations
│   └── data/                           # Processed data for econometric validation
│       ├── inventory_difference.csv    # Baseline/Inferred differences
│       ├── IO_Matrix_21H2.csv          # Input-Output matrices across periods
│       ├── IO_Matrix_22H1.csv
│       ├── IO_Matrix_22H2.csv
│       └── IO_Matrix_23H1.csv
│
└── requirements.txt                    # Project dependencies

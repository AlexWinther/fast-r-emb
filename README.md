# FAST-R-Emb: Similarity-Based Test Suite Reduction using Embeddings for Test Representation

This project is created as an extension of FAST-R but using embedding-based test representations.
The contained experiments and analysis are used for a seminar paper with the following abstract:

Large regression test suites impose substantial computational costs, motivating techniques that reduce redundancy while preserving fault detection capability. Similarity-based \ac{tsr} addresses this challenge by selecting diverse subsets of tests, yet existing scalable approaches rely primarily on lexical representations that may fail to capture semantic similarity. Here we investigate whether replacing term-frequency vectors with pretrained code embeddings improves scalable similarity-based \ac{tsr}. We integrate CodeT5+ embeddings into two state-of-the-art black-box reduction algorithms, FAST++ and FAST-CS, and evaluate performance across ten benchmark programs from SIR and Defects4J comprising $4,137$ tests and $54$ faults. Across $47,500$ paired experimental runs, embeddings significantly reduce fault detection loss for C programs containing multiple faults, increasing detection odds by 14\% when combined with FAST++. In contrast, no measurable improvement is observed for Java programs characterized by sparse, single-fault versions. Embedding generation increases preprocessing time by approximately one order of magnitude but does not affect reduction complexity. These results demonstrate that semantic representations improve similarity-based \ac{tsr} only when dataset structure, test representation and selection strategy align, highlighting the importance of representation–algorithm interaction in scalable test optimization.


## General Project Structure

- [src/](src/): Source code
- [data/](data/): Test suite data (SIR and D4J datasets) and results
- [scripts/](scripts/): Standalone scripts for analysis and figure generation
- [notebooks/](notebooks/): Interactive notebooks used for investigating results + R markdown file doing the GLMM analysis
- [outputs/](outputs/): Generated results and plots
- [presentation_examples/](presentation_examples/): Sripts to generate examples for presentation


## Running the Experiments

The environment to run the experiments and analysis is managed by [uv](https://docs.astral.sh/uv/getting-started/). Below is guide to running the experiments using uv:

### Budget experiment

To run the budget experiment where representation generation is cached run the following command:

Note this runs across:
- 10 Test suites
- 50 runs
- 5 methods (2 representations * 2 reductions + 1 baseline)
- 19 budgets (min=0.05, max=0.95, step=0.05)

= 47,500 runs

```bash
uv run budget
```

This will generate a csv with the results of the experiment, which can be used for analysis. The results will be saved to a folder in [data/](data/) following this naming convention: data/results-%Y-%m-%d-%H%M%S/results.csv where %Y-%m-%d-%H%M%S is the current date and time. Some run metadata is also saved in the same folder under run-metadata.json. The experiment run used for the paper can for example be found at [this location](data/results-2026-01-23-115413/)


### Running time experiment

To run the running time experiment with full representation generation each run the following command:

Note this runs across:
- 10 Test suites
- 10 runs
- 4 methods (2 representations * 2 reductions)
- 10 budgets (min=0.1, max=1.00, step=0.1)

= 4,000 runs

```bash
uv run running_time
```

This also generates a csv with the results of the experiment, to be used for later analysis. This follows the exact same naming convention as the budget results explained above. The running time experiment run used for the paper can for example be found at [this location](data/results-2026-01-23-204650/)


## Running the Analysis


### Budget Experiment

The analysis of the budget experiment can be run using the following command:

```bash
uv run scripts/analysis.py
```

This will generate most of the figures used in the paper, as well as output the results to the console

The results of the GLMM analysis can be found in [results.pdf](notebooks/results.pdf)


### Running time experiment

The analysis of this experiment can be found in [notebooks/results_running_time.ipynb](notebooks/results_running_time.ipynb).

This notebook generates the figures used for this


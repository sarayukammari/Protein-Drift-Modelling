# Protein Drift Modelling

This repository models how protein mutations drive a sequence away from its wild-type embedding over time. It combines sequence-derived embedding vectors with a dynamic, time-varying energy landscape to estimate drift, visualize trajectories, and quantify identity half-life.

## Project goals

- Simulate mutation-induced displacement in high-dimensional embedding space
- Track protein identity decay relative to the wild type
- Visualize drift trajectories and landscape dynamics
- Quantify how quickly a mutant loses similarity to the reference protein

## Repository structure

- `embeddings.py` — builds embedding representations from sequence data
- `generate_mutants.py` — generates mutant sequences
- `simulate_drift.py` — runs the drift model and writes output summaries
- `landscape.py` — defines the dynamic energy landscape and animation utilities
- `animate_drift.py` — creates animated drift visualizations
- `plot_3d.py` — renders 3D trajectory plots
- `wt.fasta` and `mutants.fasta` — example input sequences
- `*.csv`, `*.png`, and `*.gif` files — generated analytical outputs and plots

## Quick start

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate embeddings or run the core simulation:

   ```bash
   python embeddings.py
   python simulate_drift.py
   ```

4. Render additional visualizations:

   ```bash
   python landscape.py --csv embedding_trajectory.csv --out drift.gif
   python animate_drift.py
   python plot_3d.py
   ```

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn

## Notes

This project is focused on exploratory protein drift modelling and visualization. The generated output artifacts are useful for analysis, but they are not source code and are best kept out of version control.

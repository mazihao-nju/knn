# Virtual Waiting Time Estimation in Non-Stationary Queues

This repository contains the code and experiments for our project **“Towards Better Estimation of Virtual Waiting Time in Non-Stationary Systems”**.
The goal is to improve the accuracy and efficiency of virtual waiting time estimation under non-stationary queueing models.

## 📌 Project Overview

* **Baseline**: Reproduction of Lin et al. (2019)’s *k-NN based virtual statistics* framework for virtual waiting time estimation.
* **Enhancements**:

  * **Adaptive k-NN Estimator**: Dynamically adjusts neighborhood size $k(t)$ based on local data density, noise, and curvature, reducing bias–variance tradeoff.
  * **AFKNN (Forest-based Adaptive k-NN)**: Learns $k(t)$ using a random forest trained with KFE-informed bias–variance objectives for fast inference.
  * **KDE-Mode & Mean-Shift Clustering Estimators**: Density-aware approaches that leverage multimodal arrival structures for more robust local averaging.
* **Ground Truth Benchmark**: Implemented **Kolmogorov Forward Equations (KFEs)** to compute exact waiting time distributions, serving as the evaluation baseline.

## 🔬 Numerical Experiments

* Queueing models: **E₂(t)/M/1/8** and **H₂(t)/M/1/8** with time-varying arrival rates.
* Comparisons among fixed-k, adaptive k-NN, AFKNN, KDE-mode, and KFEs.
* Evaluation metrics include bias, variance, and EMSE (empirical mean squared error).
* Results show:

  * **Adaptive k-NN** performs robustly under regime-switching scenarios.
  * **AFKNN** achieves lowest EMSE when training distribution aligns with deployment.
  * **KDE-based clustering** captures multimodal structures with significant computational savings.

## 📂 Repository Structure

* `simulation/` – Simulation code for E₂(t) and H₂(t) queue models.
* `knn/` – Fixed-k and adaptive k-NN implementations.
* `afknn/` – Random forest–based adaptive kNN (AFKNN).
* `kde/` – KDE-mode and mean-shift clustering estimators.
* `kfe/` – Kolmogorov Forward Equations solver for ground truth computation.
* `experiments/` – Scripts for running numerical comparisons and generating plots.

## 📈 Key Findings

* Adaptive selection of $k$ significantly improves estimation compared with global fixed-k.
* KDE-mode clustering provides a scalable alternative with lower complexity.
* KFEs enable rigorous benchmarking and reveal tradeoffs between bias and variance.


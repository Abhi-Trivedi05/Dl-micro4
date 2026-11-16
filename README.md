# Reimplementation and Enhancement of LSTM and GRU Networks for Time-Series Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/experiments.ipynb)

An academic, research-oriented Python and PyTorch reimplementation and empirical enhancement of recurrent neural networks for univariate time-series forecasting.

---

## 1. Project Introduction & Problem Statement

Time-series forecasting is a critical requirement across quantitative finance, algorithmic trading, supply chain management, and demand forecasting. Financial equity prices and behavioral human activity signals exhibit complex characteristics: non-stationarity, stochastic volatility, shifting regimes, and long-range temporal dependencies.

This project delivers an independent, modular, and reproducible implementation of deep recurrent neural networks for multi-step and single-step forecasting. Moving beyond high-level monolithic scripts, we formulate a leakage-free sequential learning pipeline, implement the foundational **Vanilla (Elman) RNN**, **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Unit (GRU)** architectures from scratch in PyTorch, and benchmark them across five controlled experimental dimensions.

---

## 2. Research Motivation & Objectives

### 2.1 Base Research Paper Reference
This project is built upon the foundational methodology presented in:
> **Title:** *"An Open-Source and Reproducible Implementation of GRU and LSTM Networks for Time Series Forecasting"*  
> **Authors:** Gissel Velarde, Pedro Brañez, Alejandro Bueno, Rodrigo Heredia, and Mateo Lopez  
> **Conference:** Accepted at the 8th International Conference on Time Series and Forecasting (ITISE 2022)  
> **Reference Repository:** [Alebuenoaz/LSTM-and-GRU-Time-Series-Forecasting](https://github.com/Alebuenoaz/LSTM-and-GRU-Time-Series-Forecasting)

### 2.2 Reimplementation Objectives & Meaningful Improvements
While reproducing the core experimental premise of Velarde et al. (2022), our implementation corrects critical methodological limitations and introduces major scientific enhancements:

1. **Framework Transition:** Re-engineered from legacy TensorFlow/Keras into modular, idiomatic **PyTorch**.
2. **Elimination of Data Leakage:** In the original notebook, min-max normalization was fitted globally over the complete timeline before dataset splitting, causing future lookahead bias. We strictly compute normalization parameters ($\min, \max$) solely on the historical training split.
3. **Rigorous Evaluation Scale:** Predictions are evaluated both on the normalized $[0, 1]$ scale and inverse-transformed back to original physical currency units (INR closing price) to reflect real-world prediction error.
4. **Architectural Inclusion of Vanilla RNN:** Implemented the classical Elman RNN as a baseline to isolate the empirical benefits of gated memory cells.
5. **Early Stopping & Model Checkpointing:** Introduced validation loss monitoring and checkpoint persistence to prevent overfitting.
6. **Five Systematic Ablation Experiments:**
   * Architecture Comparison (Vanilla RNN vs. LSTM vs. GRU vs. Naive Persistence Baseline)
   * Sequence Lookback Length Sensitivity ($w \in \{7, 14, 30\}$ days)
   * Hidden Layer Capacity Scaling ($H \in \{32, 64, 128\}$ units)
   * Dropout Regularization Study ($p \in \{0.0, 0.2, 0.4\}$)
   * Computational Efficiency & Inference Latency Benchmark

---

## 3. Dataset Description

The project benchmarks recurrent models across two diverse datasets:

| Property | `BANKEX.csv` | `activities.csv` |
| :--- | :--- | :--- |
| **Domain** | Financial Stock Market (Real-World) | Synthetic Human Activity |
| **Data Matrix Dimensions** | $10 \text{ rows} \times 3,032 \text{ days}$ | $10 \text{ rows} \times 3,584 \text{ steps}$ |
| **Total Observations** | $30,320$ | $35,840$ |
| **Missing Values (NaN)** | 0 | 0 |
| **Target Variable Range** | $[8.29, 1790.65]$ INR | $[0.00, 14.22]$ Activity units |
| **Key Characteristics** | High volatility, non-stationary trend | 7-day cyclical seasonality + upward linear drift |

### Indian Banking Constituents (BANKEX)
The 10 rows in `BANKEX.csv` correspond to the daily closing prices (from 2005-07-12 to 2017-11-03) of major constituent banks:
1. Axis Bank (`AXISBANK.BO`)
2. Bank of Baroda (`BANKBARODA.BO`)
3. Federal Bank (`FEDERALBNK.BO`)
4. HDFC Bank (`HDFCBANK.BO`)
5. ICICI Bank (`ICICIBANK.BO`)
6. IndusInd Bank (`INDUSINDBK.BO`)
7. Kotak Mahindra Bank (`KOTAKBANK.BO`)
8. Punjab National Bank (`PNB.BO`)
9. State Bank of India (`SBIN.BO`)
10. Yes Bank (`YESBANK.BO`)

---

## 4. Mathematical Model Formulations

```
                         Recurrent Architecture Taxonomy
    ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
    │       Vanilla RNN       │          LSTM           │           GRU           │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │  h_t = tanh(Wx + Wh)    │  i_t, f_t, o_t, c~_t    │  r_t (reset gate)       │
    │  y^_t = W_out * h_t     │  c_t = f*c + i*c~       │  z_t (update gate)      │
    │                         │  h_t = o * tanh(c_t)    │  h_t = (1-z)h~ + z*h    │
    │  4,353 parameters       │  17,217 parameters      │  12,929 parameters      │
    └─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 4.1 Vanilla (Elman) RNN
The standard Elman RNN maintains a hidden state vector $h_t \in \mathbb{R}^H$ computed via recurrent affine transformation:
$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
$$\hat{y} = W_{fc} h_T + b_{fc}$$

### 4.2 Long Short-Term Memory (LSTM)
LSTM addresses vanishing gradients by decoupling memory storage into cell state $c_t$ governed by input $i_t$, forget $f_t$, and output $o_t$ gates:
$$f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$$
$$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$
$$\tilde{c}_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$$
$$h_t = o_t \odot \tanh(c_t)$$

### 4.3 Gated Recurrent Unit (GRU)
GRU condenses the gating mechanism into reset $r_t$ and update $z_t$ gates without a distinct cell state:
$$r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})$$
$$z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})$$
$$\tilde{h}_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))$$
$$h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1}$$

---

## 5. Project Directory Structure

```text
DL-micro4/
│
├── data/
│   ├── raw/
│   │   ├── BANKEX.csv                  # Indian banking daily closing prices
│   │   └── activities.csv              # Synthetic cyclical activity series
│   └── processed/                      # Cached partitions
│
├── models/
│   ├── __init__.py                     # Model factory interface
│   ├── vanilla_rnn.py                  # PyTorch Vanilla (Elman) RNN
│   ├── lstm.py                         # PyTorch LSTM network
│   └── gru.py                          # PyTorch GRU network
│
├── src/
│   ├── __init__.py                     # Package definition
│   ├── data_loader.py                  # Raw CSV loading and dataset inspection
│   ├── preprocessing.py                # Leakage-free TimeSeriesScaler & sliding windows
│   ├── dataset.py                      # PyTorch Dataset and DataLoader wrappers
│   ├── train.py                        # Training engine with early stopping & clipping
│   ├── evaluate.py                     # MAE, MSE, RMSE, MDA with inverse transform
│   └── utils.py                        # Seeds, parameter counters, and publication plots
│
├── notebooks/
│   └── experiments.ipynb               # Fully executed interactive Colab notebook
│
├── results/
│   ├── checkpoints/                    # Saved best model weights (.pt)
│   ├── metrics/                        # Quantitative experiment CSV logs
│   └── plots/                          # High-resolution publication figures (300 DPI)
│
├── config.py                           # Central configuration dataclasses
├── main.py                             # Command-line driver for all 5 experiments
├── requirements.txt                    # Project dependencies
├── LICENSE                             # MIT Open Source License
└── README.md                           # Comprehensive documentation
```

---

## 6. Installation & Execution

### 6.1 Installation
Clone the repository and set up a clean Python 3.9+ virtual environment:
```bash
git clone https://github.com/Alebuenoaz/LSTM-and-GRU-Time-Series-Forecasting.git
cd DL-micro4

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Running All Experiments via CLI
Execute the full experimental suite (which trains models, saves checkpoints, computes metrics, and generates all publication plots):
```bash
python main.py --epochs 60 --device cpu --dataset BANKEX
```

### 6.3 Running in Google Colab / Jupyter
Launch Jupyter or open `notebooks/experiments.ipynb` directly in Google Colab:
```bash
jupyter notebook notebooks/experiments.ipynb
```
The notebook is completely self-contained and pre-rendered with all outputs.

---

## 7. Experimental Results and Observations

All metrics below are derived from actual model training and testing on the held-out test split of Axis Bank (`AXISBANK.BO`, $251$ trading days).

### 7.1 Experiment 1: Architectural Benchmark

| Model | Trainable Parameters | Training Time (s) | Best Epoch | Test MAE (INR) | Test RMSE (INR) | Directional Accuracy (MDA %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 0 | 0.00s | — | 6.1861 | 8.5956 | 48.80% |
| **Vanilla RNN** | 4,353 | 4.44s | 53 | 6.5709 | 9.1404 | **50.00%** |
| **LSTM** | 17,217 | 10.33s | 59 | 7.2237 | 10.1372 | 48.40% |
| **GRU** | 12,929 | 9.26s | 56 | 7.1821 | 10.0978 | 47.60% |

#### Observations:
* **The Random Walk Barrier:** In financial stock forecasting at horizon $f=1$, successive price steps exhibit extreme autocorrelation ($\rho > 0.99$). The simple persistence baseline ($\hat{y}_t = x_{t-1}$) is notoriously strong.
* **Vanilla RNN Efficiency:** Vanilla RNN demonstrated superior efficiency and lower test error ($9.14$ INR vs. $10.14$ INR for LSTM), converging in less than half the training time of LSTM.

---

### 7.2 Experiment 2: Sequence Lookback Window ($w$) Sensitivity

| Lookback Window ($w$) | Vanilla RNN RMSE (INR) | LSTM RMSE (INR) | GRU RMSE (INR) |
| :---: | :---: | :---: | :---: |
| **7 days** | **9.0789** | 10.4584 | 9.8983 |
| **14 days** | 9.2196 | 10.1974 | 10.0213 |
| **30 days** | 9.1404 | 10.1372 | 10.0978 |

#### Observations:
* Shorter sequence windows ($w=7$) achieved slightly better or equivalent error than longer windows ($w=30$).
* In volatile stock prices, long historical contexts introduce spurious correlations, whereas immediate recent prices contain the highest signal-to-noise ratio.

---

### 7.3 Experiment 3: Model Capacity & Hidden Layer Size ($H$)

| Hidden Units ($H$) | Model | Parameters | Test RMSE (INR) | Test MAE (INR) |
| :---: | :--- | :---: | :---: | :---: |
| **32** | Vanilla RNN | 1,153 | 9.5253 | 6.9458 |
| **32** | LSTM | 4,513 | 11.3137 | 8.3512 |
| **32** | GRU | 3,393 | 10.1315 | 7.2618 |
| **64** | Vanilla RNN | 4,353 | 9.1404 | 6.5709 |
| **64** | LSTM | 17,217 | 10.1372 | 7.2237 |
| **64** | GRU | 12,929 | 10.0978 | 7.1821 |
| **128** | Vanilla RNN | 16,897 | **8.6186** | **6.1843** |
| **128** | LSTM | 67,201 | 10.3573 | 7.4239 |
| **128** | GRU | 50,433 | 9.5776 | 6.7824 |

#### Observations:
* Scaling hidden units to $H=128$ in Vanilla RNN achieved the best overall error ($8.62$ INR), matching the persistence baseline while preserving directional sensitivity.

---

### 7.4 Experiment 4: Regularization via Dropout ($p$)

| Dropout Rate ($p$) | Vanilla RNN RMSE | LSTM RMSE | GRU RMSE |
| :---: | :---: | :---: | :---: |
| **$p = 0.0$** | **9.1404** | **10.1372** | **10.0978** |
| **$p = 0.2$** | 9.7606 | 10.8260 | 10.8318 |
| **$p = 0.4$** | 10.2010 | 11.8680 | 11.1698 |

#### Observations:
* In single-layer recurrent architectures applied to continuous price regression, stochastic feature dropping degrades predictive precision across all models. The best performance occurs at $p = 0.0$.

---

### 7.5 Experiment 5: Computational Efficiency & Latency Benchmark

| Model | Parameters | Training Time (s) | Single-Sample Latency (ms) | Test RMSE (INR) |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla RNN** | 4,353 | **4.44s** | **0.091 ms** | **9.1404** |
| **GRU** | 12,929 | 9.26s | 0.192 ms | 10.0978 |
| **LSTM** | 17,217 | 10.33s | 0.194 ms | 10.1372 |

#### Observations:
* Vanilla RNN executes single-sample inference in under $0.1$ ms, more than $2\times$ faster than LSTM and GRU.
* GRU trains faster and has $25\%$ fewer parameters than LSTM with equal forecasting accuracy.

---

## 8. Generated Visualizations

The pipeline generates publication-quality figures saved in `results/plots/`:

1. `training_validation_loss_curves.png`: Training vs. validation loss progression over epochs.
2. `actual_vs_predicted_time_series.png`: Test set ground-truth price trajectories overlaid with RNN, LSTM, and GRU predictions.
3. `rnn_lstm_gru_performance_comparison.png`: Architectural benchmark comparison bar chart.
4. `sequence_length_rmse_comparison.png`: Sensitivity analysis across sequence lookback windows ($7, 14, 30$).
5. `hidden_size_comparison.png`: Model capacity comparison across hidden unit dimensions ($32, 64, 128$).
6. `dropout_impact_comparison.png`: Effect of dropout rates on RMSE.
7. `training_time_comparison.png`: Runtime comparison across recurrent models.

---

## 9. Academic Limitations & Future Work

1. **Univariate Limitation:** This implementation uses univariate historical closing prices. Incorporating multivariate features (trading volume, macroeconomic indicators, technical oscillators like RSI and MACD) is a promising next step.
2. **Stationarity Transformations:** Modeling raw prices directly can induce non-stationary drift. Future work can forecast log-returns or fractional differenced series.
3. **Transformer Baselines:** Extending the benchmark to temporal attention models (e.g., Informer, PatchTST) will enable comparison against modern attention mechanisms.

---

## 10. References and Attribution

1. **Velarde, G., Brañez, P., Bueno, A., Heredia, R., & Lopez, M.** (2022). *An Open-Source and Reproducible Implementation of GRU and LSTM Networks for Time Series Forecasting.* Accepted at ITISE 2022: 8th International Conference on Time Series and Forecasting.
2. **Reference GitHub Repository:** [https://github.com/Alebuenoaz/LSTM-and-GRU-Time-Series-Forecasting](https://github.com/Alebuenoaz/LSTM-and-GRU-Time-Series-Forecasting)
3. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
4. **Cho, K., van Merriënboer, B., Gulcehre, C., et al.** (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP 2014.
5. **Elman, J. L.** (1990). *Finding Structure in Time.* Cognitive Science, 14(2), 179–211.


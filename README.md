# 📊 LongHorinzon: Benchmark for Long-Horizon Time Series Forecasting

This repository provides a **modular benchmarking framework** for **long-horizon time series forecasting**, evaluated on the **ETT benchmark datasets**:

- ETTh1
- ETTh2
- ETTm1
- ETTm2

It includes implementations of:

- 🔹 **GBT (Gradient Boosting Transformer)**
- 🔹 **TOEformer (Transformer with LRVE)**
- 🔹 **ARIMA (classical baseline)**
- 🔹 **SARIMA (seasonal baseline)**

The project is designed for **research reproducibility**, **fair comparison**, and **easy extensibility**.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/IllianCurvelloFerreira/LongHorinzon.git
cd forecast-benchmark
```
Install dependencies:

```bash
pip install -r requirements.txt
```
# 📁 Project Structure

```bash
forecast-benchmark/
├── data/
│   └── ETT/
│
├── src/
│   ├── data/
│   ├── datasets/
│   ├── models/
│   │   ├── gbt/
│   │   ├── toeformer/
│   │   └── statistical/
│   │       ├── arima.py
│   │       └── sarima.py
│   │
│   ├── training/
│   │   ├── engine_gbt.py
│   │   ├── engine_toeformer.py
│   │   └── engine_statistical.py
│
├── scripts/
│   ├── run_gbt.py
│   ├── run_toeformer.py
│   ├── run_arima_sarima.py
│   └── run_all_models.py
│
├── results/
└── README.md
```

# 📦 Data

Datasets are automatically handled.

If missing, they are:

Downloaded
Preprocessed
Converted to univariate format (date, OT)

No manual setup required.

# ⚙️ Running Models

Basic Example (ETTh1, Horizon=96)

🔷 1. GBT

```bash
!python -m scripts.run_gbt \
  --model GBT \
  --root_path ./data/ETT \
  --data ETTh1 \
  --features S \
  --seq_len 168 \
  --label_len 168 \
  --pred_len 96 \
  --s_layers 3,2,1 \
  --d_layers 2 \
  --itr 5 \
  --learning_rate 0.0001 \
  --dropout 0.05 \
  --fd_model 32 \
  --d_model 512 \
  --criterion Standard \
  --train_epochs 1
```

🔷 2. TOEformer

```bash
python -m scripts.run_toeformer \
  --root_path ./data/ETT \
  --data ETTh1 \
  --target OT \
  --lookback 336 \
  --horizon 96 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --d_model 128 \
  --n_heads 4 \
  --lam2 0.1
```

Without LRVE:

```bash
--lam2 0.0
```

🔷 3. ARIMA / SARIMA

```bash
python -m scripts.run_arima_sarima \
  --root_path ./data/ETT \
  --data ETTh1 \
  --target OT \
  --horizon 96 \
  --stride_mode H \
  --max_origins 5 \
  --models ARIMA SARIMA \
  --sarima_light
```

# 🧠 Research Notes

All models use univariate forecasting (OT)
Same train/val/test split
Rolling-origin evaluation for statistical models
Designed for fair comparison

# 📌 Future Work

Add TimeGPT / foundation models
Hyperparameter search
Ensemble methods
Automated reporting (LaTeX tables)
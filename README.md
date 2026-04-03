# LongHorinzon
# 📊 GBT Benchmark for Long-Horizon Forecasting (ETT)

This repository provides a modular implementation of the **GBT (Gradient Boosting Transformer)** model for **long-horizon time series forecasting**, evaluated on the **ETT benchmark datasets**:

- ETTh1
- ETTh2
- ETTm1
- ETTm2

The project is adapted from a standalone experimental script into a clean and reproducible research structure.

---

# 🚀 Installation

Clone the repository:
```bash
git clone <your-repo-url>
cd forecast-benchmark
```

```bash
pip install -r requirements.txt
```


# ⚙️ Running GBT

Basic Example (ETTh1, Horizon=96)

```bash
python -m scripts/run_gbt.py \
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
  --learning_rate 0.00005 \
  --dropout 0.05 \
  --fd_model 32 \
  --d_model 512 \
  --time \
  --criterion Standard
```
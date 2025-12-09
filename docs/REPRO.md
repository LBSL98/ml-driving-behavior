# Reprodutibilidade — UAH DriveSet (Audit 2025-12-09)

## Pré-requisitos
- Python 3.11, Poetry
- Estrutura de dados:
  - `data/raw/uah_driveset/sensors/...`
  - `data/raw/uah_driveset/metadata/sessions.csv`

## Passos
```bash
# 1) Normalizar sessions
python - << 'PY'
import pandas as pd, os
p_in="data/raw/uah_driveset/metadata/sessions.csv"
df=pd.read_csv(p_in,comment="#")
df.columns=df.columns.str.strip().str.lower()
df=df.rename(columns={"driver":"driver_id","driverid":"driver_id",
                      "session":"session_id","route":"route_type"})
need={"driver_id","session_id","route_type","label","accel_path","gps_path"}
miss=need-set(df.columns)
assert not miss, f"Faltando colunas: {miss}"
os.makedirs("data",exist_ok=True)
df.to_csv("data/uah_sessions.csv",index=False)
print("[OK] data/uah_sessions.csv")
PY

# 2) Extrair features (10 s, 50% overlap)
poetry run python src/01_extract_features.py \
  --sessions data/uah_sessions.csv \
  --root data/raw/uah_driveset/sensors \
  --out data/features/uah_v1_win10_overlap50.parquet \
  --win 10 --overlap 0.5

# 3) Treinar baselines (LODO por motorista)
poetry run python src/02_train_baselines.py \
  --features data/features/uah_v1_win10_overlap50.parquet \
  --group driver --outdir results_audit --n_jobs 4 --balanced --by_route

# 4) PR-AUC por classe
poetry run python src/03_pr_curves.py \
  --features data/features/uah_v1_win10_overlap50.parquet \
  --outdir results_audit

# 5) Fairness por motorista/rota
poetry run python src/06_fairness_by_driver_route.py \
  --features data/features/uah_v1_win10_overlap50.parquet \
  --preds results_audit/preds_rf_driver_win10_bal.csv \
  --outdir results_audit

# 6) Comparar 5s vs 10s (gera gráfico e CSV)
poetry run python src/05_plot_baselines.py \
  --summary5 results_audit/summary_driver_win5_bal.csv \
  --summary10 results_audit/summary_driver_win10_bal.csv \
  --label5 5s --label10 10s \
  --outdir results_audit

# 7) Sweep de threshold e aplicação de τ em motorway
poetry run python src/07_threshold_sweep.py \
  --features data/features/uah_v1_win10_overlap50.parquet \
  --outdir results_audit --n_jobs 4
# aplica melhor τ encontrado
poetry run python src/07_apply_best_tau.py \
  --preds results_audit/preds_rf_driver_win10_bal.csv \
  --sweep results_audit/threshold_sweep_motorway.csv \
  --outdir results_audit

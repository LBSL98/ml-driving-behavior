# Data layout & policy
- raw/: dados originais (NÃO versionar)
- interim/: artefatos temporários (janelas, checkpoints)
- features/: tabelas finais de features por janela (.parquet)
- uah_driveset/: dataset-alvo (IMU+GPS)
NUNCA commitar raw/; versionar apenas metadados (CSV/YAML) e features.

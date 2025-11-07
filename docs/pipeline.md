# Pipeline

- Segmentação: janelas 5s, overlap 50%
- Features: Tempo, Frequência (FFT), Hjorth
- Redução: Baseline | PCA | LDA | PCA→LDA
- Modelos: RF, KNN, DT, NB
- Split: LODO (+ GroupKFold interno)


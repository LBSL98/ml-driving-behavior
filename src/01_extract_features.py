from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
from scipy.signal import welch
from scipy.stats import iqr, median_abs_deviation

import numpy as np
try:
    from numpy import trapezoid as _trapz
except Exception:
    _trapz = np.trapezoid


# ==================== I/O ====================

def load_whitespace_txt(path: Path) -> np.ndarray:
    try:
        # compatível com o aviso do pandas (usa regex de espaço)
        return pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python").values
    except Exception as e:
        print(f"[load] fail {path}: {e}", file=sys.stderr)
        return np.empty((0,0))

# ==================== Detectores de colunas ====================

def infer_acc_columns(arr: np.ndarray):
    """Retorna (t, triplets: list of (x,y,z)). Aceita [t, flag?, 3k cols]."""
    if arr.size == 0: return None, []
    ncols = arr.shape[1]
    t = arr[:, 0].astype(float)

    # caso 1: (ncols-1)%3==0  -> [t | triplets...]
    # caso 2: (ncols-2)%3==0  -> [t | flag | triplets...]
    start = 1 if (ncols - 1) % 3 == 0 else (2 if (ncols - 2) % 3 == 0 else None)
    if start is None or ncols - start < 3:
        return t, []

    rest = arr[:, start:]
    triplets = []
    for k in range(0, rest.shape[1], 3):
        if k + 3 <= rest.shape[1]:
            triplets.append(rest[:, k:k+3].astype(float))
    return t, triplets

def plaus(speed):
    return (speed >= -5) & (speed <= 250)

def infer_gps_columns(arr: np.ndarray):
    """Retorna t, speed(km/h), lat, lon (ou None se não der)."""
    if arr.size == 0: return None, None, None, None
    t = arr[:, 0].astype(float)
    # tentativa direta: [t, speed, lat, lon, ...]
    if arr.shape[1] >= 4:
        s1 = arr[:,1].astype(float)
        la1 = arr[:,2].astype(float)
        lo1 = arr[:,3].astype(float)
        if plaus(s1).mean() > 0.7 and np.all(np.abs(la1) <= 90) and np.all(np.abs(lo1) <= 180):
            return t, s1, la1, lo1
    # fallback: procure col de speed plausível e lat/lon plausíveis
    speed_idx = None
    for j in range(1, arr.shape[1]):
        col = arr[:, j].astype(float)
        if plaus(col).mean() > 0.7:
            speed_idx = j; break
    lat_idx = lon_idx = None
    for j in range(1, arr.shape[1]):
        c = arr[:, j].astype(float)
        if np.all(np.abs(c) <= 90):
            lat_idx = j; break
    for j in range(1, arr.shape[1]):
        if j == lat_idx: continue
        c = arr[:, j].astype(float)
        if np.all(np.abs(c) <= 180):
            lon_idx = j; break
    if speed_idx and lat_idx and lon_idx:
        return t, arr[:,speed_idx].astype(float), arr[:,lat_idx].astype(float), arr[:,lon_idx].astype(float)
    return t, None, None, None

# ==================== Janelas ====================

def sliding_windows(t: np.ndarray, win_s: float, hop_s: float):
    if len(t) < 2: return []
    t0, tN = float(t[0]), float(t[-1])
    starts = np.arange(t0, max(tN - win_s + 1e-9, t0)+1e-9, hop_s)
    idx = []
    for s in starts:
        e = s + win_s + 1e-9
        mask = (t >= s) & (t < e)
        ii = np.where(mask)[0]
        if ii.size >= 5:  # pelo menos 5 amostras
            idx.append(ii)
    return idx

# ==================== Estatísticas básicas ====================

def rms(x): 
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else np.nan

def jerk_stats(x, t):
    if x.size < 3: return (np.nan, np.nan)
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    dx = np.diff(x)
    j = dx / dt
    return float(np.nanmean(np.abs(j))), float(np.nanstd(np.abs(j)))

def corr_triplet(xyz):
    if xyz.shape[1] != 3 or xyz.shape[0] < 3: return (np.nan, np.nan, np.nan)
    c = np.corrcoef(xyz.T)
    return float(c[0,1]), float(c[0,2]), float(c[1,2])

def p95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    return float(np.nanpercentile(x, 95.0))

def bandpower_ratio(sig, fs, lo=0.1, hi=0.5, fmax=5.0):
    if sig is None: return np.nan
    sig = np.asarray(sig, dtype=float)
    sig = sig[np.isfinite(sig)]
    if sig.size < 8 or not np.isfinite(fs) or fs <= 0: 
        return np.nan
    x = sig - np.nanmean(sig)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0/fs)
    psd = (X.real**2 + X.imag**2)
    band = (freqs >= lo) & (freqs <= hi)
    full = (freqs >= 0.0) & (freqs <= fmax)
    bp = float(np.nansum(psd[band])) if band.any() else 0.0
    fp = float(np.nansum(psd[full])) if full.any() else 0.0
    return (bp / fp) if fp > 0 else np.nan

def jerk_extra_from_mag(mag, t):
    """RMS, skew e kurtosis do jerk(mag)."""
    mag = np.asarray(mag, dtype=float)
    if mag.size < 3: 
        return np.nan, np.nan, np.nan
    dt = np.diff(t).astype(float)
    dt[dt <= 0] = np.nan
    dm = np.diff(mag).astype(float)
    j = dm / dt
    j = j[np.isfinite(j)]
    if j.size < 3:
        return np.nan, np.nan, np.nan
    j_c = j - np.nanmean(j)
    j_rms = float(np.sqrt(np.nanmean(j_c**2)))
    std = float(np.nanstd(j_c)) if np.isfinite(np.nanstd(j_c)) else np.nan
    if not np.isfinite(std) or std == 0.0:
        return j_rms, np.nan, np.nan
    j_skew = float(np.nanmean((j_c / std)**3))
    j_kurt = float(np.nanmean((j_c / std)**4))
    return j_rms, j_skew, j_kurt

# ==================== Features por janela ====================
def _safe_fs(t: np.ndarray) -> float | None:
    if t is None or len(t) < 3: 
        return None
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0: 
        return None
    fs = 1.0 / np.median(dt)
    return float(fs) if np.isfinite(fs) and fs > 0 else None

def _band_energy(vec: np.ndarray, fs: float, band: tuple[float,float]) -> float:
    if vec.size < 16 or fs is None or not np.isfinite(fs): 
        return np.nan
    # remove DC leve
    x = vec - np.nanmean(vec)
    nper = min(len(x), 256)
    if nper < 32:
        return np.nan
    f, Pxx = welch(x, fs=fs, nperseg=nper, detrend='constant', scaling='density')
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return np.nan
    return float(_trapz(Pxx[m], f[m]))

def _lf_ratio(vec: np.ndarray, t: np.ndarray,
              lf=(0.10, 0.50), total=(0.05, 2.00)) -> float:
    fs = _safe_fs(t)
    e_tot = _band_energy(vec, fs, total)
    e_lf  = _band_energy(vec, fs, lf)
    if not np.isfinite(e_tot) or e_tot <= 0 or not np.isfinite(e_lf):
        return np.nan
    return float(e_lf / e_tot)


def feats_xyz(prefix, xyz, t):
    out = {}
    if xyz.shape[1] != 3 or xyz.shape[0] < 3:
        # preenche chaves mínimas esperadas se faltarem amostras
        for name in ("x","y","z","mag"):
            out[f"{prefix}_{name}_mean"] = np.nan
            out[f"{prefix}_{name}_std"]  = np.nan
            out[f"{prefix}_{name}_min"]  = np.nan
            out[f"{prefix}_{name}_max"]  = np.nan
            out[f"{prefix}_{name}_rms"]  = np.nan
            out[f"{prefix}_{name}_iqr"]  = np.nan
            out[f"{prefix}_{name}_mad"]  = np.nan
            out[f"{prefix}_{name}_jerk_mean_abs"] = np.nan
            out[f"{prefix}_{name}_jerk_std_abs"]  = np.nan
            out[f"{prefix}_{name}_p95"] = np.nan
            out[f"{prefix}_{name}_lf_ratio"] = np.nan
        out[f"{prefix}_corr_xy"] = np.nan
        out[f"{prefix}_corr_xz"] = np.nan
        out[f"{prefix}_corr_yz"] = np.nan
        return out

    mag = np.linalg.norm(xyz, axis=1)

    def _per_axis(name, vec):
        out_local = {}
        out_local[f"{prefix}_{name}_mean"] = float(np.mean(vec))
        out_local[f"{prefix}_{name}_std"]  = float(np.std(vec, ddof=0))
        out_local[f"{prefix}_{name}_min"]  = float(np.min(vec))
        out_local[f"{prefix}_{name}_max"]  = float(np.max(vec))
        out_local[f"{prefix}_{name}_rms"]  = rms(vec)
        out_local[f"{prefix}_{name}_iqr"]  = float(iqr(vec)) if vec.size else np.nan
        out_local[f"{prefix}_{name}_mad"]  = float(median_abs_deviation(vec)) if vec.size else np.nan
        jm, js = jerk_stats(vec, t)
        out_local[f"{prefix}_{name}_jerk_mean_abs"] = jm
        out_local[f"{prefix}_{name}_jerk_std_abs"]  = js
        # NOVO: percentil 95 da amplitude absoluta
        out_local[f"{prefix}_{name}_p95"] = float(np.percentile(np.abs(vec), 95)) if vec.size else np.nan
        # NOVO: razão de baixa frequência (captura “quieto com correções lentas”)
        out_local[f"{prefix}_{name}_lf_ratio"] = _lf_ratio(vec, t)
        return out_local

    out.update(_per_axis("x", xyz[:,0]))
    out.update(_per_axis("y", xyz[:,1]))
    out.update(_per_axis("z", xyz[:,2]))
    out.update(_per_axis("mag", mag))

    c = np.corrcoef(xyz.T)
    out[f"{prefix}_corr_xy"] = float(c[0,1])
    out[f"{prefix}_corr_xz"] = float(c[0,2])
    out[f"{prefix}_corr_yz"] = float(c[1,2])
    return out


def feats_gps(prefix, t, speed):
    out = {}
    if speed is None or speed.size < 2:  # sem GPS útil
        out[f"{prefix}_speed_mean"] = np.nan
        out[f"{prefix}_speed_std"]  = np.nan
        out[f"{prefix}_speed_max"]  = np.nan
        out[f"{prefix}_acc_mean_abs"] = np.nan
        out[f"{prefix}_stopped_frac"] = np.nan
        return out
    v = speed
    out[f"{prefix}_speed_mean"] = float(np.mean(v))
    out[f"{prefix}_speed_std"]  = float(np.std(v, ddof=0))
    out[f"{prefix}_speed_max"]  = float(np.max(v))
    # aceleração ~ dv/dt
    dt = np.diff(t); dt[dt <= 0] = np.nan
    dv = np.diff(v)
    a = dv / dt
    out[f"{prefix}_acc_mean_abs"] = float(np.nanmean(np.abs(a)))
    out[f"{prefix}_stopped_frac"] = float(np.mean(v < 1.0))
    return out

# ==================== Pipeline por sessão ====================

def process_session(row, win_s, hop_s, root):
    acc = load_whitespace_txt(root/row["accel_path"])
    gps = load_whitespace_txt(root/row["gps_path"])

    t_acc, triplets = infer_acc_columns(acc)
    t_gps, speed, lat, lon = infer_gps_columns(gps)

    if t_acc is None or len(triplets)==0:
        print(f"[warn] sem ACC utilizável: {row['accel_path']}", file=sys.stderr); return []

    # janelas pelo tempo do ACC (mais denso); GPS é agregado na mesma janela por tempo
    idx_list = sliding_windows(t_acc, win_s, hop_s)
    out_rows = []
    for ii in idx_list:
        t_w = t_acc[ii]
        feats = {
            "driver_id": int(row["driver_id"]),
            "session_id": row["session_id"],
            "label": row["label"],
            "route_type": row["route_type"],
            "t_start": float(t_w[0]),
            "t_end": float(t_w[-1]),
            "s0": int(ii[0]),            # <<< NOVO: índice inicial (ACC)
            "s1": int(ii[-1]),           # <<< NOVO: índice final   (ACC)
            "n_samples": int(len(ii)),
            "dt_median": float(np.median(np.diff(t_w))) if len(ii)>1 else np.nan,
        }
        # ACC triplets
        for k, xyz_full in enumerate(triplets):
            xyz = xyz_full[ii, :]
            feats.update(feats_xyz(f"acc{ k }", xyz, t_w))

        # GPS na mesma janela (mas alinhando por tempo absoluto aproximado)
        if t_gps is not None and speed is not None and len(t_gps) >= 2:
            m = (t_gps >= feats["t_start"]) & (t_gps < feats["t_end"])
            if m.any():
                feats.update(feats_gps("gps", t_gps[m], speed[m]))
            else:
                feats.update(feats_gps("gps", t_gps, None))
        else:
            feats.update(feats_gps("gps", None, None))
        out_rows.append(feats)
    return out_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--root", default="data/raw/uah_driveset/sensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    hop = args.win * (1.0 - args.overlap)

    # 1) lê o CSV ignorando linhas comentadas e normaliza cabeçalho
    meta = pd.read_csv(args.sessions, comment="#")
    meta.columns = meta.columns.str.strip().str.lower()
    # aceita alguns sinônimos se existirem
    meta = meta.rename(columns={
        "driver": "driver_id",
        "driverid": "driver_id",
        "session": "session_id",
        "route": "route_type"
    })
    required = ["driver_id","session_id","route_type","label","accel_path","gps_path"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise SystemExit(f"[error] sessions.csv sem colunas obrigatórias: {missing}\nHeader lido: {list(meta.columns)}")

    rows = []
    root = Path(args.root)
    for _, r in meta.iterrows():
        # cast seguro + strip
        r = r.copy()
        r["driver_id"] = int(str(r["driver_id"]).strip())
        for c in ("session_id","route_type","label","accel_path","gps_path","video_path"):
            if c in r:
                r[c] = str(r[c]).strip()
        rows.extend(process_session(r, args.win, hop, root))

    if not rows:
        print("[error] nenhum feature gerado", file=sys.stderr); sys.exit(2)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(df.groupby("label").size().rename("windows").to_string())
    print("saved:", args.out)

if __name__ == "__main__":
    main()

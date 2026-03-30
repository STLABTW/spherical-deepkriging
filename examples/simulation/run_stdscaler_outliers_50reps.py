#!/usr/bin/env python
"""
Resumable 50-rep experiment: StdScaler + clipnorm fix for dk_sphere.
Runs as a plain Python script (no Jupyter kernel) to avoid OOM crashes.
Checkpoints after every repeat — safe to kill and re-run.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Change working dir so relative CSV path works ────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

warnings.filterwarnings('ignore', message='.*input_shape.*')
warnings.filterwarnings('ignore', message='.*structure of.*inputs.*')

import time, gc, random
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import kv, gamma

import jax, jax.numpy as jnp

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ.update({"TF_CPP_MIN_LOG_LEVEL": "2"})
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")

np_f32     = np.float32
jnp_f32    = jnp.float32
dtype_basis = np.float32
jax.config.update("jax_enable_x64", False)

def init_hardware(dtype="float32"):
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    mixed_precision.set_global_policy(dtype)
    return strategy

strategy = init_hardware()

from spherical_deepkriging.basis_functions.wendland.wenland import wendland
from spherical_deepkriging.basis_functions.mrts.mrts import mrts0
from spherical_deepkriging.models.deep_kriging import DeepKrigingTrainer, DeepKrigingDefaultTrainer
from spherical_deepkriging.configs import DeepKrigingModelConfig, DeepKrigingDefaultConfig
from spherical_deepkriging.models.universal_kriging import UniversalKriging
from rpy2.robjects.conversion import localconverter
from spherical_deepkriging.basis_functions.mrts_sphere.sphere import mrts_sphere, numpy2ri_converter

# ── Experiment parameters ─────────────────────────────────────────────────────
SEED_BASE           = 50
OUTLIER_RATIO       = 0.025
OUTLIER_MULTIPLIER  = 5
EPOCHS           = 500
BATCH_SIZE       = 256
NUM_SAMPLE       = 2500
HUBER_DELTA      = 1.345
NUM_BASIS        = [10**2, 19**2, 37**2]
KNOT_NUM         = 1400
ORDER_MAX        = 1400
BASE_ORDERS      = [10, 50, 100, 150, 200, 1000]
REPEAT_TOTAL     = 50
CKPT_PATH        = "results_sphere_stdscaler_outliers_50reps_checkpoint.csv"
ALL_MODELS       = ["OLS_wendland", "OLS_sphere", "DeepKriging_wendland",
                    "DeepKriging_mrts", "DeepKriging_sphere",
                    "DeepKriging_sphere_Huber", "UniversalKriging"]


# ── Resume: find already-completed repeats ────────────────────────────────────
def completed_repeats():
    if not os.path.exists(CKPT_PATH):
        return set()
    df = pd.read_csv(CKPT_PATH)
    # A repeat is complete only if all 7 models are present
    counts = df.groupby('Repeat')['Model'].nunique()
    return set(counts[counts == len(ALL_MODELS)].index.tolist())


# ── Data simulation ───────────────────────────────────────────────────────────
def simulate_data(num_sample, outlier_ratio, outlier_multiplier, seed):
    rng = np.random.default_rng(seed)
    c, a, b = 1.5, 2.0, 1.0
    phi   = rng.uniform(0, 2 * np.pi, num_sample)
    theta = np.arccos(rng.uniform(-1, 1, num_sample))
    lat_rad = np.pi / 2 - theta
    lon_rad = phi - np.pi
    x_c = np.cos(lat_rad) * np.cos(lon_rad)
    y_c = np.cos(lat_rad) * np.sin(lon_rad)
    z_c = np.sin(lat_rad)
    coords = np.column_stack([x_c, y_c, z_c]).astype(np_f32)
    mean_trend = (5.0
        + 18.0 * np.exp(-((lat_rad - np.pi/4)**2) / 0.05)
        + 22.0 * np.exp(-((lat_rad + np.pi/4)**2) / 0.04)
        - 4.0  * np.exp(-(lat_rad**2) / 0.01)
        + np.where((lon_rad > 0) & (lon_rad < 1) & (lat_rad > 0.1) & (lat_rad < 1), -12.0, 0.0))
    anomaly_lats = np.arcsin(rng.uniform(-1, 1, 60))
    anomaly_lons = rng.uniform(-np.pi, np.pi, 60)
    ax = np.cos(anomaly_lats) * np.cos(anomaly_lons)
    ay = np.cos(anomaly_lats) * np.sin(anomaly_lons)
    az = np.sin(anomaly_lats)
    anomaly_effect = np.zeros(num_sample)
    for i in range(60):
        sq = (x_c-ax[i])**2 + (y_c-ay[i])**2 + (z_c-az[i])**2
        anomaly_effect += rng.uniform(-10, 18) * np.exp(-sq / rng.uniform(0.005, 0.03))
    mean_trend = np.maximum(mean_trend + anomaly_effect, 0.5)
    dist_matrix = np.arccos(np.clip(coords @ coords.T, -1, 1))
    cov = (np.exp(-dist_matrix / c)).astype(np_f32)
    cov += np.float32(1e-3) * np.eye(num_sample, dtype=np_f32)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.float32(1e-2) * np.eye(num_sample, dtype=np_f32)
        ev, evec = np.linalg.eigh(cov)
        L = evec @ np.diag(np.sqrt(np.maximum(ev, 1e-6)))
    xi = (L @ rng.standard_normal(num_sample).astype(np_f32)).astype(np_f32)
    xi *= 1.0 + 1.5 * np.exp(-((np.abs(lat_rad) - np.pi/4)**2) / 0.1)
    gn = (a/b) * np.power(np.maximum(1 - 1/(9*a) + xi*np.sqrt(1/(9*a)), 0), 3)
    gn -= np.mean(gn)
    z = np.maximum(mean_trend + gn, 0.0)
    idx_outliers = rng.choice(num_sample, size=int(num_sample * outlier_ratio), replace=False)
    z[idx_outliers] *= outlier_multiplier
    del coords, dist_matrix, cov, L, xi, gn, mean_trend, anomaly_effect; gc.collect()
    return pd.DataFrame({"longitude": np.rad2deg(lon_rad).astype(np_f32),
                         "latitude":  np.rad2deg(lat_rad).astype(np_f32), "z": z})


def data_preprocessing(df):
    data = df.copy()
    data[["longitude","latitude","z"]] = data[["longitude","latitude","z"]].apply(pd.to_numeric, errors="coerce")
    data.dropna(subset=["longitude","latitude","z"], inplace=True)
    lon, lat = data["longitude"].to_numpy(), data["latitude"].to_numpy()
    norm_lon = (lon - lon.min()) / (lon.max() - lon.min())
    norm_lat = (lat - lat.min()) / (lat.max() - lat.min())
    return (np.column_stack([lat, lon]).astype("float32"),
            np.column_stack([norm_lon, norm_lat]).astype("float32"),
            None,
            data['z'].to_numpy().astype("float32")[:, None])


def precompute_wendland(location, num_basis):
    parts = []
    for nb in num_basis:
        grid = np.column_stack(np.meshgrid(
            np.linspace(0,1,int(np.sqrt(nb)),dtype=np_f32),
            np.linspace(0,1,int(np.sqrt(nb)),dtype=np_f32))).reshape(-1,2).astype(np_f32)
        parts.append(wendland(location, grid, theta=np_f32(2.5)/np_f32(np.sqrt(nb)), k=2))
        del grid; gc.collect()
    return np.hstack(parts).astype(dtype_basis, copy=False)


def precompute_max_mrts(distance_type, location_data, knot_num, order_max, knot=None):
    if knot is None:
        idx_knot = np.random.choice(location_data.shape[0], knot_num, replace=False)
        knot = location_data[idx_knot].astype(np_f32)
    else:
        idx_knot = None
    if distance_type == "sphere":
        with localconverter(numpy2ri_converter):
            res_r = mrts_sphere(knot, order_max, location_data.astype(np_f32))
        res_dict = dict(zip(res_r.names(), res_r))
        phi = np.asarray(res_dict["mrts"], dtype=dtype_basis)
    else:
        phi = np.asarray(mrts0(jnp.asarray(knot, dtype=jnp_f32), k=order_max,
                               x=jnp.asarray(location_data, dtype=jnp_f32)), dtype=dtype_basis)
    return phi, idx_knot, knot


def prepare_data(categorical_data, basis, y_combined, seed, split_ratio=(0.8, 0.1, 0.1)):
    idx_all = np.arange(basis.shape[0])
    tr_r, va_r, _ = split_ratio
    tv_idx, te_idx = train_test_split(idx_all, train_size=tr_r+va_r, random_state=seed)
    tr_idx, va_idx = train_test_split(tv_idx, train_size=tr_r/(tr_r+va_r), random_state=seed)
    fl_y = lambda t: t.reshape(-1).astype(np_f32, copy=False)
    fl_x = lambda c: c.reshape(-1, basis.shape[1]).astype(np_f32)
    zv   = lambda n: np.zeros((n, 0), dtype=np_f32)
    return (fl_x(basis[tr_idx]), zv(len(tr_idx)), fl_y(y_combined[tr_idx]),
            fl_x(basis[va_idx]), zv(len(va_idx)), fl_y(y_combined[va_idx]),
            fl_x(basis[te_idx]), zv(len(te_idx)), fl_y(y_combined[te_idx]))


def cleanup_tf_session():
    K.clear_session(); gc.collect()
    try: tf.keras.backend.clear_session()
    except: pass


def train_eval(name_model, epochs, batch_size, loss, dropout_rate,
               X_train, X_train_cat, y_train,
               X_val,   X_val_cat,   y_val,
               X_test,  X_test_cat,  y_test):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    if name_model in ["OLS_wendland", "OLS_sphere"]:
        t0 = time.time()
        m = LinearRegression().fit(X_train, y_train)
        vl = float(mean_squared_error(y_val, m.predict(X_val)))
        yp = m.predict(X_test).astype(np_f32).reshape(-1)
        return {"Model": name_model, "Val_loss": vl,
                "MSPE": float(mean_squared_error(y_test, yp)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, yp))),
                "MAE":  float(mean_absolute_error(y_test, yp)),
                "R2":   float(r2_score(y_test, yp)),
                "Time": float(time.time()-t0)}, m

    elif name_model == "DeepKriging_wendland":
        config = DeepKrigingDefaultConfig(
            input_dim=X_train.shape[1], output_type='continuous',
            optimizer=Adam(learning_rate=1e-3), loss=loss,
            epochs=epochs, batch_size=batch_size, verbose=0)
    elif name_model in ["DeepKriging_mrts", "DeepKriging_sphere", "DeepKriging_sphere_Huber"]:
        # clipnorm=1.0 for sphere models prevents gradient explosion
        _opt = Adam(learning_rate=5e-3, clipnorm=1.0) if "sphere" in name_model else Adam(learning_rate=5e-3)
        config = DeepKrigingModelConfig(
            input_dim=X_train.shape[1], output_type='continuous',
            hidden_layers=[1024,512,256,128,64], activation='relu',
            dropout_rate=dropout_rate, optimizer=_opt,
            loss=loss, metrics=['mae'], epochs=epochs, batch_size=batch_size,
            patience=40, verbose=0)

    t0 = time.time()
    with strategy.scope():
        m = DeepKrigingDefaultTrainer(config) if name_model == "DeepKriging_wendland" else DeepKrigingTrainer(config)
        m.model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

    ckpt = f"best_{name_model}_{time.time_ns()}.weights.h5"
    cbs  = [tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", mode="min",
                save_best_only=True, save_weights_only=True, verbose=0)]
    if name_model != "DeepKriging_wendland":
        cbs += [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.patience,
                                             restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=max(5, config.patience//2),
                                                 min_lr=1e-6, verbose=0)]
    tr_ds = tf.data.Dataset.from_tensor_slices(((X_train, X_train_cat), y_train))\
              .batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices(((X_val,   X_val_cat),   y_val))\
              .batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    hist  = m.model.fit(tr_ds, validation_data=va_ds, epochs=epochs, callbacks=cbs, verbose=0)
    if os.path.exists(ckpt):
        m.model.load_weights(ckpt); os.remove(ckpt)
    vl = float(np.min(hist.history["val_loss"]))
    yp = m.model.predict([X_test, X_test_cat], verbose=0).reshape(-1).astype(np_f32)
    del tr_ds, va_ds; gc.collect()
    return {"Model": name_model, "Val_loss": vl,
            "MSPE": float(mean_squared_error(y_test, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, yp))),
            "MAE":  float(mean_absolute_error(y_test, yp)),
            "R2":   float(r2_score(y_test, yp)),
            "Time": float(time.time()-t0)}, m


# ── Main experiment loop ──────────────────────────────────────────────────────
done = completed_repeats()
print(f"Already completed repeats: {sorted(done)}")
print(f"Remaining: {REPEAT_TOTAL - len(done)}")

for repeat in range(REPEAT_TOTAL):
    rep_num = repeat + 1
    if rep_num in done:
        print(f"  Skipping repeat {rep_num} (already in checkpoint)")
        continue

    seed = SEED_BASE + repeat

    print(f"\n{'='*80}")
    print(f"Repeat {rep_num}/{REPEAT_TOTAL}, Seed={seed}")
    print(f"{'='*80}")

    best_orders = {}
    dataframe = simulate_data(NUM_SAMPLE, OUTLIER_RATIO, OUTLIER_MULTIPLIER, seed)
    location_data, location_data_norm, categorical_data, y_combined = data_preprocessing(dataframe)

    # Sphere basis
    max_Phi_sphere, idx_knot, knot = precompute_max_mrts(
        "sphere", location_data, KNOT_NUM, ORDER_MAX, knot=None)
    max_Phi_sphere = max_Phi_sphere.astype(dtype_basis, copy=False)

    # FIX: StandardScaler fitted on train split
    _tv, _ = train_test_split(np.arange(max_Phi_sphere.shape[0]), train_size=0.9, random_state=seed)
    _tr, _ = train_test_split(_tv, train_size=8/9, random_state=seed)
    _scaler = StandardScaler()
    _scaler.fit(max_Phi_sphere[_tr])
    max_Phi_sphere = _scaler.transform(max_Phi_sphere).astype(dtype_basis)
    print(f"Sphere basis after StandardScaler: mean={max_Phi_sphere.mean():.4f}, std={max_Phi_sphere.std():.4f}")

    # Euclidean MRTS basis
    max_Phi_mrts, _, _ = precompute_max_mrts(
        "mrts", location_data, KNOT_NUM, ORDER_MAX, knot=location_data[idx_knot])
    max_Phi_mrts = max_Phi_mrts.astype(dtype_basis, copy=False)

    # Wendland basis
    Phi_wendland = precompute_wendland(location_data_norm, NUM_BASIS)

    # ── TUNING ────────────────────────────────────────────────────────────────
    def tune(name, basis_fn, model_key, loss_fn, use_strategy=True):
        best_val, best_order, results = float('inf'), None, []
        print(f"\nTuning {name}")
        for order in BASE_ORDERS:
            phi = basis_fn(order)
            parts = prepare_data(categorical_data, phi, y_combined, seed)
            if use_strategy:
                with strategy.scope():
                    mt, md = train_eval(name, EPOCHS, BATCH_SIZE, loss_fn, 0.01, *parts)
                cleanup_tf_session()
            else:
                mt, md = train_eval(name, None, None, None, None, *parts)
                del md
            results.append({'order': order, 'val_loss': mt["Val_loss"], 'mspe': mt["MSPE"]})
            if mt["Val_loss"] < best_val:
                best_val = mt["Val_loss"]; best_order = order
            del phi, parts; gc.collect()
        print(f"   {'Order':<10} {'Val Loss':<12} {'Test MSE':<12}")
        for r in results:
            mk = " *" if r['order'] == best_order else ""
            print(f"   {r['order']:<10} {r['val_loss']:<12.4f} {r['mspe']:<12.4f}{mk}")
        print(f"   Best order: {best_order}")
        best_orders[model_key] = best_order

    tune("OLS_sphere",              lambda o: max_Phi_sphere[:,:o].astype(np_f32), 'OLS_sphere',              None, use_strategy=False)
    tune("DeepKriging_mrts",        lambda o: max_Phi_mrts[:,:o].astype(np_f32),   'DeepKriging_mrts',        "mse")
    tune("DeepKriging_sphere",      lambda o: max_Phi_sphere[:,:o].astype(np_f32), 'DeepKriging_sphere',      "mse")
    tune("DeepKriging_sphere_Huber",lambda o: max_Phi_sphere[:,:o].astype(np_f32), 'DeepKriging_sphere_Huber',Huber(delta=HUBER_DELTA))

    # UniversalKriging tuning
    best_val_uk, best_order_uk = float('inf'), None
    print("\nTuning UniversalKriging")
    for order in BASE_ORDERS:
        phi = max_Phi_sphere[:, :order].astype(np_f32)
        idx_all = np.arange(phi.shape[0])
        tv, te = train_test_split(idx_all, train_size=0.9, random_state=seed)
        tr, va = train_test_split(tv, train_size=8/9, random_state=seed)
        uk = UniversalKriging(num_neighbors=30, cov_function='exponential')
        uk.fit(location_data[tr], phi[tr], y_combined[tr].flatten(), center_y=True)
        yv = uk.predict(location_data[va], phi[va], return_centered=True)
        vl = mean_squared_error(y_combined[va].flatten() - uk.y_mean, yv)
        print(f"   {order:<10} {vl:<12.4f} {'*' if vl < best_val_uk else ''}")
        if vl < best_val_uk:
            best_val_uk = vl; best_order_uk = order
        uk.cleanup(); del uk, phi; gc.collect()
    print(f"   Best order: {best_order_uk}")
    best_orders['UniversalKriging'] = best_order_uk

    # ── EVALUATION ────────────────────────────────────────────────────────────
    Record = {}

    # OLS_wendland
    mt, md = train_eval("OLS_wendland", None, None, None, None,
                        *prepare_data(categorical_data, Phi_wendland, y_combined, seed))
    Record["OLS_wendland"] = {**{k: mt[k] for k in ("MSPE","RMSE","MAE","R2","Time")}, "Param": "--"}
    del md; gc.collect()

    # OLS_sphere
    phi = max_Phi_sphere[:, :best_orders['OLS_sphere']].astype(np_f32)
    mt, md = train_eval("OLS_sphere", None, None, None, None,
                        *prepare_data(categorical_data, phi, y_combined, seed))
    Record["OLS_sphere"] = {**{k: mt[k] for k in ("MSPE","RMSE","MAE","R2","Time")}, "Param": best_orders['OLS_sphere']}
    del md, phi; gc.collect()

    # DeepKriging_wendland
    with strategy.scope():
        mt, md = train_eval("DeepKriging_wendland", EPOCHS, BATCH_SIZE, "mse", 0.01,
                            *prepare_data(categorical_data, Phi_wendland, y_combined, seed))
    Record["DeepKriging_wendland"] = {**{k: mt[k] for k in ("MSPE","RMSE","MAE","R2","Time")}, "Param": "--"}
    del md; cleanup_tf_session()

    # DeepKriging_mrts
    phi = max_Phi_mrts[:, :best_orders['DeepKriging_mrts']].astype(np_f32)
    with strategy.scope():
        mt, md = train_eval("DeepKriging_mrts", EPOCHS, BATCH_SIZE, "mse", 0.01,
                            *prepare_data(categorical_data, phi, y_combined, seed))
    Record["DeepKriging_mrts"] = {**{k: mt[k] for k in ("MSPE","RMSE","MAE","R2","Time")}, "Param": best_orders['DeepKriging_mrts']}
    del md, phi; cleanup_tf_session()

    # DeepKriging_sphere
    phi = max_Phi_sphere[:, :best_orders['DeepKriging_sphere']].astype(np_f32)
    with strategy.scope():
        mt, md = train_eval("DeepKriging_sphere", EPOCHS, BATCH_SIZE, "mse", 0.01,
                            *prepare_data(categorical_data, phi, y_combined, seed))
    Record["DeepKriging_sphere"] = {**{k: mt[k] for k in ("MSPE","RMSE","MAE","R2","Time")}, "Param": best_orders['DeepKriging_sphere']}
    del md, phi; cleanup_tf_session()

    # DeepKriging_sphere_Huber
    phi = max_Phi_sphere[:, :best_orders['DeepKriging_sphere_Huber']].astype(np_f32)
    with strategy.scope():
        mt, md = train_eval("DeepKriging_sphere_Huber", EPOCHS, BATCH_SIZE, Huber(delta=HUBER_DELTA), 0.01,
                            *prepare_data(categorical_data, phi, y_combined, seed))
    Record["DeepKriging_sphere_Huber"] = {**{k: mt[k] for k in ("MSPE","RMSE","MAE","R2","Time")}, "Param": best_orders['DeepKriging_sphere_Huber']}
    del md, phi; cleanup_tf_session()

    # UniversalKriging
    t0 = time.time()
    phi = max_Phi_sphere[:, :best_orders['UniversalKriging']].astype(np_f32)
    tv, te = train_test_split(np.arange(phi.shape[0]), train_size=0.9, random_state=seed)
    tr, _  = train_test_split(tv, train_size=8/9, random_state=seed)
    uk = UniversalKriging(num_neighbors=30, cov_function='exponential')
    uk.fit(location_data[tr], phi[tr], y_combined[tr].flatten(), center_y=True)
    yp = uk.predict(location_data[te], phi[te], return_centered=False)
    Record["UniversalKriging"] = {
        "MSPE": mean_squared_error(y_combined[te].flatten(), yp),
        "RMSE": np.sqrt(mean_squared_error(y_combined[te].flatten(), yp)),
        "MAE":  mean_absolute_error(y_combined[te].flatten(), yp),
        "R2":   r2_score(y_combined[te].flatten(), yp),
        "Time": time.time()-t0, "Param": best_orders['UniversalKriging']}
    uk.cleanup(); del uk, phi; gc.collect()

    # Print results table
    rows = []
    for mn in ALL_MODELS:
        rows.append({"Model": mn, "Param": Record[mn]["Param"],
                     "MSPE": f"{Record[mn]['MSPE']:.4f}", "RMSE": f"{Record[mn]['RMSE']:.4f}",
                     "MAE":  f"{Record[mn]['MAE']:.4f}",  "R2":   f"{Record[mn]['R2']:.4f}",
                     "Time": f"{Record[mn]['Time']:.1f}s"})
    print("\n", pd.DataFrame(rows).to_markdown(index=False, tablefmt="github"), sep="")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_rows = [{"Model": mn, "Repeat": rep_num,
                  "MSPE": Record[mn]["MSPE"], "RMSE": Record[mn]["RMSE"],
                  "MAE":  Record[mn]["MAE"],  "R2":   Record[mn]["R2"]}
                 for mn in ALL_MODELS]
    ckpt_df = pd.DataFrame(ckpt_rows)
    if os.path.exists(CKPT_PATH):
        ckpt_df.to_csv(CKPT_PATH, mode='a', header=False, index=False)
    else:
        ckpt_df.to_csv(CKPT_PATH, index=False)
    print(f"Checkpoint saved -> {CKPT_PATH}  (repeat {rep_num}/{REPEAT_TOTAL})")

    del Phi_wendland, max_Phi_sphere, max_Phi_mrts, dataframe, location_data, location_data_norm
    cleanup_tf_session(); gc.collect()
    print(f"Completed repeat {rep_num}/{REPEAT_TOTAL}")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL SUMMARY — 50 repeats, Outliers (2.5%x5), StdScaler + clipnorm=1.0 for sphere")
print("="*80)
df = pd.read_csv(CKPT_PATH)
avg = []
for mn in ALL_MODELS:
    sub = df[df.Model == mn]
    avg.append({"Model": mn,
                "MSPE": f"{sub.MSPE.mean():.3f}±{sub.MSPE.std():.3f}",
                "RMSE": f"{sub.RMSE.mean():.3f}±{sub.RMSE.std():.3f}",
                "MAE":  f"{sub.MAE.mean():.3f}±{sub.MAE.std():.3f}",
                "R2":   f"{sub.R2.mean():.3f}±{sub.R2.std():.3f}",
                "Bad(MSPE>5)": int((sub.MSPE > 5).sum())})
print("\n", pd.DataFrame(avg).to_markdown(index=False, tablefmt="github"), sep="")

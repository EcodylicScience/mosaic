import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch import amp

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import time

from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, f1_score

###########################################################################
######################### TCN Params
###########################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
BATCH_SIZE = 8192
EPOCHS = 25
PATIENCE = 5                # early stopping on macro-F1 (val)
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
WIDTH = 256                 # hidden channels in TCN
KERNEL_SIZE = 5             # 1D conv kernel
DILATIONS = [1, 2, 4, 8, 16]
PRINT_FRAC = 0.1            # print ~every 10% of an epoch
VAL_FRAC = 0.1              # take 10% of TRAIN for threshold tuning/early stopping
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

BASE_FEATS = 1
###########################################################################
######################### XGB Params
###########################################################################
RANDOM_STATE = 42
# first ones I tried
XGB_PARAMS_0 = dict(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda", # GPU acceleration
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,
    random_state=123,
    eval_metric="logloss",
)

# deeper but regularized
XGB_PARAMS_1 = dict(
    objective="binary:logistic",
    device="cuda",
    tree_method="hist",
    eval_metric=["logloss","aucpr"],

    n_estimators=3000,
    learning_rate=0.03,

    max_depth=10,
    min_child_weight=8,     # regularize splits
    gamma=2.0,              # further split penalty

    subsample=0.7,
    colsample_bytree=0.6,
    colsample_bylevel=0.8,

    reg_lambda=2.0,
    reg_alpha=0.5,

    max_bin=256,            # GPU hist binning (try 256/512)
    sampling_method="gradient_based",
    n_jobs=-1,
    random_state=123,
)

# shallow trees, many estimators
XGB_PARAMS_2 = dict(
    objective="binary:logistic",
    device="cuda",
    tree_method="hist",
    eval_metric=["logloss","aucpr"],

    n_estimators=4000,
    learning_rate=0.02,

    max_depth=4,
    min_child_weight=3,
    gamma=0.0,

    subsample=0.9,
    colsample_bytree=0.9,

    reg_lambda=1.0,
    reg_alpha=0.0,

    max_bin=256,
    sampling_method="gradient_based",
    n_jobs=-1,
    random_state=123,
)

# leaf-wise growth
XGB_PARAMS_3 = dict(
    objective="binary:logistic",
    device="cuda",
    tree_method="hist",
    eval_metric=["logloss","aucpr"],

    n_estimators=3000,
    learning_rate=0.03,

    grow_policy="lossguide",
    max_depth=0,         # ignored with lossguide
    max_leaves=64,       # tune 32–256
    min_child_weight=6,
    gamma=1.0,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.5,
    reg_alpha=0.25,

    max_bin=512,
    sampling_method="gradient_based",
    n_jobs=-1,
    random_state=123,
)

# class StackedMultiLabelDataset(Dataset):
#     def __init__(self, X_flat: np.ndarray, Y: np.ndarray, time_steps, has_pool, pooled_dim):
#         self.X = X_flat.astype(np.float32, copy=False)
#         self.Y = Y.astype(np.float32,   copy=False)
#         self.time_steps = time_steps
#         self.has_pool = has_pool
#         self.pooled_dim = pooled_dim

#     def __len__(self):
#         return self.Y.shape[0]

#     def __getitem__(self, idx):
#         x = self.X[idx]
#         # temporal part -> (F, T)
#         temporal_flat_dim = BASE_FEATS * self.time_steps
#         x_temporal = x[:temporal_flat_dim].reshape(self.time_steps, BASE_FEATS).T  # (F, T)

#         if self.has_pool and self.pooled_dim > 0:
#             n_stats = self.pooled_dim // BASE_FEATS
#             off = temporal_flat_dim
#             pooled_ch = []
#             for _ in range(n_stats):
#                 chan = x[off:off+BASE_FEATS]               # (F,)
#                 pooled_ch.append(np.tile(chan[:, None], (1, self.time_steps)))
#                 off += BASE_FEATS
#             x_full = np.concatenate([x_temporal] + pooled_ch, axis=0)  # (F*(1+n_stats), T)
#         else:
#             x_full = x_temporal

#         return torch.from_numpy(x_full), torch.from_numpy(self.Y[idx])

class FlatStackedDataset(torch.utils.data.Dataset):
    def __init__(self, X_flat: np.ndarray, Y: np.ndarray):
        self.X = X_flat.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32,   copy=False)
    def __len__(self): return self.Y.shape[0]
    def __getitem__(self, idx):
        # return flat row + label; collate will reshape/broadcast for the whole batch
        return self.X[idx], self.Y[idx]

def make_collate_fn(time_steps: int, base_feats: int, has_pool: bool, pooled_dim: int):
    temporal_flat = base_feats * time_steps
    n_stats = (pooled_dim // base_feats) if (has_pool and pooled_dim > 0) else 0

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.from_numpy(np.stack(xs, axis=0))  # (B, feat_total), CPU
        y = torch.from_numpy(np.stack(ys, axis=0))  # (B, C)

        B = x.size(0)
        # temporal -> (B, F, T)
        x_temporal = x[:, :temporal_flat].view(B, time_steps, base_feats).permute(0, 2, 1).contiguous()

        if n_stats > 0:
            # (B, n_stats, F)
            pooled = x[:, temporal_flat:temporal_flat + n_stats*base_feats].view(B, n_stats, base_feats)
            # expand to (B, n_stats*F, T) without materializing tiles
            pooled = pooled.view(B, n_stats * base_feats, 1).expand(B, n_stats * base_feats, time_steps)
            x_full = torch.cat([x_temporal, pooled], dim=1)  # (B, F*(1+n_stats), T)
        else:
            x_full = x_temporal
        return x_full, y
    return collate

# helper: recursively convert numpy scalars/arrays/dtypes to JSON-safe python types
def to_jsonable(obj):
    if isinstance(obj, dict):
        # also coerce keys to str to avoid numpy.integer keys exploding json.dump
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

def undersample_then_smote(X, y, USE_UNDERSAMPLE, UNDER_SAMPLER_RATIO, USE_SMOTE):
    # 1) random undersample majority to ~ratio * minority
    if USE_UNDERSAMPLE:
        # compute target counts
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:  # binary expected in per-class training
            return X, y
        maj = classes[np.argmax(counts)]
        minc = classes[np.argmin(counts)]
        n_min = counts.min()
        n_maj_target = int(round(n_min * UNDER_SAMPLER_RATIO))
        sampling_strategy = {minc: n_min, maj: min(n_maj_target, counts.max())}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
        X, y = rus.fit_resample(X, y)
    # 2) optional SMOTE to further balance
    if USE_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X, y = sm.fit_resample(X, y)
    return X, y

# -----------------------
# TCN blocks (residual)
# -----------------------
class StackedTCNDataset(Dataset):
    def __init__(self, X_flat: np.ndarray, y: np.ndarray, time_steps, has_pool):
        # zero-copy where possible
        self.X_flat = np.asarray(X_flat, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.time_steps = time_steps
        self.has_pool = has_pool

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X_flat[idx]
        x_temporal = x[:temporal_flat_dim].reshape(self.time_steps, BASE_FEATS).T   # (F, T)

        if self.has_pool:
            pooled_channels = []
            off = temporal_flat_dim
            for _ in range(n_pool_stats):
                chan = x[off: off + BASE_FEATS]          # (F,)
                pooled_channels.append(np.tile(chan[:, None], (1, self.time_steps)))  # (F, T)
                off += BASE_FEATS
            x_full = np.concatenate([x_temporal] + pooled_channels, axis=0)      # (F*(1+n_pool), T)
        else:
            x_full = x_temporal

        return torch.from_numpy(x_full), torch.tensor(self.y[idx])

        
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[..., :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return self.net(x) + self.down(x)

class SimpleTCN(nn.Module):
    def __init__(self, in_ch, n_classes, widths=WIDTH, ks=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        # (optional) light input norm helps when concatenating pooled channels
        self.in_norm = nn.BatchNorm1d(in_ch)
        chs = [in_ch] + [widths] * len(dilations)
        self.tcn = nn.Sequential(*[
            TemporalBlock(chs[i], chs[i+1], ks, d, dropout) for i, d in enumerate(dilations)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(widths, widths), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths, n_classes),
        )
    def forward(self, x):
        x = self.in_norm(x)
        return self.head(self.tcn(x))


class MultiLabelTCN(nn.Module):
    def __init__(self, in_ch, n_beh, widths=WIDTH, ks=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        chs = [in_ch] + [widths] * len(dilations)
        self.tcn = nn.Sequential(*[
            TemporalBlock(chs[i], chs[i+1], ks, d, dropout) for i, d in enumerate(dilations)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(widths, widths), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths, n_beh)  # logits for each behavior
        )
    def forward(self, x):  # x: (B, C, T)
        z = self.tcn(x)
        logits = self.head(z)  # (B, n_beh)
        return logits

def run_epoch(model, criterion, opt, scaler, scheduler, loader, train=True, epoch_idx=1, phase="train"):
    model.train(mode=train)
    t0 = time.time()
    total_loss, seen = 0.0, 0
    preds, targs = [], []

    num_batches = len(loader)
    print_every = max(1, int(num_batches * PRINT_FRAC))

    for bi, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        bs = xb.size(0)

        if train:
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        else:
            with torch.no_grad(), amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

        total_loss += loss.item() * bs
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        targs.append(yb.detach().cpu().numpy())
        seen += bs

        if (bi % print_every == 0) or (bi == num_batches):
            elapsed = time.time() - t0
            it_per_sec = bi / max(1e-6, elapsed)
            eta = (num_batches - bi) / max(1e-6, it_per_sec)
            print(f"[{phase}] epoch {epoch_idx:02d} {bi:>4d}/{num_batches:<4d} "
                  f"loss={total_loss/seen:.4f}  {it_per_sec:.2f} it/s | ETA {eta/60:.1f} min")

    avg_loss = total_loss / len(loader.dataset)
    P = np.concatenate(preds, axis=0)  # probs
    Y = np.concatenate(targs, axis=0)
    # temporary fixed-threshold macro-F1 for early stopping signal
    Yhat = (P >= 0.5).astype(int)
    macro_f1 = f1_score(Y, Yhat, average="macro", zero_division=0)
    return avg_loss, macro_f1, P, Y

def _state_dict_for_save(m: nn.Module):
    return m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict()

def _load_state_dict(m: nn.Module, sd):
    if isinstance(m, nn.DataParallel): m.module.load_state_dict(sd)
    else: m.load_state_dict(sd)

def collect_probs(model,loader):
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            with amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                lg = model(xb)
            all_p.append(torch.sigmoid(lg).cpu().numpy())
            all_y.append(yb.numpy())
    return np.vstack(all_p), np.vstack(all_y)


def tune_thresholds(P, Y, N_BEH, beta=1.0, clip=(0.02, 0.98)):
    thr = np.zeros(P.shape[1], dtype=np.float32)
    for j in range(P.shape[1]):
        prec, rec, t = precision_recall_curve(Y[:, j], P[:, j])
        if t.size == 0:
            thr[j] = 0.5
            continue
        f = (1+beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-9)
        # align with thresholds (skip first PR point)
        f = f[1:]; t = t
        if f.size == 0:
            thr[j] = 0.5; continue
        best = int(np.nanargmax(f))
        thr_j = float(t[best])
        thr[j] = float(np.clip(thr_j, clip[0], clip[1]))
    return thr
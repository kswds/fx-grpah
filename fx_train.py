import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import os

# =========================================================
FILE_PATH = "factor_final_daily.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
LOOKBACK = 20
RV_WINDOW = 20  #Realized Volatility    
BATCH_SIZE = 128
EPOCHS = 30
LR = 3e-4

GNN_TYPE = "gat"    # "gcn", "sage", "gat"

CCYS = ["USD","EUR","JPY","GBP","CAD","AUD","CHF","NZD","SEK","NOK"]
USD_IDX = CCYS.index("USD")
N_CCY = len(CCYS)

LOCAL_DIM = 4

GLOBAL_FEATURES = [
    "Global_Gold",
    "Global_VIX",
    "Global_Oil",
    "Global_US10Y",
    "Global_Copper",
    "Global_SP500",
    "Global_US2Y",
]
MACRO_DIM = len(GLOBAL_FEATURES)

HIDDEN = 64
HEADS = 4 

LAMBDA_VAR = 0.005 
LAMBDA_A_L1 = 1e-4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
# =========================================================
# 1. Load data
# =========================================================
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
else:
    print(f"Warning: '{FILE_PATH}' not found. Using dummy data.")
    dates = pd.date_range(start="2020-01-01", periods=500)
    df = pd.DataFrame({"Date": dates})
    for c in CCYS:
        df[f"{c}_FX"] = np.random.uniform(100, 150, 500) if c == "JPY" else np.random.uniform(0.5, 1.5, 500)
        df[f"{c}_Yield10Y"] = np.random.uniform(1, 5, 500)
        df[f"{c}_Stock"] = np.cumsum(np.random.randn(500)) + 100
    for g in GLOBAL_FEATURES:
        df[g] = np.cumsum(np.random.randn(500)) + 50

# =========================================================
# 2. FX normalize
# =========================================================
USD_PER_UNIT = {"EUR","GBP","AUD","NZD"} # USD per 1 unit of currency
FOREIGN_PER_USD = {"JPY","CAD","CHF","SEK","NOK"} # currency per 1 USD

def fx_to_log(series, ccy):
    x = series.astype(float)
    if ccy in USD_PER_UNIT:
        return np.log(x)
    elif ccy in FOREIGN_PER_USD:
        return -np.log(x)
    elif ccy == "USD":
        return np.zeros(len(x))
    else:
        raise ValueError(ccy)

p_fx = {c: fx_to_log(df[f"{c}_FX"], c) for c in CCYS}
p_fx = pd.DataFrame(p_fx)
r_fx = p_fx.diff()

# =========================================================
# 3. Features
# =========================================================
feat = {}
for c in CCYS:
    feat[f"{c}_StockRet"] = np.log(df[f"{c}_Stock"]).diff()
    feat[f"{c}_dY10"]     = df[f"{c}_Yield10Y"].diff()
    feat[f"{c}_FXRet"]    = r_fx[c]

for g in GLOBAL_FEATURES:
    if g == "Global_VIX" or "Yield" in g or "US10Y" in g or "US2Y" in g:
        feat[g] = df[g].diff()
    else:
        feat[g] = np.log(df[g]).diff()

data = pd.DataFrame(feat).dropna().reset_index(drop=True)

# =========================================================
# 4. Tensors
# =========================================================
BASE_LOCAL_DIM = 3 
def build_tensors_base(d):
    T = len(d)
    X_local_base = np.zeros((T, N_CCY, BASE_LOCAL_DIM), dtype=np.float32)
    X_macro = d[GLOBAL_FEATURES].values.astype(np.float32)
    Y = np.zeros((T, N_CCY), dtype=np.float32)

    for i, c in enumerate(CCYS):
        X_local_base[:, i, 0] = d[f"{c}_FXRet"].values
        X_local_base[:, i, 1] = d[f"{c}_dY10"].values
        X_local_base[:, i, 2] = d[f"{c}_StockRet"].values
        Y[:, i] = d[f"{c}_FXRet"].values

    X_local_base = (X_local_base - X_local_base.mean(axis=(0,1))) / (X_local_base.std(axis=(0,1)) + 1e-6)
    X_macro = (X_macro - X_macro.mean(axis=0)) / (X_macro.std(axis=0) + 1e-6)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)
    return X_local_base, X_macro, Y

X_local_base, X_macro, Y = build_tensors_base(data)

# =========================================================
# 5. Dataset
# =========================================================
def realized_vol_within_window(fxret_seq, window=20):
    L, N = fxret_seq.shape
    rv = torch.zeros((L, N), dtype=fxret_seq.dtype)
    for t in range(L):
        s = max(0, t - window + 1)
        seg = fxret_seq[s:t+1]
        if seg.size(0) >= 2:
            rv[t] = seg.std(dim=0, unbiased=False)
        else:
            rv[t] = 0.0
    return rv

class FXDataset(Dataset):
    def __init__(self, X_local_base, X_macro, Y, lookback, rv_window=20, macro_mode="real"):
        self.X_local_base = X_local_base
        self.X_macro = X_macro
        self.Y = Y
        self.L = lookback
        self.rv_window = min(rv_window, lookback)
        self.macro_mode = macro_mode

    def __len__(self):
        return len(self.X_local_base) - self.L

    def __getitem__(self, idx):
        xl_base = torch.tensor(self.X_local_base[idx:idx+self.L], dtype=torch.float32)
        xm = torch.tensor(self.X_macro[idx:idx+self.L], dtype=torch.float32)
        if self.macro_mode == "zero":
            xm = torch.zeros_like(xm)
        y = torch.tensor(self.Y[idx+self.L], dtype=torch.float32)
        
        fxret_seq = xl_base[:, :, 0]
        rv_seq = realized_vol_within_window(fxret_seq, window=self.rv_window)
        xl = torch.cat([xl_base, rv_seq.unsqueeze(-1)], dim=2)
        return xl, xm, y

#fully-connected currency graph
def fully_connected_edge_index(N):
    edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    return torch.tensor(edges, dtype=torch.long).T

EDGE_INDEX_SINGLE = fully_connected_edge_index(N_CCY).to(DEVICE)

# =========================================================
# 6. Model: GRU(Local) + GNN + Macro Factors
# =========================================================
class FXStrengthGNN_HeteroMacro(nn.Module):
    def __init__(self, local_dim=4, macro_dim=7, hidden=64, gnn_type="gat"):
        super().__init__()
        self.hidden = hidden
        self.macro_dim = macro_dim
        self.local_gru = nn.GRU(local_dim, hidden, batch_first=True) #step1: 시계열 인코딩
        #step2: multi-currency spillover 
        if gnn_type == "gcn":
            self.ccy_gnn = GCNConv(hidden, hidden)
        elif gnn_type == "sage":
            self.ccy_gnn = SAGEConv(hidden, hidden)
        elif gnn_type == "gat":
            self.ccy_gnn = GATConv(hidden, hidden, heads=HEADS, concat=False)
        else:
            raise ValueError(gnn_type)
        #step3: heterogenous macro-to-currency effects
        self.macro_embed = nn.Linear(macro_dim, macro_dim * hidden, bias=False)
        self.A = nn.Parameter(torch.zeros(N_CCY, macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape
        #1. Local encdoing per currency 
        x = xl.permute(0, 2, 1, 3).reshape(B*N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0) 
        #2. curreyncy-currency spillover
        E = edge_index_single.size(1)
        edge_b = edge_index_single.repeat(1, B)
        offset = torch.arange(B, device=xl.device).repeat_interleave(E) * N
        edge_b = edge_b + offset.unsqueeze(0)

        z = self.ccy_gnn(h, edge_b)
        z_ccy = z.view(B, N, self.hidden)
        #3. Macro embedding
        m_t = xm[:, -1, :]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)
        #4. Heterogenous transmission
        A = self.A.unsqueeze(0).unsqueeze(-1)
        u_exp = u.unsqueeze(1)
        m_msg = (A * u_exp).sum(dim=2)
        #5. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)
        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, USD_IDX:USD_IDX+1]
        return rhat, ds, z_ccy, m_msg

# =========================================================
# 7. Loss and Metrics
# =========================================================
def loss_fn(rhat, y, ds, A_param):
    mask = torch.ones(N_CCY, dtype=torch.bool, device=y.device)
    mask[USD_IDX] = False
    mse = ((rhat[:, mask] - y[:, mask])**2).mean() 
    var_term = -ds.var(dim=1).mean() #latent collpase 방지
    l1_A = A_param.abs().mean() #sparsity
    return mse + LAMBDA_VAR * var_term + LAMBDA_A_L1 * l1_A

def triangle_error(ds_np):
    T, N = ds_np.shape
    err = 0.0
    cnt = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i != j and j != k and i != k:
                    e = (ds_np[:, i]-ds_np[:, j]) + (ds_np[:, j]-ds_np[:, k]) + (ds_np[:, k]-ds_np[:, i])
                    err += np.abs(e).mean()
                    cnt += 1
    return err / max(cnt, 1)

def macro_usage_ratio(z_ccy, m_msg, eps=1e-12):
    num = torch.norm(m_msg, dim=(1,2)).mean()
    den = (torch.norm(m_msg, dim=(1,2)) + torch.norm(z_ccy, dim=(1,2)) + eps).mean()
    return (num / den).item()

def heterogeneity_score(A):
    var_f = A.var(dim=0, unbiased=False)
    return var_f.mean().item(), var_f.detach().cpu().numpy()

# =========================================================
# 8. Train/Eval
# =========================================================
def train_and_evaluate(macro_mode, label):
    n_total = len(X_local_base)
    split_idx = int(n_total * 0.8)

    train_local = X_local_base[:split_idx]
    train_macro = X_macro[:split_idx]
    
    local_mean = train_local.mean(axis=(0, 1), keepdims=True)
    local_std  = train_local.std(axis=(0, 1), keepdims=True) + 1e-6
    
    macro_mean = train_macro.mean(axis=0, keepdims=True)
    macro_std  = train_macro.std(axis=0, keepdims=True) + 1e-6
    
    X_local_scaled = (X_local_base - local_mean) / local_std
    X_macro_scaled = (X_macro - macro_mean) / macro_std
    
    dataset = FXDataset(
        X_local_base=X_local_scaled,
        X_macro=X_macro_scaled,
        Y=Y, 
        lookback=LOOKBACK,
        rv_window=RV_WINDOW,
        macro_mode=macro_mode
    )

    n = len(dataset)
    split = int(n * 0.8)
    train_ds = torch.utils.data.Subset(dataset, list(range(0, split)))
    test_ds  = torch.utils.data.Subset(dataset, list(range(split, n)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FXStrengthGNN_HeteroMacro(
        local_dim=LOCAL_DIM,
        macro_dim=MACRO_DIM,
        hidden=HIDDEN,
        gnn_type=GNN_TYPE
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---- train ----
    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(DEVICE), xm.to(DEVICE), yb.to(DEVICE)
            rhat, ds, z_ccy, m_msg = model(xl, xm, EDGE_INDEX_SINGLE)
            loss = loss_fn(rhat, yb, ds, model.A)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
        if epoch in (1, EPOCHS) or epoch % 5 == 0:
            print(f"[{label}] epoch {epoch:02d}/{EPOCHS} | mean loss = {float(np.mean(losses)):.4f}")

    # ---- eval ----
    model.eval()
    rhat_all, y_all, ds_all = [], [], []
    z_all, m_all = [], []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm, yb = xl.to(DEVICE), xm.to(DEVICE), yb.to(DEVICE)
            rhat, ds, z_ccy, m_msg = model(xl, xm, EDGE_INDEX_SINGLE)
            rhat_all.append(rhat.cpu())
            y_all.append(yb.cpu())
            ds_all.append(ds.cpu())
            z_all.append(z_ccy.cpu())
            m_all.append(m_msg.cpu())

    rhat = torch.cat(rhat_all, dim=0)
    y    = torch.cat(y_all, dim=0)
    ds   = torch.cat(ds_all, dim=0)
    z_ccy = torch.cat(z_all, dim=0)   # [T,N,H]
    m_msg = torch.cat(m_all, dim=0)

    mask = torch.ones(N_CCY, dtype=torch.bool)
    mask[USD_IDX] = False

    rmse = torch.sqrt(((rhat[:,mask] - y[:,mask])**2).mean()).item()
    mae  = torch.abs(rhat[:,mask] - y[:,mask]).mean().item()
    hit  = ((torch.sign(rhat[:,mask]) == torch.sign(y[:,mask])).float()).mean().item()

    strength_norm = torch.norm(ds, dim=1).mean().item()
    tri_err = triangle_error(ds.numpy())

    mur = macro_usage_ratio(z_ccy, m_msg) #매크로 정보 사용 비율
    hs_mean, hs_vec = heterogeneity_score(model.A.detach().cpu()) 

    print(f"\n===== {label} =====")
    print(f"RMSE               : {rmse:.4f}")
    print(f"MAE                : {mae:.4f}")
    print(f"Directional Acc.   : {hit:.4f}")
    print(f"Mean ||Δs||        : {strength_norm:.4f}")
    print(f"Triangle Error     : {tri_err:.6e}")
    print(f"Macro Usage Ratio  : {mur:.4f}   (macro message share)")
    print(f"Heterogeneity Score: {hs_mean:.6f} (mean Var_i(a_i,f))")
    print(f" per-factor Var    : {np.array2string(hs_vec, precision=6)}")
    
    return {"rmse": rmse, "mae": mae, "hit": hit, "mur": mur, "hs": hs_mean, "model": model}
# =========================================================
# 9. Main Execution
# =========================================================
if __name__ == "__main__":
    if RV_WINDOW > LOOKBACK:
        raise ValueError("RV_WINDOW must be <= LOOKBACK.")

    print(">>> Start Training...")
    
    res_macro = train_and_evaluate("real", label="WITH MACRO")
    res_zero  = train_and_evaluate("zero", label="WITHOUT MACRO")

    print("\n===== Macro Contribution (Hetero Edges) =====")
    print(f"Δ RMSE : {res_zero['rmse'] - res_macro['rmse']:.4f}")
    print(f"Δ MAE  : {res_zero['mae']  - res_macro['mae']:.4f}")
    print(f"Δ Hit  : {res_macro['hit'] - res_zero['hit']:.4f}")
    print(f"Δ MUR  : {res_macro['mur'] - res_zero['mur']:.4f}")
    print(f"HS(mean Var across factors): {res_macro['hs']:.6f}")

import os
import itertools
from xml.parsers.expat import model

# ==========================================
# 0. Environment Setup
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import random
import gc 
from collections import Counter
import matplotlib.pyplot as plt
import sys
import datetime


# ==========================================
# 1. Random Seed
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True) 
    print(f"Random seed fixed to: {seed}")

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
# ==========================================
# 2. Dataset Processing
# ==========================================
class SupplyChainDataset:
    def __init__(self, trade_file, industry_file, inventory_file, scale_factor=1e8): 
        self.scale_factor = scale_factor 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*40)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*40)
        
        try:
            print("Loading data files...")
            self.trade_df = pd.read_csv(trade_file, low_memory=False).dropna(how='all')
            self.industry_df = pd.read_csv(industry_file, low_memory=False).dropna(how='all')
            self.inventory_df = pd.read_csv(inventory_file, low_memory=False).dropna(how='all')
        except FileNotFoundError as e:
            print(f"Error: Missing file {e.filename}")
            raise
        
        self.prepare_data()
        self.generate_detailed_report()

    def prepare_data(self):
        print("Preprocessing data...")
        self.trade_df['rpt'] = pd.to_datetime(self.trade_df['rpt'])
        self.inventory_df['data'] = pd.to_datetime(self.inventory_df['data'])
        self.industry_df['ticker'] = self.industry_df['ticker'].fillna(0).astype(int).astype(str)
        self.inventory_df['ticker'] = self.inventory_df['ticker'].astype(str)

        self.trade_df['supplier'] = self.trade_df['supplier'].astype(str).str.strip()
        self.trade_df['customer'] = self.trade_df['customer'].astype(str).str.strip()

        valid_suppliers = set(x for x in self.trade_df['supplier'] if x.lower() != 'nan' and x != '')
        valid_customers = set(x for x in self.trade_df['customer'] if x.lower() != 'nan' and x != '')
        all_companies_set = valid_suppliers.union(valid_customers)
        all_companies = sorted(list(all_companies_set))

        self.company_to_idx = {cid: i for i, cid in enumerate(all_companies)}
        self.idx_to_company = {i: cid for cid, i in self.company_to_idx.items()}
        self.num_nodes = len(all_companies)
        
        self.comp_info = self.industry_df.set_index('company_id')
        self.ticker_map = dict(zip(self.industry_df['company_id'], self.industry_df['ticker']))
        
        code_col_name = None
        for col in ['final_industry_code', 'predicted_industry_code', 'industry_code', 'IndustryCode']:
            if col in self.industry_df.columns: code_col_name = col; break
        if code_col_name is None: raise KeyError("Industry code column not found")

        unique_codes = self.industry_df[code_col_name].dropna().astype(str).unique()
        unique_codes = sorted([c for c in unique_codes if c.strip() != ''])
        if 'Natural_Person' not in unique_codes: unique_codes.append('Natural_Person')
        self.industry_to_idx = {code: i+1 for i, code in enumerate(unique_codes)}
        self.num_industries = len(unique_codes) + 1
        
        self.node_industry_indices = torch.zeros(self.num_nodes, dtype=torch.long)
        self.node_custom_codes = [None] * self.num_nodes 
        self.node_industry_names = [None] * self.num_nodes
        self.is_listed = np.zeros(self.num_nodes, dtype=bool) 
        valid_tickers = set(self.inventory_df['ticker'])

        for cid, idx in self.company_to_idx.items():
            ind_code_val = None; ind_name_val = None
            if str(cid).lower().startswith('p'): 
                ind_code_val = 'Natural_Person'; ind_name_val = 'Natural_Person' 
            elif cid in self.comp_info.index:
                row = self.comp_info.loc[cid]
                if code_col_name in row:
                    val = row[code_col_name]
                    if isinstance(val, pd.Series): val = val.iloc[0]
                    if pd.notna(val): ind_code_val = str(val).strip()
                if 'industry_name' in row:
                    n_val = row['industry_name']
                    if isinstance(n_val, pd.Series): n_val = n_val.iloc[0]
                    if pd.notna(n_val): ind_name_val = str(n_val).strip()
            
            if ind_code_val:
                self.node_custom_codes[idx] = ind_code_val
                if ind_code_val in self.industry_to_idx:
                    self.node_industry_indices[idx] = self.industry_to_idx[ind_code_val]
            if ind_name_val: self.node_industry_names[idx] = ind_name_val
                
            if cid in self.ticker_map and self.ticker_map[cid] in valid_tickers:
                self.is_listed[idx] = True
        
        self.node_ids = torch.arange(self.num_nodes, dtype=torch.long)
        self.identify_node_roles()

        print("Aggregating trade data by half-year...")
        self.dates = []
        for year in range(2015, 2024):
            d1 = pd.Timestamp(f"{year}-06-30")
            if d1 <= pd.Timestamp("2023-06-30"): self.dates.append(d1)
            d2 = pd.Timestamp(f"{year}-12-31")
            if d2 <= pd.Timestamp("2023-06-30"): self.dates.append(d2)
        
        self.temporal_data = []
        prev_date = pd.Timestamp("2014-12-31")
        
        for curr_date in self.dates:
            period_trades = self.trade_df[
                (self.trade_df['rpt'] > prev_date) & 
                (self.trade_df['rpt'] <= curr_date)
            ]
            agg = period_trades.groupby(['supplier', 'customer'])['trade_amount'].sum().reset_index()
            
            src = agg['supplier'].map(self.company_to_idx).values
            dst = agg['customer'].map(self.company_to_idx).values
            amt = agg['trade_amount'].fillna(0).values / self.scale_factor
            
            valid = ~np.isnan(src) & ~np.isnan(dst)
            if valid.sum() > 0:
                edge_index = torch.tensor(np.stack([src[valid], dst[valid]]), dtype=torch.long)
                edge_attr = torch.tensor(amt[valid], dtype=torch.float32).unsqueeze(1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)

            # Seasonal feature
            is_year_end = 1 if curr_date.month == 12 else 0
            season_tensor = torch.full((self.num_nodes,), is_year_end, dtype=torch.long)
            # Purchase / sales features
            P = np.zeros(self.num_nodes); S = np.zeros(self.num_nodes)
            p_agg = period_trades.groupby('customer')['trade_amount'].sum() / self.scale_factor
            s_agg = period_trades.groupby('supplier')['trade_amount'].sum() / self.scale_factor
            for cid, val in p_agg.items():
                if cid in self.company_to_idx: P[self.company_to_idx[cid]] = val
            for cid, val in s_agg.items():
                if cid in self.company_to_idx: S[self.company_to_idx[cid]] = val
            
            # Labels
            label = np.zeros(self.num_nodes)
            mask = np.zeros(self.num_nodes, dtype=bool)
            current_is_listed = np.zeros(self.num_nodes, dtype=int) 
            
            current_inv = self.inventory_df[self.inventory_df['data'] == curr_date]
            inv_map = dict(zip(current_inv['ticker'], current_inv['Inventory']))

            for i in range(self.num_nodes):
                if self.is_listed[i]:
                    cid = self.idx_to_company[i]
                    tk = self.ticker_map.get(cid)
                    val = inv_map.get(tk, np.nan) 
                    
                    if not np.isnan(val): 
                        label[i] = val / self.scale_factor
                        mask[i] = True 
                        current_is_listed[i] = 1 
            
            self.temporal_data.append({
                'edge_index': edge_index, 'edge_attr': edge_attr,
                'P': torch.tensor(P, dtype=torch.float32), 'S': torch.tensor(S, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32), 'mask': torch.tensor(mask, dtype=torch.bool),
                'is_listed': torch.tensor(current_is_listed, dtype=torch.long), 
                'season': season_tensor,   
                'date': curr_date
            })
            prev_date = curr_date
            
        print(f"Aggregation complete. Total periods: {len(self.temporal_data)}")
        self.init_F0_at_specific_date()

    def identify_node_roles(self):
        s_sum = self.trade_df.groupby('supplier')['trade_amount'].sum()
        p_sum = self.trade_df.groupby('customer')['trade_amount'].sum()
        self.is_source = np.zeros(self.num_nodes, dtype=bool)
        self.is_sink = np.zeros(self.num_nodes, dtype=bool)
        self.node_roles = ['Intermediate'] * self.num_nodes 
        for cid, idx in self.company_to_idx.items():
            s = s_sum.get(cid, 0); p = p_sum.get(cid, 0)
            if s > 0 and p == 0: 
                self.is_source[idx] = True; self.node_roles[idx] = 'Source' 
            elif p > 0 and s == 0: 
                self.is_sink[idx] = True; self.node_roles[idx] = 'Sink'   
        
        mask_np = (self.is_source | self.is_sink)
        excluded_count = 0
        for i in range(self.num_nodes):
            code = self.node_custom_codes[i]
            if code is not None:
                if code.startswith('CSF_40') or code.startswith('CSF_60'):
                    mask_np[i] = True
                    excluded_count += 1
        
        self.ignore_in_loss_mask = torch.tensor(mask_np)

    def init_F0_at_specific_date(self, sme_ratio=0.1): 
        target_date = pd.Timestamp("2015-06-30")
        target_idx = 0
        for idx, d in enumerate(self.dates):
            if d == target_date: target_idx = idx; break
        
        self.F0 = torch.zeros(self.num_nodes, dtype=torch.float32)
        self.init_data_idx = target_idx
        
        target_inv_df = self.inventory_df[self.inventory_df['data'] == target_date]
        ticker_to_val = dict(zip(target_inv_df['ticker'], target_inv_df['Inventory']))
        
        industry_values = {} 
        has_true_init = np.zeros(self.num_nodes, dtype=bool) 
        
        for i in range(self.num_nodes):
            if self.is_listed[i]:
                cid = self.idx_to_company[i]
                tk = self.ticker_map.get(cid)
                val = ticker_to_val.get(tk, np.nan) 
                
                if not np.isnan(val):
                    scaled_val = val / self.scale_factor 
                    self.F0[i] = scaled_val
                    has_true_init[i] = True  
                    
                    ind_idx = self.node_industry_indices[i].item()
                    if ind_idx not in industry_values: industry_values[ind_idx] = []
                    if scaled_val > 0: industry_values[ind_idx].append(scaled_val)
        
        industry_means = {k: np.mean(v) for k, v in industry_values.items() if len(v) > 0}
        
        for i in range(self.num_nodes):
            if not has_true_init[i]:
                ind_idx = self.node_industry_indices[i].item()
                if ind_idx in industry_means:
                    ratio = 0.5 if self.is_listed[i] else sme_ratio
                    self.F0[i] = industry_means[ind_idx] * ratio

    def generate_detailed_report(self):
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Nodes: {self.num_nodes}")
        print(f"Time span: {self.dates[0].date()} to {self.dates[-1].date()}")
        print("=" * 60 + "\n")

class AsymmetricGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AsymmetricGNNLayer, self).__init__()
        self.W_s = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        
        self.W_s_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r_self = nn.Linear(in_dim, out_dim, bias=False)

        self.norm_s = nn.LayerNorm(out_dim)
        self.norm_r = nn.LayerNorm(out_dim)

    def forward(self, S, R, edge_index, edge_weight):
        row, col = edge_index
        num_nodes = S.size(0)
        
        if edge_index.shape[1] == 0: 
            return F.relu(self.norm_s(self.W_s_self(S))), F.relu(self.norm_r(self.W_r_self(R)))

        max_in = torch.zeros(num_nodes, 1, device=S.device)
        max_in.scatter_reduce_(0, col.unsqueeze(1), edge_weight, reduce='amax', include_self=False)
        max_out = torch.zeros(num_nodes, 1, device=R.device)
        max_out.scatter_reduce_(0, row.unsqueeze(1), edge_weight, reduce='amax', include_self=False)
        
        w_in = edge_weight / (max_in[col] + 1e-8)
        w_out = edge_weight / (max_out[row] + 1e-8)

        S_trans = self.W_s(S)
        msg_s = S_trans[col] * w_out
        S_new = torch.zeros_like(S_trans)
        S_new.index_add_(0, row, msg_s)
        
        S_agg = self.W_s_self(S) + S_new
        S_out = F.relu(self.norm_s(S_agg))

        R_trans = self.W_r(R)
        msg_r = R_trans[row] * w_in
        R_new = torch.zeros_like(R_trans)
        R_new.index_add_(0, col, msg_r)
        
        R_agg = self.W_r_self(R) + R_new
        R_out = F.relu(self.norm_r(R_agg))

        return S_out, R_out

class SupplyChainGNN(nn.Module):
    def __init__(self, num_nodes, num_industries, id_dim, ind_dim, listed_dim, season_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(SupplyChainGNN, self).__init__()
        self.id_embedding = nn.Embedding(num_nodes, id_dim)
        self.ind_embedding = nn.Embedding(num_industries, ind_dim)
        self.listed_embedding = nn.Embedding(2, listed_dim)
        self.season_embedding = nn.Embedding(2, season_dim) 

        fusion_input_dim = id_dim + ind_dim + listed_dim + season_dim + 1
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            AsymmetricGNNLayer(in_dim=hidden_dim, out_dim=hidden_dim) 
            for _ in range(num_layers)
        ])
        self.gnn_dropout = nn.Dropout(dropout_rate)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)

        )
        
    def forward(self, node_ids, ind_indices, is_listed, season, delta_F, edge_index, edge_weight):
        x_id = self.id_embedding(node_ids)
        x_ind = self.ind_embedding(ind_indices)
        x_listed = self.listed_embedding(is_listed)
        x_season = self.season_embedding(season)
        inv_feature = (torch.sign(delta_F) * torch.log1p(torch.abs(delta_F))).unsqueeze(1)
        x_cat = torch.cat([x_id, x_ind, x_listed, x_season, inv_feature], dim=1)
        x = self.feature_fusion(x_cat)
        
        S, R = x, x
        
        for i, layer in enumerate(self.layers):
            S, R = layer(S, R, edge_index, edge_weight)
            if i < self.num_layers - 1:
                S = self.gnn_dropout(S)
                R = self.gnn_dropout(R)
                
        Z = S + R
        
        raw_out = self.mlp(Z)
        alpha = 10 * torch.tanh(raw_out[:, 0])
        
        beta = raw_out[:, 1] 
        
        return alpha, beta  
# ==========================================
# 4. training and evaluation
# ==========================================
def train_and_predict(dataset, params, epochs=2000, output_dir='output_results', search_mode=False):
    set_seed(42) 
    
    if not search_mode:
        os.makedirs(output_dir, exist_ok=True)
    
    device = dataset.device
    node_ids = dataset.node_ids.to(device)
    node_ind_indices = dataset.node_industry_indices.to(device)
    ignore_mask = dataset.ignore_in_loss_mask.to(device)
    
    model = SupplyChainGNN(
        num_nodes=dataset.num_nodes,
        num_industries=dataset.num_industries,
        id_dim=32, ind_dim=16, listed_dim=4, season_dim=4,
        hidden_dim=params['hidden_dim'], 
        num_layers=params.get('num_layers', 2),
        dropout_rate=params['dropout_rate']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    train_end_date = pd.Timestamp("2020-12-31")
    train_indices = []
    test_indices = []
    
    for i, date in enumerate(dataset.dates):
        if i <= dataset.init_data_idx: continue
        if date <= train_end_date:
            train_indices.append(i)
        else:
            test_indices.append(i)
            
    if not train_indices: 
        train_indices = [i for i in range(len(dataset.dates)) if i > dataset.init_data_idx]
    
    last_train_idx = train_indices[-1] if train_indices else dataset.init_data_idx
    max_idx = len(dataset.dates) - 1

    if not search_mode:
        print(f"Training set contains {len(train_indices)} periods, "
          f"test set contains {len(test_indices)} periods.")
        print(f"Start training for {epochs} epochs... "
          f"Results will be saved to '{output_dir}'")

    if not search_mode:
    temporal_data_gpu = {}
    for t in range(dataset.init_data_idx, max_idx + 1):
        data_cpu = dataset.temporal_data[t]
        temporal_data_gpu[t] = {
            'edge_index': data_cpu['edge_index'].to(device, non_blocking=True),
            'edge_attr': data_cpu['edge_attr'].to(device, non_blocking=True),
            'P': data_cpu['P'].to(device, non_blocking=True),
            'S': data_cpu['S'].to(device, non_blocking=True),
            'label': data_cpu['label'].to(device, non_blocking=True),
            'mask': data_cpu['mask'].to(device, non_blocking=True),
            'is_listed': data_cpu['is_listed'].to(device, non_blocking=True),
            'season': data_cpu['season'].to(device, non_blocking=True), # 【新增】
            'date': data_cpu['date']
        }
        
    loss_history = []  

    # ==========================
    # Phase 1: Training
    # ==========================
    eps = 1e-6 
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        curr_F = dataset.F0.to(device).detach()
        prev_F = curr_F.clone()
        
        epoch_loss_val = 0.0 
        
        for t in range(dataset.init_data_idx + 1, last_train_idx + 1):
            data_gpu = temporal_data_gpu[t]
            edge_index = data_gpu['edge_index']
            edge_attr = data_gpu['edge_attr']
            P = data_gpu['P']
            S = data_gpu['S']
            label = data_gpu['label']
            mask = data_gpu['mask']
            curr_is_listed = data_gpu['is_listed']
            season = data_gpu['season']  
            delta_F = curr_F - prev_F 

            alpha, beta = model(node_ids, node_ind_indices, curr_is_listed, season, delta_F, edge_index, edge_attr)

            F_pred_raw = curr_F + (P + eps) * (1 + alpha) - S + beta
            
            loss_step = 0.0
            valid_mask = mask & (~ignore_mask)
            
            if valid_mask.sum() > 0:
                loss_step = loss_step + F.huber_loss(F_pred_raw[valid_mask], label[valid_mask], delta=0.5)
                
            loss_neg = torch.relu(-F_pred_raw).mean()
            if loss_neg > 0: 
                neg_weight = params.get('neg_weight', 0.1) 
                loss_step = loss_step + neg_weight * loss_neg
                
            if isinstance(loss_step, torch.Tensor):
                loss_step.backward()
                epoch_loss_val += loss_step.item()

            prev_F = curr_F.clone()
            next_F = F_pred_raw.detach()
            
            next_F = torch.where(mask, label, next_F)
            curr_F = torch.relu(next_F).clamp(max=50000.0)
    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(epoch_loss_val)
        
        if not search_mode:
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {epoch_loss_val:.6f}")
        else:
            if (epoch+1) == epochs:
                print(f"    --> 该组参数最终 Train Loss: {epoch_loss_val:.6f}")

    # ==========================
    # Phase 2: Evaluation
    # ==========================
    if not search_mode:
        print("\n" + "="*50)
        print("开始逐期评估 (Evaluation)")
        print("="*50)
    
    model.eval()
    curr_F = dataset.F0.to(device).detach()
    prev_F = curr_F.clone()
    period_metrics = [] 
    
    with torch.no_grad():
        for t in range(dataset.init_data_idx + 1, max_idx + 1):
            data_gpu = temporal_data_gpu[t]
            curr_date = data_gpu['date']
            curr_date_str = curr_date.strftime('%Y-%m-%d')
            
            edge_index = data_gpu['edge_index']
            edge_attr = data_gpu['edge_attr']
            P = data_gpu['P']
            S = data_gpu['S']
            label = data_gpu['label']
            mask = data_gpu['mask']
            curr_is_listed = data_gpu['is_listed']
            season = data_gpu['season']
            
            prev_data_gpu = temporal_data_gpu[t-1]
            prev_label = prev_data_gpu['label']
            prev_mask = prev_data_gpu['mask']

            delta_F = curr_F - prev_F 
            alpha, beta = model(node_ids, node_ind_indices, curr_is_listed, season, delta_F, edge_index, edge_attr)
            F_pred_raw = curr_F + (P + eps) * (1 + alpha) - S + beta
            F_final = torch.relu(F_pred_raw)
            
            if t in test_indices:
                eval_mask = mask & (~ignore_mask)
                mda_mask = mask & prev_mask & (~ignore_mask)
                
                y_pred_val = F_final[eval_mask]
                y_true_val = label[eval_mask]
                
                if len(y_true_val) > 0:
                    mse = torch.mean((y_true_val - y_pred_val) ** 2).item()
                    rmse = mse ** 0.5
                    mae = torch.mean(torch.abs(y_true_val - y_pred_val)).item()
                    
                    non_zero_mask = y_true_val != 0
                    if non_zero_mask.sum() > 0:
                        mape = torch.mean(torch.abs((y_true_val[non_zero_mask] - y_pred_val[non_zero_mask]) / y_true_val[non_zero_mask])).item()
                    else:
                        mape = float('nan')

                    wape = (torch.sum(torch.abs(y_true_val - y_pred_val)) / (torch.sum(y_true_val) + 1e-8)).item()
                        
                    period_metrics.append({
                        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'WAPE': wape
                    })

                    if not search_mode:
                        print(f"\n>>> data: {curr_date_str} (count: {len(y_pred_val)})")
                        print(f"    RMSE : {rmse:,.4f}")
                        print(f"    MAE  : {mae:,.4f}")
                        print(f"    MAPE : {mape:.4f}")
                        print(f"    WAPE : {wape:.4f}")
                
                if not search_mode:
                    out_data = []
                    
                    y_pred_np = F_final.cpu().numpy()
                    y_true_np = label.cpu().numpy()
                    mask_vals_np = mask.cpu().numpy() 
                    curr_is_listed_np = curr_is_listed.cpu().numpy() 
                    alpha_vals_np = alpha.detach().cpu().numpy()
                    
                    for i in range(dataset.num_nodes):
                        cid = dataset.idx_to_company[i]
                        industry = dataset.node_industry_names[i] if dataset.node_industry_names[i] else "Unknown"
                        role = dataset.node_roles[i]
                        custom_code = dataset.node_custom_codes[i] if dataset.node_custom_codes[i] else ""
                        
                        name = cid
                        if cid in dataset.comp_info.index:
                            row = dataset.comp_info.loc[cid]
                            if 'company_name' in row:
                                val = row['company_name']
                                if isinstance(val, pd.Series): val = val.iloc[0]
                                name = val
                        
                        out_data.append({
                            'Date': curr_date_str,
                            'Company ID': cid,
                            'Company Name': name,
                            'Industry': industry,
                            'Industry Code': custom_code,
                            'Node Role': role,
                            'Predicted Inventory': y_pred_np[i],
                            'True Inventory': y_true_np[i] if mask_vals_np[i] else None,
                            'Is Listed (Dynamic)': bool(curr_is_listed_np[i]), 
                            'Production Coeff (Alpha)': alpha_vals_np[i]
                        })
                    
                    fname = f'prediction_neg_f_{curr_date_str}.csv'
                    save_path = os.path.join(output_dir, fname)
                    pd.DataFrame(out_data).to_csv(save_path, index=False)
                    print(f" saving {save_path}")
                    
            prev_F = curr_F.clone()
            next_F = F_pred_raw
            next_F = torch.where(mask, label, next_F)
            curr_F = torch.relu(next_F).clamp(max=50000.0)

    del temporal_data_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if search_mode:
        final_loss = loss_history[-1] if loss_history else float('nan')
        return period_metrics, final_loss
    else:
        return model, dataset


# ==========================================
# 5. Grid Search Function
# ==========================================
def run_grid_search(dataset, param_grid, search_epochs=2000, save_path='grid_search_results.csv'):
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print("\n" + "*" * 60)
    print(f"Starting grid search... Generated {len(combinations)} parameter combinations.")
    print(f"Note: Grid search epochs are set to {search_epochs} for faster execution.")
    print("*" * 60 + "\n")
    
    if os.path.exists(save_path):
        os.remove(save_path)
    
    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] 测试参数: {params}")
        set_seed(42)

        metrics_list, final_loss = train_and_predict(dataset, params, epochs=search_epochs, search_mode=True)
        row = params.copy()
        row['Final_Train_Loss'] = final_loss 
        
        rmse_list, mae_list, mape_list, wape_list = [], [], [], []
        
        for idx, m in enumerate(metrics_list):
            t = idx + 1
            row[f'T{t}_RMSE'] = m['RMSE']
            row[f'T{t}_MAE'] = m['MAE']
            row[f'T{t}_MAPE'] = m['MAPE']
            row[f'T{t}_WAPE'] = m['WAPE']
            
            rmse_list.append(m['RMSE'])
            mae_list.append(m['MAE'])
            mape_list.append(m['MAPE'])
            wape_list.append(m['WAPE'])
        
        row['Avg_RMSE'] = np.mean(rmse_list) if rmse_list else float('nan')
        row['Avg_MAE'] = np.mean(mae_list) if mae_list else float('nan')
        row['Avg_MAPE'] = np.mean(mape_list) if mape_list else float('nan')
        row['Avg_WAPE'] = np.mean(wape_list) if wape_list else float('nan')
        
        df_row = pd.DataFrame([row])
        
        if i == 0:
            df_row.to_csv(save_path, mode='w', header=True, index=False)
        else:
            df_row.to_csv(save_path, mode='a', header=False, index=False)
        
        del metrics_list, df_row, row
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    print("\n" + "="*60)
    print(f"Grid search completed! Results saved to: {save_path}")
    print("="*60 + "\n")
    
    return pd.read_csv(save_path)

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":

    # Generate unique log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"grid_search_log_{timestamp}.txt"

    sys.stdout = Logger(log_filename)

    print(f"System logs will also be saved to: {log_filename}")

    set_seed(42)

    files = {
        'trade': './dataset/supply_chain_trade.csv',
        'ind': './dataset/industry_class.csv',
        'inv': './dataset/company_inventory.csv'
    }

    # Check dataset files
    if all(os.path.exists(f) for f in files.values()):

        cache_file = './dataset/processed_supply_chain_130k.pt'

        # Load cached dataset
        if os.path.exists(cache_file):

            print(f"Cached dataset found: {cache_file}, loading...")

            ds = torch.load(cache_file, map_location='cpu', weights_only=False)

            ds.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build dataset from CSV
        else:

            print("No cache found, processing CSV files...")

            ds = SupplyChainDataset(
                files['trade'],
                files['ind'],
                files['inv'],
                scale_factor=1e8
            )

            torch.save(ds, cache_file)

            print("Dataset processed and cached locally.")

        # Grid search parameters
        param_grid = {
            'num_layers': [2, 3, 4, 5, 6],
            'hidden_dim': [64, 128, 256, 512],
            'weight_decay': [1e-3, 1e-4, 1e-5, 1e-6],
            'neg_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
            'dropout_rate': [0.2]
        }
        
        results_df = run_grid_search(ds, param_grid, search_epochs=2000)
        
        print("\nExperiment result preview (partial columns):")
        print(results_df[['num_layers', 'hidden_dim', 'lr', 'Avg_RMSE', 'Avg_MAE', 'Avg_MAPE', 'Avg_WAPE']].head(10))
        print("\nAll processes completed. "
      "Please check 'grid_search_results.csv' for detailed comparison.")
    else:
        print("Missing files. Please check whether the required files exist in the dataset folder.")

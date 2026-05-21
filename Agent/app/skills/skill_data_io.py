import pandas as pd
import numpy as np

def load_io_matrices(filepaths: list, common_indices: list) -> list:
    """Load IO matrices for multiple time cross-sections and align indices"""
    matrices = []
    for path in filepaths:
        df = pd.read_csv(path, index_col=0)
        # Force alignment to common industries, fill missing with 0
        df = df.reindex(index=common_indices, columns=common_indices, fill_value=0.0)
        print("1. Matrix shape (rows, columns):", df.shape)
        print("2. Top-left 4x4 slice of the matrix looks like this:\n", df.iloc[:4, :4])
        matrices.append(df)
    return matrices

def extract_spatial_features(industry_name: str, df_inv: pd.DataFrame, ios: list):
    times = ['21H2', '22H1', '22H2', '23H1']
    y = [float(df_inv.loc[industry_name, f'InvDiff_{t}']) for t in times]
    wy = []
    for idx_t, io in enumerate(ios):
        w_ij = io.loc[industry_name, :].values / 1e8
        y_j = df_inv[f'InvDiff_{times[idx_t]}'].values
        # Avoid calculation errors caused by nan
        wy_val = np.nansum(w_ij * y_j)
        wy.append(float(wy_val))
    return round_list(y), round_list(wy)

def extract_gmm_features(industry_name: str, df_inv: pd.DataFrame):
    times = ['21H2', '22H1', '22H2', '23H1']
    y = [float(df_inv.loc[industry_name, f'InvDiff_{t}']) for t in times]
    return round_list(y[1:]), round_list(y[:-1])

def extract_cross_lag_features(industry_name: str, ios: list):
    d_data = [float(io.loc[industry_name, :].sum() / 1e8) for io in ios]
    s_data = [float(io.loc[:, industry_name].sum() / 1e8) for io in ios]
    return round_list(d_data), round_list(s_data)

def extract_did_features(industry_name: str, ios: list):
    d_data = [float(io.loc[industry_name, :].sum() / 1e8) for io in ios]
    s_data = [float(io.loc[:, industry_name].sum() / 1e8) for io in ios]
    return round(float(np.mean(d_data)), 2), round(float(np.mean(s_data)), 2)

def extract_forecast_features(industry_name: str, df_inv: pd.DataFrame, ios: list):
    times = ['21H2', '22H1', '22H2', '23H1']
    y_data = [float(df_inv.loc[industry_name, f'InvDiff_{t}']) for t in times]
    d_data = [float(io.loc[industry_name, :].sum() / 1e8) for io in ios]
    s_data = [float(io.loc[:, industry_name].sum() / 1e8) for io in ios]
    return round_list(y_data), round_list(s_data), round_list(d_data)

def round_list(lst):
    # Handle possible NaN or Inf
    return [round(x, 2) if not np.isnan(x) and not np.isinf(x) else 0.0 for x in lst]
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()

from app.schemas import AgentAnalysisResponse
from app.skills.skill_data_inventory import load_inventory_data
from app.skills.skill_data_io import (
    load_io_matrices, 
    extract_spatial_features, 
    extract_gmm_features, 
    extract_cross_lag_features, 
    extract_did_features
)
from app.skills.skill_model_spatial import execute_spatial_model_via_llm
from app.skills.skill_model_gmm import execute_gmm_model_via_llm
from app.skills.skill_model_cross_lag import execute_cross_lag_model_via_llm
from app.skills.skill_model_did import execute_did_model_via_llm

app = FastAPI(title="产业链因果分析 Agent API", version="1.0.0")

# 简单缓存机制
DATA_CACHE = {}

@app.on_event("startup")
def load_all_data():
    """启动时预加载数据"""
    try:
        # 请确保对应路径下有数据文件
        DATA_CACHE['inventory'] = load_inventory_data('data/all_industries_io_grouped_by_sector.csv')
        DATA_CACHE['io_matrices'] = load_io_matrices(
            ['data/IO_Matrix_21H2.csv', 'data/IO_Matrix_22H1.csv', 
             'data/IO_Matrix_22H2.csv', 'data/IO_Matrix_23H1.csv'],
            DATA_CACHE['inventory'].index.tolist()
        )
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Warning: Data load failed. Make sure data files are in 'data/' folder. Error: {e}")

@app.get("/api/v1/analyze/spatial", response_model=AgentAnalysisResponse)
def analyze_spatial(industry_name: str):
    """模型 1：空间面板模型分析"""
    inventory_df = DATA_CACHE.get('inventory')
    if inventory_df is None or industry_name not in inventory_df.index:
        raise HTTPException(status_code=404, detail="未找到产业数据，请检查行业名称或数据是否加载")
        
    y_data, wy_data = extract_spatial_features(industry_name, inventory_df, DATA_CACHE['io_matrices'])
    llm_result = execute_spatial_model_via_llm(industry_name, y_data, wy_data)
    
    return AgentAnalysisResponse(industry_name=industry_name, model_name="空间面板模型", llm_analysis=llm_result)

@app.get("/api/v1/analyze/gmm", response_model=AgentAnalysisResponse)
def analyze_gmm(industry_name: str):
    """模型 2：动态面板系统 GMM 分析"""
    inventory_df = DATA_CACHE.get('inventory')
    if inventory_df is None or industry_name not in inventory_df.index:
        raise HTTPException(status_code=404, detail="未找到产业数据")
        
    y_t, y_t_1 = extract_gmm_features(industry_name, inventory_df)
    llm_result = execute_gmm_model_via_llm(industry_name, y_t, y_t_1)
    
    return AgentAnalysisResponse(industry_name=industry_name, model_name="动态面板系统 GMM", llm_analysis=llm_result)

@app.get("/api/v1/analyze/cross-lag", response_model=AgentAnalysisResponse)
def analyze_cross_lag(industry_name: str):
    """模型 3：交叉滞后模型分析"""
    inventory_df = DATA_CACHE.get('inventory')
    if inventory_df is None or industry_name not in inventory_df.index:
        raise HTTPException(status_code=404, detail="未找到产业数据")
        
    d_data, s_data = extract_cross_lag_features(industry_name, DATA_CACHE['io_matrices'])
    llm_result = execute_cross_lag_model_via_llm(industry_name, d_data, s_data)
    
    return AgentAnalysisResponse(industry_name=industry_name, model_name="交叉滞后模型", llm_analysis=llm_result)

@app.get("/api/v1/analyze/did", response_model=AgentAnalysisResponse)
def analyze_did(industry_name: str, shock_name: str = "2022年全球供应链突发断裂"):
    """模型 4：网络双重差分分析"""
    inventory_df = DATA_CACHE.get('inventory')
    if inventory_df is None or industry_name not in inventory_df.index:
        raise HTTPException(status_code=404, detail="未找到产业数据")
        
    d_mean, s_mean = extract_did_features(industry_name, DATA_CACHE['io_matrices'])
    llm_result = execute_did_model_via_llm(industry_name, d_mean, s_mean, shock_name)
    
    return AgentAnalysisResponse(industry_name=industry_name, model_name="网络双重差分", llm_analysis=llm_result)
import os
import pandas as pd
import time
import json
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
from app.skills.skill_data_inventory import load_inventory_data
from app.skills.skill_data_io import (
    load_io_matrices, 
    extract_spatial_features, 
    extract_gmm_features, 
    extract_cross_lag_features, 
    extract_did_features,
    extract_forecast_features
)

from app.skills.skill_model_spatial import execute_spatial_model_via_llm
from app.skills.skill_model_gmm import execute_gmm_model_via_llm
from app.skills.skill_model_cross_lag import execute_cross_lag_model_via_llm
from app.skills.skill_model_did import execute_did_model_via_llm
from app.skills.skill_model_forecast import execute_forecast_model_via_llm
from app.skills.skill_model_synthesis import execute_synthesis_and_correction_via_llm

def calculate_deterministic_consistency(all_models_results: dict) -> dict:
    """
    Calculates the consensus and voting details across all LLM models 
    for two key evaluation dimensions.
    """
    dim1_votes = []
    dim2_votes = []
    
    for model_name, res in all_models_results.items():
        if isinstance(res, dict):
            dim1 = res.get("core_dimension_1", "error").strip()
            dim2 = res.get("core_dimension_2", "error").strip()
            dim1_votes.append(dim1)
            dim2_votes.append(dim2)
            
    dim1_counter = Counter(dim1_votes)
    dim2_counter = Counter(dim2_votes)
    
    top_dim1, count_dim1 = dim1_counter.most_common(1)[0] if dim1_counter else ("error", 0)
    top_dim2, count_dim2 = dim2_counter.most_common(1)[0] if dim2_counter else ("error", 0)
    
    total_models = len(all_models_results)
    
    def get_consensus_level(count):
        if count >= 4: return "High Consensus (>=80%)"
        if count == 3: return "Divergence (Majority Vote 60%)"
        return "Severe Conflict (No Consensus <60%)"

    return {
        "dim1_consensus": f"{top_dim1} ({count_dim1}/{total_models}) - {get_consensus_level(count_dim1)}",
        "dim2_consensus": f"{top_dim2} ({count_dim2}/{total_models}) - {get_consensus_level(count_dim2)}",
        "dim1_details": str(dict(dim1_counter)),
        "dim2_details": str(dict(dim2_counter))
    }

def main():
    print(">>> Loading local macroeconomic and industry data...")
    df_inv = load_inventory_data('data/inventory_difference.csv')
    industries = df_inv.index.tolist()
    io_matrices = load_io_matrices(
        ['data/IO_Matrix_21H2.csv', 'data/IO_Matrix_22H1.csv', 
         'data/IO_Matrix_22H2.csv', 'data/IO_Matrix_23H1.csv'],
        industries
    )

    # 1. Specify the list of target industries to process
    target_industries = [
        "Commodity Chemicals",
        "Fertilizers & Agricultural Chemicals",
        "Industrial Gases",
        "Paper Products",
        "Aerospace & Defense",
        "Construction & Engineering",
        "Heavy Electrical Equipment",
        "Construction Machinery & Heavy Transportation Equipment",
        "Household Appliances",
        "Health Care Equipment",
        "Pharmaceuticals",
        "Communications Equipment",
        "Technology Distributors",
        "Electric Utilities",
        "Water Utilities",
        "Renewable Electricity",
        "Technology Hardware, Storage & Peripherals",
        "Oil & Gas Equipment & Services",
        "Health Care Distributors",
        "Semiconductors",
        "Industrial Machinery & Supplies & Components"
    ]
    
    # 2. Safety filtering: Exclude industries that do not exist in the data table
    valid_targets = [ind for ind in target_industries if ind in industries]
    if not valid_targets:
        print("Error: None of the specified industry names exist in the data table. Please check spelling!")
        return

    # 3. Define the 5 expert LLM models
    EXPERT_MODELS = [
        "claude-sonnet-4-5",
        "gpt-5.4",
        "deepseek-reasoner",
        "glm-5",
        "kimi-k2.5"
    ]

    total = len(valid_targets)
    output_filename = "analysis_results.csv"
    
    print(f">>>> Data loading completed. Proceeding to run the full workflow for {total} industries across {len(EXPERT_MODELS)} models...")

    for i, industry_name in enumerate(valid_targets):
        print(f"\n=========================================================")
        print(f"Running [{i+1}/{total}]: {industry_name}")
        
        row_data = {
            "Sector": df_inv.loc[industry_name, 'Sector'],
            "Industry Code": df_inv.loc[industry_name, 'Industry Code'],
            "Industry Name": industry_name
        }
        
        all_models_synthesis_results = {}

        detailed_industry_log = {
             "Industry_Name": industry_name,
             "Sector": df_inv.loc[industry_name, 'Sector'],
             "Expert_Models_Details": {}
        }

        for current_model in EXPERT_MODELS:
            print(f"  ▶ Launching full workflow deduction for [{current_model}]...")
            raw_results = {}
            
            try:
                # 1. Spatial Panel Model
                y_data, wy_data = extract_spatial_features(industry_name, df_inv, io_matrices)
                res1 = execute_spatial_model_via_llm(industry_name, y_data, wy_data, model_name=current_model)
                raw_results["spatial_model"] = res1.get('summary', str(res1))
                
                # 2. Dynamic Panel System GMM
                y_t, y_t_1 = extract_gmm_features(industry_name, df_inv)
                res2 = execute_gmm_model_via_llm(industry_name, y_t, y_t_1, model_name=current_model)
                raw_results["gmm_model"] = res2.get('summary', str(res2))
                
                # 3. Cross-Lagged Model
                d_data, s_data = extract_cross_lag_features(industry_name, io_matrices)
                res3 = execute_cross_lag_model_via_llm(industry_name, d_data, s_data, model_name=current_model)
                raw_results["cross_lag_model"] = res3.get('summary', str(res3))
                
                # 4. Network Difference-in-Differences (DID)
                y_data, wy_data = extract_did_features(industry_name, io_matrices)
                res4 = execute_did_model_via_llm(industry_name, y_data, wy_data, model_name=current_model)
                raw_results["did_model"] = res4.get('summary', str(res4))

                # 5. Supply-Demand Forecasting & Bullwhip Effect Model
                y_data_fc, s_data_fc, d_data_fc = extract_forecast_features(industry_name, df_inv, io_matrices)
                res5 = execute_forecast_model_via_llm(industry_name, y_data_fc, s_data_fc, d_data_fc, model_name=current_model)
                raw_results["forecast_model"] = res5.get('summary', str(res5))
                
                # ================= [Expert Self-Review and Bias Correction] =================
                print(f" Raw results from the 5 models: {raw_results}")
                synthesis_res = execute_synthesis_and_correction_via_llm(industry_name, raw_results, model_name=current_model)
                
                # Store the model's structured JSON dictionary containing dimension tags into the main registry
                all_models_synthesis_results[current_model] = synthesis_res
                
                # Keep long-text final conclusions in separate columns for manual qualitative audits later
                row_data[f"{current_model}_Long_Conclusion"] = synthesis_res.get("final_synthesis", "No clear conclusion")
                row_data[f"{current_model}_Contradictions_Found"] = synthesis_res.get("contradictions_found", "None")

                # ===============================================================
                # [Update 2]: Archive full 5-dimensional raw results and expert corrections
                # ===============================================================
                detailed_industry_log["Expert_Models_Details"][current_model] = {
                    "Raw_5_Dimensions_Results": raw_results,
                    "Expert_Correction_Process": synthesis_res
                }
                
            except Exception as e:
                print(f"    [!] {current_model} crashed during processing: {e}")
                all_models_synthesis_results[current_model] = {}
                row_data[f"{current_model}_Long_Conclusion"] = f"Error: {str(e)}"
                
                # Log the crash details
                detailed_industry_log["Expert_Models_Details"][current_model] = {
                    "Error": str(e)
                }
            
            # API call buffer to prevent rate-limiting/concurrency cap triggers
            time.sleep(1) 

        # ===============================================================
        # [Update 3]: Export detailed logs into an individual JSON per industry
        # ===============================================================
        os.makedirs("detailed_reports", exist_ok=True) # Automatically generate container folder
        
        # Sanitize filename by replacing special characters like slashes and spaces
        safe_filename = industry_name.replace("/", "_").replace(" ", "_")
        detail_file_path = f"detailed_reports/{safe_filename}_detailed_process.json"
        
        with open(detail_file_path, "w", encoding="utf-8") as f:
            json.dump(detailed_industry_log, f, ensure_ascii=False, indent=4)
        print(f"  --> Detailed raw data and correction processes archived to: {detail_file_path}")
        # ===============================================================

        # ================= [Deterministic Consistency Alignment via Code] =================
        print(f"  --> All 5 models finished. Executing multi-dimensional voting alignment...")
        consistency_stats = calculate_deterministic_consistency(all_models_synthesis_results)
        
        # Append cross-voted consensus results to CSV structure
        row_data["Core_Dimension_1(Driver_Type)_Final_Consensus"] = consistency_stats["dim1_consensus"]
        row_data["Core_Dimension_1_Voting_Details"] = consistency_stats["dim1_details"]
        row_data["Core_Dimension_2(Inventory_Cycle)_Final_Consensus"] = consistency_stats["dim2_consensus"]
        row_data["Core_Dimension_2_Voting_Details"] = consistency_stats["dim2_details"]

        # ================= [Row-by-Row Incremental CSV Write] =================
        df_row = pd.DataFrame([row_data])
        if i == 0 and not os.path.exists(output_filename):
            df_row.to_csv(output_filename, index=False, encoding='utf-8-sig', mode='w')
        else:
            df_row.to_csv(output_filename, index=False, encoding='utf-8-sig', mode='a', header=False)

    print(f"\n All industry parallel batch processing and consensus verification completed! Analysis results appended to: {output_filename}")

if __name__ == "__main__":
    main()
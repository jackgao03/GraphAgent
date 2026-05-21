import os
import json
import re
from openai import OpenAI

from app.utils.llm_router import get_llm_client

def parse_json_safely(content: str) -> dict:
    """
    Ultimate safe parsing function v3.0 (XML tags + stack matching + regex fallback)
    """
    # 0. [New] XML tag interceptor: accurately extract content inside <final action>
    match = re.search(r'<final\s*action>(.*?)</final\s*action>', content, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1)
        
    # 1. Strip Markdown tags
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
        
    # 2. Find the first left brace boundaries
    start_idx = content.find('{')
    if start_idx == -1:
        raise ValueError("Starting brace '{' for JSON not found in the returned text")
        
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
                
    if end_idx == -1:
        raise ValueError("JSON braces are not closed")
        
    clean_json_str = content[start_idx:end_idx+1]
    
    # 3. Try standard parsing; if it fails, trigger regex fallback mechanism
    try:
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        result = {}
        keys = list(re.finditer(r'"([a-zA-Z0-9_]+)"\s*:', clean_json_str))
        
        if not keys:
            raise ValueError("JSON format is severely broken, and regex cannot extract any keys")
            
        for i in range(len(keys)):
            key = keys[i].group(1)
            start_val = keys[i].end()
            end_val = keys[i+1].start() if i + 1 < len(keys) else clean_json_str.rfind('}')
            
            val_str = clean_json_str[start_val:end_val].strip()
            
            if val_str.endswith(','):
                val_str = val_str[:-1].strip()
                
            if val_str.startswith('"') and val_str.endswith('"'):
                val_str = val_str[1:-1]
                val_str = val_str.replace('\\"', '"') 
            else:
                try:
                    if '.' in val_str:
                        val_str = float(val_str)
                    else:
                        val_str = int(val_str)
                except ValueError:
                    pass 
                    
            result[key] = val_str
        return result

def execute_synthesis_and_correction_via_llm(industry_name: str, results_dict: dict, model_name: str="gpt-4o") -> dict:
    """
    Receives the outputs from the five models, utilizes Chain of Thought (CoT) and XML tags for bias correction, and returns JSON.
    """
    client = get_llm_client(model_name)
    input_text = ""
    for k, v in results_dict.items():
        input_text += f"\n[{k} Conclusion]:\n{v}\n"

    # [New] Require the LLM to use <reasoning> and <final action> tags
    prompt = f"""You are a chief economist and top global supply chain expert. You need to review the calculation conclusions of five independent quantitative models for [{industry_name}], resolve any potential contradictions among them, and issue a final industry insight report.

### 1. Calculation and Analytical Perspectives of the Five Experts
You must understand the basis of the following five conclusions:
1. **Spatial Panel Model**: Calculates cross-sectional network spillovers (ρ). Assesses how directly this industry is infected by inventory fluctuations of other nodes in the upstream/downstream supply chain (co-directional resonance or reverse hedging).
2. **Dynamic Panel System GMM**: Calculates endogeneity and autocorrelation over time (α). Assesses the extent to which the industry's current inventory changes are driven by historical inertia from the previous period.
3. **Cross-Lagged Model**: Calculates causal direction. Determines whether the current industry dilemma or prosperity is "pulled by downstream demand" (Demand-Driven) or "pushed by upstream capacity" (Supply-Driven).
4. **Network DID (Difference-in-Differences) Model**: Introduces a real external macroeconomic shock (e.g., pandemic lockdowns, geopolitical conflicts). Calculates the absolute immunity and spillover scope of this industry when facing a black swan event.
5. **Supply-Demand Forecast & Bullwhip Effect Model**: Forward-looking projection (t+1). Determines whether the industry's inventory acts as a "Buffer" to absorb external shocks or an "Accelerator (Bullwhip Effect)" that amplifies supply-demand mismatches.

### 2. Common Sense Rules of Supply, Demand, and Inventory
When judging whether the five conclusions contradict each other, please strictly follow these economic theories:
1. **Kitchin Cycle**: The inventory cycle generally follows the pattern of "Active Destocking -> Passive Destocking -> Active Restocking -> Passive Restocking". For example: when demand suddenly drops and capacity is rigid, it must first experience "Passive Restocking"; in the early stages of demand recovery, it often manifests as "Passive Destocking".
2. **Bullwhip Effect**: The further upstream in the supply chain and further from end consumers, the more order fluctuations will be multiplied.
3. **Capacity Rigidity**: Asset-heavy or long-manufacturing-cycle industries (e.g., steel, chemicals, agriculture) have strong capacity building inertia. At the same time, when asset-heavy industries face shrinking demand, they would rather passively accumulate inventory than immediately shut down and cut off supply.
4. **Defensive Stockpiling**: When facing external shocks such as supply chain disruptions, a surge in upstream procurement by node enterprises is often not due to improving downstream demand, but out of panic for supply chain security (redundant stockpiling).

### 3. Original Model Analysis Results
{input_text}

### 4. Review and Output Format Requirements (Extremely Important)
To ensure the rigor of the analysis, you must follow this two-step output format:

Step 1: Write down your thought process inside the `<reasoning>` tag. Cross-validate the 5 conclusions above, clearly point out which conclusions contradict each other, and cite the common sense rules of supply/demand and inventory to correct them.
Step 2: After thinking, fill the finalized data into JSON, and **strictly place it inside the `<final action>` tag**.

**Please strictly follow this output template:**
<reasoning>
Write down your reasoning process, logical validation, and correction basis in detail here...
</reasoning>

<final action>
{{
    "logic_check": "Briefly state whether the logic between the five model conclusions can corroborate each other (Keep under 150 words)",
    "contradictions_found": "List the contradictory points found in the model conclusions, pointing out which deductions defy common sense. If none, write 'No obvious contradictions' (Keep under 100 words)",
    "correction_reasoning": "The final corrected reasoning conclusion regarding the contradictions above (Keep under 150 words)",
    "final_synthesis": "Synthesize the real logic after correction, provide the cycle stage and driver type of this industry in H1 2023, and analyze the final judgment of supply chain vulnerability and future supply/demand trends (Keep under 300 words)",
    "core_dimension_1": "Driver type in H1 2023, please output ONLY ONE of the following three phrases: Demand-Driven / Supply-Driven / Bi-directional-Driven",
    "core_dimension_2": "Cycle stage in H1 2023, please output ONLY ONE of the following four phrases: Active Restocking / Passive Restocking / Active Destocking / Passive Destocking"
}}
</final action>

### 5. Extremely Important Mandatory Output Rules
1. All your analysis and deductions must be conducted within `<reasoning>`, and the final results must ONLY be written in the corresponding JSON field values within `<final action>`.
2. Ensure the JSON format inside `<final action>` is absolutely valid, do not miss any commas!
3. If you need to use quotes inside the JSON text, please use single quotes (' ') or escaped quotes (\\"), do not break the outer double quote structure of the JSON!"""

    max_retries = 3
    error_feedback = "" 

    for attempt in range(max_retries):
        try:
            # 1. Dynamically build prompt: if failed previously, append the error as a warning
            current_prompt = prompt
            if error_feedback:
                current_prompt += f"\n\n[System Warning]: Your previous output parsing failed! Error reason: {error_feedback}. You must correct the format and strictly output valid JSON!"

            # 2. Make the API request
            res = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": "You are a rigorous chief expert in industry chain analysis. Please output the evaluation results according to the requirements."},
                    {"role": "user", "content": current_prompt}
                ], 
                temperature=0.2,
                stream=False
            )
            
            # 3. Extract content (Compatible with special cases like DeepSeek)
            content = res.choices[0].message.content or ""
            if "deepseek-reasoner" in model_name:
                reasoning = getattr(res.choices[0].message, 'reasoning_content', "") or ""
                if not content.strip() and reasoning.strip():
                    content = reasoning
            
            content = content.strip()
            if not content:
                raise ValueError("API returned empty data (possibly due to concurrency limits or everything was written in the reasoning tags)")
            
            # 4. Safe parsing (if this errors out, it jumps straight to except)
            parsed_dict = parse_json_safely(content)
              
            # If successful, return directly and break the loop!
            return parsed_dict

        except Exception as e:
            # Record the error to "teach" the LLM in the next iteration
            error_feedback = str(e)
            print(f"  {model_name} generated invalid format on attempt {attempt + 1}, retrying with error prompt...")
            
            if attempt == max_retries - 1:
                # Total failure after 3 attempts, gracefully return the final struggle result to prevent crash
                return {"summary": f"[Model crashed 3 consecutive times on format] Final error: {error_feedback}"}
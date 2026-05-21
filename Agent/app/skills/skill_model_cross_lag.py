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
        
    # 2. Find the first left brace
    start_idx = content.find('{')
    if start_idx == -1:
        raise ValueError("Starting brace '{' for JSON not found in the returned text")
        
    # 3. Stack structure brace matching
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
    
    # 4. Try standard parsing; if it fails, trigger regex fallback mechanism
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

def execute_cross_lag_model_via_llm(industry_name: str, d_data: list, s_data: list, model_name: str = "gpt-4o") -> dict:
    client = get_llm_client(model_name)
    prompt = f"""You are a senior Chinese industry and supply chain analysis expert. Please deduce the causal characteristics of [{industry_name}] based on the following cross-lagged model data and methodology.

### 1. Methodology and Core Formulas
- **Core Formulas**:
  Demand side: $D_{{it}} = \\alpha_1 D_{{i,t-1}} + \\beta_1 S_{{i,t-1}} + \\varepsilon_{{1it}}$
  Supply side: $S_{{it}} = \\alpha_2 S_{{i,t-1}} + \\beta_2 D_{{i,t-1}} + \\varepsilon_{{2it}}$
- **Core Logic**: This task only requires you to calculate $\\beta_1$ and $\\beta_2$ strictly according to the above formulas based on the given sequences, and judge whether the industry is "Demand-Driven", "Supply-Driven", or "Bi-directional-Driven".

### 2. Input Data (4 half-years from 21H2 to 23H1)
- Historical total demand amount from downstream clients sequence D_{{it}}: {d_data}
- Historical total procurement amount from suppliers sequence S_{{it}}: {s_data}

### 3. Analysis and Output Format Requirements (Extremely Important)
**Step 1**: Write down the rigorous calculation process inside the `<reasoning>` tag. **You must strictly calculate using the formulas, estimations or approximations are strictly prohibited:**
<reasoning>
Write the key calculation process and result analysis
</reasoning>

**Step 2**: Fill the final calculated precise data into JSON, strictly placed inside the `<final action>` tag.
<final action>
{{
    "exact_beta_1": "The precise value of β1 calculated using the formula (round to 4 decimal places)",
    "exact_beta_2": "The precise value of β2 calculated using the formula (round to 4 decimal places)",
    "drive_type": "Demand-Driven / Supply-Driven / Bi-directional-Driven",
    "reason": "Briefly describe the underlying commercial or economic reason for judging this drive type",
    "summary": "After strict formula calculation, the precise value of β1 is [exact_beta_1], and β2 is [exact_beta_2]. The industry exhibits [drive_type] characteristics because [reason]."
}}
</final action>

### 4. Extremely Important Mandatory Output Rules
1. All your analysis and deductions must be conducted within `<reasoning>`, and the final results must ONLY be written in the corresponding JSON field values within `<final action>`.
2. Ensure the JSON format inside `<final action>` is absolutely valid, do not miss any fields or commas!
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
                    {"role": "system", "content": "You are a rigorous industry chain analysis expert."},
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
            
            # 5. Fallback to supplement summary field
            if "summary" not in parsed_dict:
                parsed_dict["summary"] = "[Format Degradation Remediation] Model missing summary, residual data: " + str(parsed_dict)
                
            # If successful, return directly and break the loop!
            return parsed_dict

        except Exception as e:
            # Record the error to "teach" the LLM in the next iteration
            error_feedback = str(e)
            print(f"  {model_name} generated invalid format on attempt {attempt + 1}, retrying with error prompt...")
            
            if attempt == max_retries - 1:
                # Total failure after 3 attempts, gracefully return the final struggle result to prevent crash
                return {"summary": f"[Model crashed 3 consecutive times on format] Final error: {error_feedback}"}
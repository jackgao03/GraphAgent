
import os
from openai import OpenAI

def get_llm_client(model_name: str) -> OpenAI:

    if "gpt" in model_name:
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
    elif "deepseek" in model_name:
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        
    elif "claude" in model_name:
        return OpenAI(
            api_key=os.getenv("CLAUDE_API_KEY"),
            base_url=os.getenv("CLAUDE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        
    elif "glm" in model_name:
        return OpenAI(
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        )
    elif "kimi" in model_name:
        return OpenAI(
            api_key=os.getenv("Kimi_API_KEY"),
            base_url=os.getenv("Kimi_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        )   
    else:
        return OpenAI(
            api_key=os.getenv("PROXY_API_KEY"),
            base_url=os.getenv("PROXY_BASE_URL")
        )
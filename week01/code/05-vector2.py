import requests
import os

api_url = os.getenv('GLM_API_BASE_URL_EM')
api_key = os.getenv('GLM_API_KEY')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "input": "需要生成向量的文本",
    "model": "Embedding-2"  # 或 "deepseek-embedding-v1"
}

response = requests.post(api_url, headers=headers, json=data)
print(response)
#!/usr/bin/env python3
"""测试LLM调用（支持 OpenAI 和阿里云 DashScope）

API 选择基于模型名称自动判断：
- gpt-* 模型 → OpenAI API
- qwen-* 模型 → 阿里云 DashScope API
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.utils import config

def test_llm():
    """测试LLM调用，输出helloworld"""
    try:
        # 根据模型名获取 API 配置
        api_key, base_url, service_name = config.get_api_config()
        
        print(f"使用 {service_name} API")
        print(f"模型: {config.OPENAI_MODEL}")
        
        # 创建客户端 (openai 包支持 OpenAI 和 DashScope)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 构建提示
        prompt = "请输出'helloworld'，不要输出任何其他内容"
        
        # 调用LLM
        print("正在调用 LLM...")
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个简单的文本生成助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=100
        )
        
        # 提取生成的内容
        generated_content = response.choices[0].message.content.strip()
        
        print(f"LLM输出: {generated_content}")
        
        # 验证输出
        if "helloworld" in generated_content.lower().replace(" ", ""):
            print("测试成功！")
        else:
            print(f"测试完成，但输出与预期不同: {generated_content}")
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("\n请检查环境变量:")
        print("  - 使用 OpenAI (gpt-4等): 设置 OPENAI_API_KEY")
        print("  - 使用阿里云 (qwen等): 设置 DASHSCOPE_API_KEY")
        print("\n或创建 .env 文件 (参考 .env.example)")
    except Exception as e:
        print(f"LLM调用失败: {e}")
        print("测试失败！")

if __name__ == "__main__":
    test_llm()

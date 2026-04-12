#!/usr/bin/env python3
"""测试LLM调用（支持 OpenAI 和阿里云 DashScope）"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.utils import config

def test_llm():
    """测试LLM调用，输出helloworld"""
    try:
        # 检查 API Key
        if not config.API_KEY:
            print("错误: 未配置 API Key")
            print("请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量")
            print("或创建 .env 文件 (参考 .env.example)")
            return
        
        # 创建客户端 (openai 包支持 OpenAI 和 DashScope)
        client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL
        )
        
        # 显示使用的服务
        if config.OPENAI_API_KEY:
            print(f"使用 OpenAI API (模型: {config.OPENAI_MODEL})")
        else:
            print(f"使用阿里云 DashScope API (模型: {config.OPENAI_MODEL})")
        
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
        
    except Exception as e:
        print(f"LLM调用失败: {e}")
        print("测试失败！")

if __name__ == "__main__":
    test_llm()

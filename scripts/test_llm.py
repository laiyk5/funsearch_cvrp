#!/usr/bin/env python3
# 测试LLM调用（阿里通义千问大模型）

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.models import get_normal_client, ALI_TONGYI_TURBO_MODEL

def test_llm():
    """测试LLM调用，输出helloworld"""
    try:
        # 创建阿里通义千问大模型客户端
        client = get_normal_client()
        
        # 构建提示
        prompt = "请输出'helloworld'，不要输出任何其他内容"
        
        # 调用LLM
        print("正在调用阿里通义千问大模型...")
        response = client.chat.completions.create(
            model=ALI_TONGYI_TURBO_MODEL,
            messages=[
                {"role": "system", "content": "你是一个简单的文本生成助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        # 提取生成的内容
        generated_content = response.choices[0].message.content.strip()
        
        print(f"LLM输出: {generated_content}")
        print("测试成功！")
        
    except Exception as e:
        print(f"LLM调用失败: {e}")
        print("测试失败！")

if __name__ == "__main__":
    test_llm()

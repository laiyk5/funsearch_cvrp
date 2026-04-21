import logging

from openai import OpenAI

def check_llm(client: OpenAI, model: str):
    """测试LLM调用，输出helloworld"""
    try:
        # 构建提示
        prompt = "请输出'helloworld'，不要输出任何其他内容"
        
        # 调用LLM
        logging.info("正在调用 LLM...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个简单的文本生成助手"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100
        )
        
        # 提取生成的内容
        generated_content = response.choices[0].message.content

        if generated_content is None:
            logging.warning("LLM没有返回内容")
            generated_content = ""
        else:
            generated_content = generated_content.strip()
        
        logging.info(f"LLM输出: {generated_content}")
        
        # 验证输出
        if "helloworld" in generated_content.lower().replace(" ", ""):
            logging.info("测试成功！")
        else:
            logging.info(f"测试完成，但输出与预期不同: {generated_content}")
        
    except ValueError as e:
        logging.error(f"配置错误: {e}")
        logging.info("\n请检查环境变量:")
        raise e
    

from funsearch_cvrp.cvrp.io import load_cvrplib_instance

from .. import config

import logging

from openai import OpenAI


from funsearch_cvrp.config import config

def check_llm(client: OpenAI):
    """测试LLM调用，输出helloworld"""
    try:
        # 构建提示
        prompt = "请输出'helloworld'，不要输出任何其他内容"
        
        # 调用LLM
        logging.info("正在调用 LLM...")
        response = client.chat.completions.create(
            model=config["LLM"].get("MODEL", ""),
            messages=[
                {"role": "system", "content": "你是一个简单的文本生成助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=config["LLM"].getfloat("LLM_TEMPERATURE", 0.7),
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
    

def check_load_cvrplib(path: str):
    """测试是否能正确加载CVRPLib格式的实例"""
    from funsearch_cvrp.cvrp.io import load_cvrplib_folder
    try:
        instance = load_cvrplib_folder(path)
        logging.info(f"成功加载实例: {instance}")
    except Exception as e:
        logging.error(f"加载实例失败: {e}")
        raise e
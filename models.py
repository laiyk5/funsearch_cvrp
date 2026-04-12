# models.py
# 可用模型列表，以及获得访问模型的客户端
#     实际使用时可以根据自己的实际情况调整

# 阿里的通义千问大模型
#    官网: https://bailian.console.aliyun.com/#/home
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen3-max-2025-09-23"
ALI_TONGYI_TURBO_MODEL = "qwen3-max-2025-09-23"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_R10528 = "deepseek-r1-0528"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING_V3 = "text-embedding-v3"
ALI_TONGYI_EMBEDDING_V4 = "text-embedding-v4"

# DeepSeek
#   官网：https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY_OS_VAR_NAME = "DEEPSEEK_API_KEY"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"

# 腾讯混元
'''
#   官网：https://hunyuan.cloud.tencent.com/#/app/modelSquare
TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME = "HUNYUAN_API_KEY"
TENCENT_HUNYUAN_URL = "https://api.hunyuan.cloud.tencent.com/v1"
TENCENT_HUNYUAN_TURBO_MODEL = "hunyuan-turbos-latest"
TENCENT_HUNYUAN_REASONER_MODEL = "hunyuan-t1-latest"
TENCENT_HUNYUAN_LONGCONTEXT_MODEL = "hunyuan-large-longcontext"
# TENCENT_HUNYUAN_EMBEDDING = "hunyuan-embedding"
# TENCENT_SECRET_ID_OS_VAR_NAME = "Tencent_SecretId"
# TENCENT_SECRET_KEY_OS_VAR_NAME = "Tencent_SecretKey"
'''


import os
from openai import OpenAI
import inspect


# 使用原生api获得指定平台的客户端 (默认是：阿里通义千问)
def get_normal_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                      base_url=ALI_TONGYI_URL,
                      verbose=False, debug=False):
    """
    使用原生api获得指定平台的客户端，但未指定具体模型，缺省平台为阿里云百炼
    也可以通过传入api_key，base_url两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url}")
    if (debug):
        print(f"{function_name}-平台：{base_url},key：{api_key}")
    return OpenAI(api_key=api_key, base_url=base_url)

"""
配置文件：读取环境变量和配置信息
"""
import os
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用配置类"""
    
    # SD模型配置
    SD_MODEL_PATH: str = os.getenv("SD_MODEL_PATH", "./models/stable-diffusion-v1-5")
    
    # LoRA配置
    LORA_MODELS: Optional[List[Dict[str, Any]]] = None
    LORA_MODELS_DIR: Optional[str] = os.getenv("LORA_MODELS_DIR")
    LORA_MODELS_LIST: Optional[str] = os.getenv("LORA_MODELS_LIST")
    
    # 默认采样方法
    DEFAULT_SCHEDULER: Optional[str] = os.getenv("DEFAULT_SCHEDULER")
    
    # MinIO配置
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "sd-images")
    
    # API认证配置
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    API_KEYS = [key.strip() for key in API_KEYS if key.strip()]  # 清理空字符串
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    
    # 运行设备
    DEVICE: str = os.getenv("DEVICE", "cuda")
    
    # 回调配置
    CALLBACK_RETRY_TIMES: int = int(os.getenv("CALLBACK_RETRY_TIMES", "3"))
    CALLBACK_RETRY_INTERVAL: int = int(os.getenv("CALLBACK_RETRY_INTERVAL", "5"))
    
    # 健康检查接口配置
    HEALTH_CHECK_NO_AUTH: bool = os.getenv("HEALTH_CHECK_NO_AUTH", "false").lower() == "true"
    
    # 任务数据库配置
    TASK_DB_PATH: str = os.getenv("TASK_DB_PATH", "tasks.db")
    
    @classmethod
    def load_lora_models(cls) -> Optional[List[Dict[str, Any]]]:
        """加载LoRA模型配置"""
        if cls.LORA_MODELS is not None:
            return cls.LORA_MODELS
        
        # 方式1: 从LORA_MODELS环境变量读取JSON字符串
        lora_models_str = os.getenv("LORA_MODELS")
        if lora_models_str:
            try:
                cls.LORA_MODELS = json.loads(lora_models_str)
                return cls.LORA_MODELS
            except json.JSONDecodeError:
                pass
        
        # 方式2: 从LORA_MODELS_DIR + LORA_MODELS_LIST配置
        if cls.LORA_MODELS_DIR and cls.LORA_MODELS_LIST:
            lora_list = [f.strip() for f in cls.LORA_MODELS_LIST.split(",") if f.strip()]
            cls.LORA_MODELS = []
            for lora_file in lora_list:
                lora_path = Path(cls.LORA_MODELS_DIR) / lora_file
                if lora_path.exists():
                    cls.LORA_MODELS.append({
                        "path": str(lora_path),
                        "weight": 1.0
                    })
            if cls.LORA_MODELS:
                return cls.LORA_MODELS
        
        return None
    
    @classmethod
    def validate(cls) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        if not cls.SD_MODEL_PATH:
            errors.append("SD_MODEL_PATH未配置")
        elif not Path(cls.SD_MODEL_PATH).exists():
            errors.append(f"SD模型路径不存在: {cls.SD_MODEL_PATH}")
        
        if not cls.MINIO_ENDPOINT:
            errors.append("MINIO_ENDPOINT未配置")
        
        if not cls.MINIO_ACCESS_KEY or not cls.MINIO_SECRET_KEY:
            errors.append("MinIO访问密钥未配置")
        
        if not cls.MINIO_BUCKET:
            errors.append("MINIO_BUCKET未配置")
        
        if not cls.API_KEYS:
            errors.append("API_KEYS未配置，至少需要配置一个API Key")
        
        return errors


# 加载LoRA配置
Config.load_lora_models()

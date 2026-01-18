"""
配置文件：从YAML文件读取配置信息
"""
import yaml
from typing import List, Dict, Optional, Any
from pathlib import Path
from app.utils.logger import logger


class Config:
    """应用配置类"""
    
    # 配置文件路径
    CONFIG_FILE: str = "config.yaml"
    
    # 配置数据
    _config_data: Dict[str, Any] = {}
    
    # SD模型配置
    SD_MODEL_PATH: str = "./models/stable-diffusion-v1-5"
    
    # LoRA配置
    LORA_MODELS: Optional[List[Dict[str, Any]]] = None
    LORA_MODELS_DIR: Optional[str] = None
    LORA_MODELS_LIST: Optional[str] = None
    
    # 默认采样方法
    DEFAULT_SCHEDULER: Optional[str] = None
    
    # MinIO配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "sd-images"
    
    # API认证配置
    API_KEYS: List[str] = []
    API_KEY_HEADER: str = "X-API-Key"
    
    # 运行设备
    DEVICE: str = "cuda"
    
    # 回调配置
    CALLBACK_RETRY_TIMES: int = 3
    CALLBACK_RETRY_INTERVAL: int = 5
    
    # 健康检查接口配置
    HEALTH_CHECK_NO_AUTH: bool = True
    
    # 任务数据库配置
    TASK_DB_PATH: str = "tasks.db"
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> None:
        """从YAML文件加载配置"""
        if config_path:
            cls.CONFIG_FILE = config_path
        
        config_file = Path(cls.CONFIG_FILE)
        if not config_file.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {cls.CONFIG_FILE}\n"
                f"请创建配置文件或使用 config.yaml.example 作为模板"
            )
        
        with open(config_file, 'r', encoding='utf-8') as f:
            cls._config_data = yaml.safe_load(f) or {}
        
        # 加载SD模型配置
        sd_config = cls._config_data.get("sd_model", {})
        cls.SD_MODEL_PATH = sd_config.get("path", cls.SD_MODEL_PATH)
        
        # 加载LoRA配置
        lora_config = cls._config_data.get("lora", {})
        cls.LORA_MODELS = lora_config.get("models")
        cls.LORA_MODELS_DIR = lora_config.get("models_dir")
        cls.LORA_MODELS_LIST = lora_config.get("models_list")
        
        # 加载默认采样方法
        cls.DEFAULT_SCHEDULER = cls._config_data.get("default_scheduler")
        
        # 加载MinIO配置
        minio_config = cls._config_data.get("minio", {})
        cls.MINIO_ENDPOINT = minio_config.get("endpoint", cls.MINIO_ENDPOINT)
        cls.MINIO_ACCESS_KEY = minio_config.get("access_key", cls.MINIO_ACCESS_KEY)
        cls.MINIO_SECRET_KEY = minio_config.get("secret_key", cls.MINIO_SECRET_KEY)
        cls.MINIO_BUCKET = minio_config.get("bucket", cls.MINIO_BUCKET)
        
        # 加载API认证配置
        api_config = cls._config_data.get("api", {})
        api_keys = api_config.get("keys", [])
        if isinstance(api_keys, str):
            api_keys = [key.strip() for key in api_keys.split(",") if key.strip()]
        cls.API_KEYS = api_keys if isinstance(api_keys, list) else []
        cls.API_KEY_HEADER = api_config.get("key_header", cls.API_KEY_HEADER)
        
        # 加载运行设备
        cls.DEVICE = cls._config_data.get("device", cls.DEVICE)
        
        # 加载回调配置
        callback_config = cls._config_data.get("callback", {})
        cls.CALLBACK_RETRY_TIMES = callback_config.get("retry_times", cls.CALLBACK_RETRY_TIMES)
        cls.CALLBACK_RETRY_INTERVAL = callback_config.get("retry_interval", cls.CALLBACK_RETRY_INTERVAL)
        
        # 加载健康检查配置
        cls.HEALTH_CHECK_NO_AUTH = cls._config_data.get("health_check", {}).get("no_auth", cls.HEALTH_CHECK_NO_AUTH)
        
        # 加载任务数据库配置
        cls.TASK_DB_PATH = cls._config_data.get("task_db", {}).get("path", cls.TASK_DB_PATH)
        
        # 加载LoRA模型
        cls.load_lora_models()
    
    @classmethod
    def load_lora_models(cls) -> Optional[List[Dict[str, Any]]]:
        """加载LoRA模型配置"""
        # 方式1: 从配置中直接读取LORA_MODELS列表（已设置）
        if cls.LORA_MODELS is not None:
            return cls.LORA_MODELS if cls.LORA_MODELS else None
        
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
        
        # 未配置LoRA模型
        cls.LORA_MODELS = []
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


# 加载配置
try:
    Config.load_config()
except FileNotFoundError as e:
    logger.warning(f"警告: {e}")
    logger.warning("使用默认配置，请创建 config.yaml 文件")

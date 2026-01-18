"""
日志配置模块
统一管理应用的日志输出，抑制第三方库的详细日志
"""
import logging
import sys
from pathlib import Path

# 配置日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 创建日志目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(
    name: str = "sd_api",
    level: int = logging.INFO,
    log_file: bool = True,
    console: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 是否输出到文件
        console: 是否输出到控制台
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        file_handler = logging.FileHandler(
            LOG_DIR / f"{name}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def suppress_third_party_logs():
    """
    抑制第三方库的详细日志输出
    只保留 WARNING 及以上级别的日志
    """
    # 抑制 diffusers 库的详细日志
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    
    # 抑制 transformers 库的详细日志
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    # 抑制 huggingface_hub 库的详细日志
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    # 抑制 torch 相关库的详细日志
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # 抑制 PIL 库的日志
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # 抑制 urllib3 的详细日志（httpx 等库使用）
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # 抑制 httpx 的详细日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # 抑制 minio 的详细日志
    logging.getLogger("minio").setLevel(logging.WARNING)
    
    # 抑制 asyncio 的详细日志
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # 抑制 fastapi 的详细日志（可选，如果需要更详细的API日志可以注释掉）
    # logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    # 抑制 uvicorn 的访问日志（如果需要可以单独配置）
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# 创建默认的日志记录器
logger = setup_logger()

# 抑制第三方库日志
suppress_third_party_logs()

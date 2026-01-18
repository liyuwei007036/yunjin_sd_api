"""
回调URL推送服务
"""
import asyncio
import httpx
from typing import Optional, List
from urllib.parse import urlparse
from app.config import Config
from app.utils.logger import logger


class CallbackService:
    """回调服务类"""
    
    def __init__(self):
        """初始化回调服务"""
        self.retry_times = Config.CALLBACK_RETRY_TIMES
        self.retry_interval = Config.CALLBACK_RETRY_INTERVAL
    
    async def send_callback(
        self,
        callback_url: str,
        task_id: str,
        status: str,
        image_url: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        error_message: Optional[str] = None
    ):
        """
        发送回调通知
        
        Args:
            callback_url: 回调URL
            task_id: 任务ID
            status: 任务状态 (completed/failed)
            image_url: 单张图片URL（当num_images=1时）
            image_urls: 图片URL列表（当num_images>1时）
            error_message: 错误信息（失败时）
        """
        # 验证callback_url是否有效
        if not callback_url or not callback_url.strip():
            logger.warning(f"回调URL为空，跳过推送: task_id={task_id}")
            return
        
        # 检查URL格式
        try:
            parsed = urlparse(callback_url)
            if not parsed.scheme or parsed.scheme not in ["http", "https"]:
                logger.warning(f"回调URL格式无效（缺少协议）: task_id={task_id}, callback_url={callback_url}")
                return
            if not parsed.netloc:
                logger.warning(f"回调URL格式无效（缺少域名）: task_id={task_id}, callback_url={callback_url}")
                return
        except Exception as e:
            logger.warning(f"回调URL格式验证失败: task_id={task_id}, callback_url={callback_url}, error={e}")
            return
        
        # 构造回调数据
        callback_data = {
            "task_id": task_id,
            "status": status
        }
        
        if status == "completed":
            if image_url:
                callback_data["image_url"] = image_url
            elif image_urls:
                callback_data["image_urls"] = image_urls
        elif status == "failed" and error_message:
            callback_data["error_message"] = error_message
        
        # 重试推送
        for attempt in range(self.retry_times):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        callback_url,
                        json=callback_data,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    logger.info(f"回调推送成功: task_id={task_id}, callback_url={callback_url}")
                    return  # 成功则返回
            except Exception as e:
                if attempt < self.retry_times - 1:
                    logger.warning(f"回调推送失败（尝试 {attempt + 1}/{self.retry_times}）: {e}, 将在{self.retry_interval}秒后重试")
                    await asyncio.sleep(self.retry_interval)
                else:
                    logger.error(f"回调推送最终失败: task_id={task_id}, callback_url={callback_url}, error={e}")


# 全局回调服务实例
callback_service = CallbackService()

"""
任务状态模型
"""
from typing import Optional, Union, List
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """任务数据模型"""
    task_id: str
    status: TaskStatus
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result_url: Optional[str] = None
    result_urls: Optional[List[str]] = None
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    
    def update_status(self, status: TaskStatus):
        """更新任务状态"""
        self.status = status
        self.updated_at = datetime.now()
    
    def set_result(self, url: Union[str, List[str]]):
        """设置任务结果"""
        if isinstance(url, list):
            self.result_urls = url
        else:
            self.result_url = url
        self.update_status(TaskStatus.COMPLETED)
    
    def set_error(self, error_message: str):
        """设置错误信息"""
        self.error_message = error_message
        self.update_status(TaskStatus.FAILED)

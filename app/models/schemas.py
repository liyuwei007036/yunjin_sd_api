"""
Pydantic数据模型：请求/响应Schema
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """图片生成请求模型"""
    prompt: str = Field(..., description="提示词")
    init_image: Optional[str] = Field(None, description="初始图片base64编码，如果提供则执行图生图，否则执行文生图")
    negative_prompt: Optional[str] = Field(None, description="负面提示词")
    width: Optional[int] = Field(None, description="图片宽度")
    height: Optional[int] = Field(None, description="图片高度")
    num_inference_steps: Optional[int] = Field(None, description="推理步数")
    guidance_scale: Optional[float] = Field(None, description="引导强度")
    strength: Optional[float] = Field(None, description="图生图强度（仅图生图）")
    scheduler: Optional[str] = Field(None, description="采样方法/scheduler名称")
    num_images: Optional[int] = Field(1, ge=1, le=10, description="生成图片张数，默认为1")
    seed: Optional[int] = Field(None, description="随机种子，用于控制生成结果的可复现性")
    output_format: Optional[Literal["png", "jpg", "jpeg"]] = Field("png", description="输出图片格式")
    callback_url: Optional[str] = Field(None, description="回调URL，生成完成后将结果推送到该URL")
    
    @field_validator("num_images")
    @classmethod
    def validate_num_images(cls, v):
        if v and (v < 1 or v > 10):
            raise ValueError("num_images必须在1-10之间")
        return v or 1


class GenerateResponse(BaseModel):
    """图片生成响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field("pending", description="任务状态")
    message: str = Field("任务已创建", description="响应消息")


class TaskStatusResponse(BaseModel):
    """任务状态查询响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    result_url: Optional[str] = Field(None, description="单张图片URL（当num_images=1且完成时）")
    result_urls: Optional[List[str]] = Field(None, description="图片URL列表（当num_images>1且完成时）")
    error_message: Optional[str] = Field(None, description="错误信息（失败时）")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field("ok", description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")

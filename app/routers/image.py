"""
图片生成API路由
"""
import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from app.models.schemas import GenerateRequest, GenerateResponse, TaskStatusResponse
from app.models.task import TaskStatus
from app.services.sd_service import SDService
from app.services.oss_service import OSSService
from app.services.callback_service import CallbackService
from app.utils.task_manager import task_manager
from app.auth import require_auth
from app.config import Config

router = APIRouter(prefix="/api/v1", tags=["image"])

# 服务实例（全局单例）
sd_service: Optional[SDService] = None
oss_service: Optional[OSSService] = None
callback_service: Optional[CallbackService] = None


def get_sd_service() -> SDService:
    """获取SD服务实例（单例）"""
    global sd_service
    if sd_service is None:
        sd_service = SDService()
    return sd_service


def get_oss_service() -> OSSService:
    """获取OSS服务实例（单例）"""
    global oss_service
    if oss_service is None:
        oss_service = OSSService()
    return oss_service


def get_callback_service() -> CallbackService:
    """获取回调服务实例（单例）"""
    global callback_service
    if callback_service is None:
        callback_service = CallbackService()
    return callback_service


async def generate_image_task(
    task_id: str,
    prompt: str,
    init_image: Optional[str],
    negative_prompt: Optional[str],
    num_images: int,
    scheduler: Optional[str],
    seed: Optional[int],
    output_format: str,
    width: Optional[int],
    height: Optional[int],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    strength: Optional[float],
    callback_url: Optional[str],
):
    """
    异步图片生成任务
    
    Args:
        task_id: 任务ID
        prompt: 提示词
        init_image: 初始图片（base64）
        negative_prompt: 负面提示词
        num_images: 生成图片张数
        scheduler: 采样方法
        seed: 随机种子
        output_format: 输出格式
        width: 图片宽度
        height: 图片高度
        num_inference_steps: 推理步数
        guidance_scale: 引导强度
        strength: 图生图强度
        callback_url: 回调URL
    """
    # 更新任务状态为处理中
    task_manager.update_status(task_id, TaskStatus.PROCESSING)
    
    try:
        sd_svc = get_sd_service()
        oss_svc = get_oss_service()
        cb_svc = get_callback_service()
        
        # 根据是否有init_image决定生成方式
        if init_image:
            # 图生图
            images = sd_svc.image_to_image(
                prompt=prompt,
                init_image=init_image,
                negative_prompt=negative_prompt,
                num_images=num_images,
                scheduler=scheduler,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            )
        else:
            # 文生图
            images = sd_svc.text_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=num_images,
                scheduler=scheduler,
                seed=seed,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        
        # 处理单张或多张图片
        if num_images == 1:
            # 单张图片：返回单个URL
            image = images[0] if isinstance(images, list) else images
            url = oss_svc.upload_image(image, output_format=output_format)
            task_manager.complete_task(task_id, url)
            
            # 如果有回调URL，异步推送结果
            if callback_url:
                await cb_svc.send_callback(
                    callback_url=callback_url,
                    task_id=task_id,
                    status="completed",
                    image_url=url
                )
        else:
            # 多张图片：返回URL列表
            urls = oss_svc.upload_images(images, output_format=output_format)
            task_manager.complete_task(task_id, urls)
            
            # 如果有回调URL，异步推送结果
            if callback_url:
                await cb_svc.send_callback(
                    callback_url=callback_url,
                    task_id=task_id,
                    status="completed",
                    image_urls=urls
                )
    except Exception as e:
        error_msg = str(e)
        print(f"图片生成失败: task_id={task_id}, error={error_msg}")
        task_manager.fail_task(task_id, error_msg)
        
        # 失败时也推送回调
        if callback_url:
            cb_svc = get_callback_service()
            await cb_svc.send_callback(
                callback_url=callback_url,
                task_id=task_id,
                status="failed",
                error_message=error_msg
            )


@router.post("/generate", response_model=GenerateResponse, summary="图片生成接口")
async def generate_image(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(require_auth),
):
    """
    统一的图片生成接口（文生图/图生图）
    
    - 如果提供init_image参数，则执行图生图
    - 如果不提供init_image参数，则执行文生图
    - 返回task_id，任务将在后台异步执行
    """
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务
    task_manager.create_task(task_id, callback_url=request.callback_url)
    
    # 添加后台任务
    background_tasks.add_task(
        generate_image_task,
        task_id=task_id,
        prompt=request.prompt,
        init_image=request.init_image,
        negative_prompt=request.negative_prompt,
        num_images=request.num_images or 1,
        scheduler=request.scheduler,
        seed=request.seed,
        output_format=request.output_format or "png",
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        strength=request.strength,
        callback_url=request.callback_url,
    )
    
    return GenerateResponse(
        task_id=task_id,
        status="pending",
        message="任务已创建，正在处理中"
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse, summary="查询任务状态")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(require_auth),
):
    """
    查询任务状态
    
    返回任务当前状态和结果（如果已完成）
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 构建响应
    response_data = {
        "task_id": task.task_id,
        "status": task.status.value,
        "result_url": task.result_url,
        "result_urls": task.result_urls,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }
    
    return TaskStatusResponse(**response_data)

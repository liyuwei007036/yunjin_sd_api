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
from app.services.llm_service import get_llm_service
from app.utils.task_manager import task_manager
from app.auth import require_auth
from app.utils.logger import logger
from app.config import Config
import gc

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


def cleanup_services():
    """清理所有服务实例的资源"""
    global sd_service, oss_service, callback_service
    logger.info("开始清理服务实例...")
    
    if sd_service is not None:
        try:
            sd_service.cleanup()
            del sd_service
            sd_service = None
            logger.info("SD服务已清理")
        except Exception as e:
            logger.error(f"清理SD服务时出错: {e}", exc_info=True)
    
    if oss_service is not None:
        try:
            # OSS服务通常不需要特殊清理，但可以删除引用
            del oss_service
            oss_service = None
            logger.info("OSS服务已清理")
        except Exception as e:
            logger.error(f"清理OSS服务时出错: {e}", exc_info=True)
    
    if callback_service is not None:
        try:
            del callback_service
            callback_service = None
            logger.info("回调服务已清理")
        except Exception as e:
            logger.error(f"清理回调服务时出错: {e}", exc_info=True)
    
    # 强制垃圾回收
    gc.collect()
    
    logger.info("所有服务实例已清理")


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
        
        # 如果用户手动提供了prompt（不是通过LLM转换），自动添加LoRA触发词
        # 注意：LLM转换的prompt已经在llm_service中添加了触发词
        lora_trigger_words = Config.get_lora_trigger_words()
        if lora_trigger_words and prompt:
            prompt_lower = prompt.lower()
            has_trigger = any(trigger.lower() in prompt_lower for trigger in lora_trigger_words)
            if not has_trigger:
                trigger_str = ", ".join(lora_trigger_words)
                prompt = f"{trigger_str}, {prompt}"
                logger.info(f"已自动为手动prompt添加LoRA触发词: {trigger_str}")
        
        # 根据是否有init_image决定生成方式
        if init_image:
            # 图生图
            logger.info(f"开始图生图任务: task_id={task_id}, prompt={prompt[:50]}..., num_images={num_images}")
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
            logger.info(f"图生图完成: task_id={task_id}, 生成图片数量={len(images) if isinstance(images, list) else 1}")
        else:
            # 文生图
            logger.info(f"开始文生图任务: task_id={task_id}, prompt={prompt[:50]}..., num_images={num_images}, width={width}, height={height}")
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
            logger.info(f"文生图完成: task_id={task_id}, 生成图片数量={len(images) if isinstance(images, list) else 1}")
        
        # 处理单张或多张图片
        if num_images == 1:
            # 单张图片：返回单个URL
            image = images[0] if isinstance(images, list) else images
            logger.info(f"开始上传图片到OSS: task_id={task_id}, format={output_format}")
            url = oss_svc.upload_image(image, output_format=output_format)
            logger.info(f"图片生成成功: task_id={task_id}, url={url}")
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
            logger.info(f"开始上传图片到OSS: task_id={task_id}, num_images={num_images}, format={output_format}")
            urls = oss_svc.upload_images(images, output_format=output_format)
            logger.info(f"图片生成成功: task_id={task_id}, num_images={len(urls)}, urls={urls}")
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
        logger.error(f"图片生成失败: task_id={task_id}, error={error_msg}", exc_info=True)
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
    - 如果提供natural_language，将自动转换为prompt和negative_prompt
    - 返回task_id，任务将在后台异步执行
    """
    # 处理自然语言转换
    prompt = request.prompt
    negative_prompt = request.negative_prompt
    
    # 判断是否为图生图任务
    is_img2img = bool(request.init_image and request.init_image.strip())
    
    if request.natural_language:
        # 使用自然语言，需要转换为提示词
        llm_service = get_llm_service()
        if not llm_service:
            raise HTTPException(
                status_code=503,
                detail="LLM服务未配置，无法使用自然语言输入功能。请在config.yaml中配置llm相关参数，或使用prompt和negative_prompt字段"
            )
        
        try:
            task_type = "图生图" if is_img2img else "文生图"
            logger.info(f"开始LLM转换 ({task_type}): natural_language='{request.natural_language[:50]}...'")
            
            # 如果是图生图且有原图，使用基于原图识别的方法
            if is_img2img and request.init_image:
                logger.info("使用基于原图识别的LLM转换方法，生成贴近原图的prompt")
                converted_prompt, converted_negative_prompt = llm_service.convert_img2img_prompts_with_image(
                    request.natural_language,
                    request.init_image
                )
            else:
                # 普通模式（文生图或没有原图的图生图）
                converted_prompt, converted_negative_prompt = llm_service.convert_to_prompts(
                    request.natural_language, 
                    is_img2img=is_img2img
                )
            
            # 如果用户提供了手动prompt，则使用用户提供的（覆盖LLM生成的）
            if not prompt or not prompt.strip():
                prompt = converted_prompt
            if not negative_prompt or not negative_prompt.strip():
                negative_prompt = converted_negative_prompt
            
            logger.info(f"LLM转换完成 ({task_type}): prompt长度={len(prompt)}, negative_prompt长度={len(negative_prompt)}")
        except Exception as e:
            error_msg = f"自然语言转换失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    
    # 确保prompt不为空
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="必须提供prompt或natural_language字段")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务，保存prompt信息
    task_manager.create_task(
        task_id, 
        callback_url=request.callback_url,
        prompt=prompt,
        negative_prompt=negative_prompt
    )
    
    # 添加后台任务
    background_tasks.add_task(
        generate_image_task,
        task_id=task_id,
        prompt=prompt,
        init_image=request.init_image,
        negative_prompt=negative_prompt,
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
        "prompt": task.prompt,
        "negative_prompt": task.negative_prompt,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }
    
    return TaskStatusResponse(**response_data)

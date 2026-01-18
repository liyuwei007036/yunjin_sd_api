"""
SD模型生成服务（支持LoRA）
"""
import torch
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from app.config import Config


# Scheduler映射表
SCHEDULER_MAP = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "DDIMScheduler": DDIMScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "PNDMScheduler": PNDMScheduler,
    "LMSDiscreteScheduler": LMSDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
    "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
    "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
}


class SDService:
    """SD模型服务类"""
    
    def __init__(self):
        """初始化SD模型服务"""
        self.device = Config.DEVICE if torch.cuda.is_available() and Config.DEVICE == "cuda" else "cpu"
        self.model_path = Config.SD_MODEL_PATH
        self.lora_models = Config.load_lora_models()
        self.default_scheduler = Config.DEFAULT_SCHEDULER
        
        # 初始化pipeline
        self.text2img_pipeline = None
        self.img2img_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """加载SD模型和LoRA"""
        print(f"加载SD模型: {self.model_path}, 设备: {self.device}")
        
        # 加载文生图pipeline
        self.text2img_pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        # 加载图生图pipeline
        self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        # 加载LoRA模型
        if self.lora_models:
            print(f"加载LoRA模型: {len(self.lora_models)}个")
            for lora_config in self.lora_models:
                lora_path = lora_config.get("path")
                lora_weight = lora_config.get("weight", 1.0)
                if lora_path:
                    print(f"  - 加载LoRA: {lora_path}, 权重: {lora_weight}")
                    self.text2img_pipeline.load_lora_weights(lora_path, weight=lora_weight)
                    self.img2img_pipeline.load_lora_weights(lora_path, weight=lora_weight)
        
        # 移动到设备
        self.text2img_pipeline = self.text2img_pipeline.to(self.device)
        self.img2img_pipeline = self.img2img_pipeline.to(self.device)
        
        # 设置默认scheduler
        if self.default_scheduler and self.default_scheduler in SCHEDULER_MAP:
            scheduler_class = SCHEDULER_MAP[self.default_scheduler]
            scheduler = scheduler_class.from_config(self.text2img_pipeline.scheduler.config)
            self.text2img_pipeline.scheduler = scheduler
            self.img2img_pipeline.scheduler = scheduler
        
        print("SD模型加载完成")
    
    def _get_scheduler(self, scheduler_name: Optional[str] = None):
        """获取scheduler实例"""
        if scheduler_name and scheduler_name in SCHEDULER_MAP:
            scheduler_class = SCHEDULER_MAP[scheduler_name]
            return scheduler_class.from_config(self.text2img_pipeline.scheduler.config)
        return None
    
    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """解码base64图片"""
        import base64
        from io import BytesIO
        
        # 移除data:image前缀（如果有）
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image.convert("RGB")
    
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        scheduler: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        文生图
        
        Args:
            prompt: 提示词
            negative_prompt: 负面提示词
            num_images: 生成图片张数
            scheduler: 采样方法
            width: 图片宽度
            height: 图片高度
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            **kwargs: 其他参数
        
        Returns:
            PIL Image对象或列表
        """
        # 设置scheduler
        if scheduler:
            new_scheduler = self._get_scheduler(scheduler)
            if new_scheduler:
                self.text2img_pipeline.scheduler = new_scheduler
        
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 准备参数
        generate_kwargs = {
            "prompt": prompt,
            "num_images_per_prompt": num_images,
        }
        
        if negative_prompt:
            generate_kwargs["negative_prompt"] = negative_prompt
        if width:
            generate_kwargs["width"] = width
        if height:
            generate_kwargs["height"] = height
        if num_inference_steps:
            generate_kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale:
            generate_kwargs["guidance_scale"] = guidance_scale
        if generator:
            generate_kwargs["generator"] = generator
        
        # 生成图片
        result = self.text2img_pipeline(**generate_kwargs)
        
        images = result.images
        
        if num_images == 1:
            return images[0]
        return images
    
    def image_to_image(
        self,
        prompt: str,
        init_image: Union[str, Image.Image],
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        scheduler: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        图生图
        
        Args:
            prompt: 提示词
            init_image: 初始图片（base64字符串或PIL Image）
            negative_prompt: 负面提示词
            num_images: 生成图片张数
            scheduler: 采样方法
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            strength: 图生图强度
            seed: 随机种子
            **kwargs: 其他参数
        
        Returns:
            PIL Image对象或列表
        """
        # 转换初始图片
        if isinstance(init_image, str):
            init_image = self._decode_base64_image(init_image)
        
        # 设置scheduler
        if scheduler:
            new_scheduler = self._get_scheduler(scheduler)
            if new_scheduler:
                self.img2img_pipeline.scheduler = new_scheduler
        
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 准备参数
        generate_kwargs = {
            "prompt": prompt,
            "image": init_image,
            "num_images_per_prompt": num_images,
        }
        
        if negative_prompt:
            generate_kwargs["negative_prompt"] = negative_prompt
        if num_inference_steps:
            generate_kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale:
            generate_kwargs["guidance_scale"] = guidance_scale
        if strength:
            generate_kwargs["strength"] = strength
        if generator:
            generate_kwargs["generator"] = generator
        
        # 合并额外参数
        generate_kwargs.update(kwargs)
        
        # 生成图片
        result = self.img2img_pipeline(**generate_kwargs)
        
        images = result.images
        
        if num_images == 1:
            return images[0]
        return images

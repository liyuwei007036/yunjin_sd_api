"""
SD模型生成服务（支持LoRA和LoHA）
"""
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import numpy as np
from safetensors import safe_open
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
try:
    from compel import Compel
    COMPEL_AVAILABLE = True
except ImportError:
    COMPEL_AVAILABLE = False
    logger.warning("Compel库未安装，prompt权重语法将无法使用。请运行: pip install compel")
from app.config import Config
from app.utils.logger import logger


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
        # Compel实例（用于处理prompt权重）
        self.compel = None
        self.compel_negative = None
        self._load_models()
    
    def _load_models(self):
        """加载SD模型和LoRA"""
        logger.info(f"加载SD模型: {self.model_path}, 设备: {self.device}")
        
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
        
        # 移动到设备（在加载LoRA之前）
        self.text2img_pipeline = self.text2img_pipeline.to(self.device)
        self.img2img_pipeline = self.img2img_pipeline.to(self.device)
        
        # 加载LoRA模型
        if self.lora_models:
            logger.info(f"加载LoRA模型: {len(self.lora_models)}个")
            for lora_config in self.lora_models:
                lora_path = lora_config.get("path")
                lora_weight = lora_config.get("weight", 1.0)
                if lora_path:
                    try:
                        logger.info(f"加载LoRA: {lora_path}, 权重: {lora_weight}")
                        # 将路径转换为Path对象
                        lora_path_obj = Path(lora_path)
                        
                        # 检查是否是文件路径（通过扩展名判断）
                        if lora_path_obj.suffix.lower() in ['.safetensors', '.bin', '.pt', '.ckpt']:
                            # 文件路径，需要分离目录和文件名
                            lora_dir = str(lora_path_obj.parent.resolve())
                            weight_name = lora_path_obj.name
                            
                            # 先尝试标准LoRA加载
                            try:
                                self.text2img_pipeline.load_lora_weights(lora_dir, weight_name=weight_name, weight=lora_weight)
                                self.img2img_pipeline.load_lora_weights(lora_dir, weight_name=weight_name, weight=lora_weight)
                                logger.info(f"LoRA加载成功: {lora_path}")
                            except (ValueError, Exception) as e:
                                error_msg = str(e)
                                # 检查是否是LoHA格式错误
                                if "have not been correctly renamed" in error_msg or "hada_w" in error_msg:
                                    logger.warning(f"检测到LoHA格式，尝试使用手动加载方式: {lora_path}")
                                    try:
                                        self._load_loha_weights(lora_path, lora_weight)
                                        logger.info(f"LoHA权重手动加载成功: {lora_path}")
                                    except Exception as loha_error:
                                        logger.error(f"LoHA手动加载也失败: {lora_path}, 错误: {str(loha_error)}")
                                        logger.warning(f"跳过此LoRA模型: {lora_path}")
                                else:
                                    raise
                        else:
                            # 目录路径，直接使用
                            try:
                                self.text2img_pipeline.load_lora_weights(lora_path, weight=lora_weight)
                                self.img2img_pipeline.load_lora_weights(lora_path, weight=lora_weight)
                                logger.info(f"LoRA加载成功: {lora_path}")
                            except (ValueError, Exception) as e:
                                error_msg = str(e)
                                # 检查是否是LoHA格式错误
                                if "have not been correctly renamed" in error_msg or "hada_w" in error_msg:
                                    logger.warning(f"检测到LoHA格式，尝试使用手动加载方式: {lora_path}")
                                    # 对于目录，尝试加载目录中的第一个safetensors文件
                                    lora_dir = Path(lora_path)
                                    safetensors_files = list(lora_dir.glob("*.safetensors"))
                                    if safetensors_files:
                                        try:
                                            self._load_loha_weights(str(safetensors_files[0]), lora_weight)
                                            logger.info(f"LoHA权重手动加载成功: {lora_path}")
                                        except Exception as loha_error:
                                            logger.error(f"LoHA手动加载也失败: {lora_path}, 错误: {str(loha_error)}")
                                            logger.warning(f"跳过此LoRA模型: {lora_path}")
                                    else:
                                        logger.error(f"目录中未找到safetensors文件: {lora_path}")
                                        logger.warning(f"跳过此LoRA模型: {lora_path}")
                                else:
                                    raise
                    except Exception as e:
                        error_msg = str(e)
                        # 如果是LoHA相关错误且已经尝试过手动加载，则只记录警告
                        if "have not been correctly renamed" in error_msg or "hada_w" in error_msg:
                            logger.warning(f"LoHA模型加载失败，已跳过: {lora_path}")
                        else:
                            logger.error(f"LoRA加载失败: {lora_path}, 错误: {str(e)}", exc_info=True)
                            raise
        
        # 设置默认scheduler
        if self.default_scheduler and self.default_scheduler in SCHEDULER_MAP:
            scheduler_class = SCHEDULER_MAP[self.default_scheduler]
            scheduler = scheduler_class.from_config(self.text2img_pipeline.scheduler.config)
            self.text2img_pipeline.scheduler = scheduler
            self.img2img_pipeline.scheduler = scheduler
        
        # 初始化Compel（用于处理prompt权重语法，如(cat:1.5)）
        if COMPEL_AVAILABLE:
            try:
                self.compel = Compel(
                    tokenizer=self.text2img_pipeline.tokenizer,
                    text_encoder=self.text2img_pipeline.text_encoder
                )
                self.compel_negative = Compel(
                    tokenizer=self.text2img_pipeline.tokenizer,
                    text_encoder=self.text2img_pipeline.text_encoder
                )
                logger.info("Compel已初始化，支持prompt权重语法（如(cat:1.5)）")
            except Exception as e:
                logger.warning(f"Compel初始化失败: {e}，将使用普通prompt处理")
                self.compel = None
                self.compel_negative = None
        else:
            logger.warning("Compel库未安装，prompt权重语法将无法使用")
        
        logger.info("SD模型加载完成")
    
    def cleanup(self):
        """
        清理资源，释放模型和内存
        应该在应用关闭时调用
        """
        logger.info("开始清理SD服务资源...")
        
        try:
            # 清理Compel实例
            if self.compel is not None:
                del self.compel
                self.compel = None
            if self.compel_negative is not None:
                del self.compel_negative
                self.compel_negative = None
            
            # 清理pipeline
            if self.text2img_pipeline is not None:
                # 将模型移回CPU（如果是GPU）
                if self.device == "cuda":
                    try:
                        self.text2img_pipeline = self.text2img_pipeline.to("cpu")
                    except Exception as e:
                        logger.warning(f"将text2img_pipeline移到CPU时出错: {e}")
                
                # 删除pipeline
                del self.text2img_pipeline
                self.text2img_pipeline = None
            
            if self.img2img_pipeline is not None:
                # 将模型移回CPU（如果是GPU）
                if self.device == "cuda":
                    try:
                        self.img2img_pipeline = self.img2img_pipeline.to("cpu")
                    except Exception as e:
                        logger.warning(f"将img2img_pipeline移到CPU时出错: {e}")
                
                # 删除pipeline
                del self.img2img_pipeline
                self.img2img_pipeline = None
            
            # 清理PyTorch缓存
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logger.info("GPU缓存已清理")
                except Exception as e:
                    logger.warning(f"清理GPU缓存时出错: {e}")
            
            # 清理CPU缓存（强制垃圾回收）
            import gc
            gc.collect()
            logger.info("CPU缓存已清理")
            
            logger.info("SD服务资源清理完成")
        except Exception as e:
            logger.error(f"清理SD服务资源时出错: {e}", exc_info=True)
    
    def _is_loha_format(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """检查是否是LoHA格式"""
        for key in state_dict.keys():
            if "hada_w1_a" in key or "hada_w2_a" in key or "hada_w1_b" in key or "hada_w2_b" in key:
                return True
        return False
    
    def _convert_lora_key_to_unet_key(self, lora_key: str) -> Optional[str]:
        """
        将LoRA键名转换为UNet键名
        
        LoRA键名格式: lora_unet_input_blocks_0_1_transformer_blocks_0_attn1_to_k.hada_w1_a
        UNet键名格式: down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.weight
        """
        if not lora_key.startswith("lora_unet_"):
            return None
        
        # 移除lora_unet_前缀和LoHA后缀
        key = lora_key.replace("lora_unet_", "")
        for suffix in [".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".alpha", ".to_out_0"]:
            key = key.replace(suffix, "")
        
        # 获取UNet的所有键
        unet_keys = list(self.text2img_pipeline.unet.state_dict().keys())
        
        # 构建键名映射规则
        # input_blocks -> down_blocks
        # output_blocks -> up_blocks
        # middle_block -> mid_block
        
        # 转换键名模式
        key_parts = key.split("_")
        
        # 特殊处理: input_blocks -> down_blocks
        if "input_blocks" in key:
            key = key.replace("input_blocks", "down_blocks")
        elif "output_blocks" in key:
            key = key.replace("output_blocks", "up_blocks")
        
        # 将下划线转换为点，并添加.weight后缀
        # 例如: down_blocks_0_attentions_1 -> down_blocks.0.attentions.1.weight
        converted_key = key.replace("_", ".")
        
        # 尝试精确匹配
        for unet_key in unet_keys:
            # 移除.weight或.bias后缀进行比较
            unet_key_base = unet_key.rsplit(".", 1)[0] if "." in unet_key else unet_key
            converted_key_base = converted_key.rsplit(".", 1)[0] if "." in converted_key else converted_key
            
            if unet_key_base == converted_key_base or unet_key_base.endswith(converted_key_base):
                return unet_key
        
        # 如果精确匹配失败，尝试模糊匹配
        best_match = None
        best_score = 0
        
        converted_parts = converted_key.split(".")
        for unet_key in unet_keys:
            if "weight" not in unet_key:
                continue
            
            unet_parts = unet_key.split(".")
            # 计算匹配的连续部分数量
            score = 0
            i = 0
            j = 0
            while i < len(converted_parts) and j < len(unet_parts):
                if converted_parts[i] in unet_parts[j] or unet_parts[j] in converted_parts[i]:
                    score += 2
                    i += 1
                    j += 1
                elif any(cp in up for cp in converted_parts[i:] for up in unet_parts[j:]):
                    # 部分匹配
                    score += 1
                    i += 1
                else:
                    j += 1
            
            if score > best_score:
                best_score = score
                best_match = unet_key
        
        return best_match if best_score >= 3 else None
    
    def _load_loha_weights(self, lora_path: str, weight: float = 1.0):
        """
        手动加载LoHA格式的权重
        
        LoHA (Low-Rank Hadamard Product) 使用 hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b 等键
        基于WebUI的network_hada.py实现逻辑
        """
        logger.info(f"开始手动加载LoHA权重: {lora_path}, 权重: {weight}")
        
        # 加载safetensors文件
        state_dict = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        if not self._is_loha_format(state_dict):
            raise ValueError(f"文件不是LoHA格式: {lora_path}")
        
        # 获取UNet
        unet = self.text2img_pipeline.unet
        unet_state_dict = unet.state_dict()
        
        # 组织LoHA权重数据
        loha_modules = {}
        alpha_values = {}
        
        for key, tensor in state_dict.items():
            if ".alpha" in key:
                base_key = key.replace(".alpha", "")
                alpha_val = tensor.item() if tensor.numel() == 1 else float(tensor.mean().item())
                alpha_values[base_key] = alpha_val
            elif any(suffix in key for suffix in [".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b"]):
                # 提取基础键名
                base_key = key.rsplit(".", 1)[0]
                if base_key not in loha_modules:
                    loha_modules[base_key] = {}
                loha_modules[base_key][key] = tensor
        
        # 应用LoHA权重
        applied_count = 0
        for base_key, weights in loha_modules.items():
            # 获取对应的UNet键
            unet_key = self._convert_lora_key_to_unet_key(base_key)
            if unet_key is None or unet_key not in unet_state_dict:
                continue
            
            # 检查是否有完整的LoHA权重
            w1_a_key = f"{base_key}.hada_w1_a"
            w1_b_key = f"{base_key}.hada_w1_b"
            w2_a_key = f"{base_key}.hada_w2_a"
            w2_b_key = f"{base_key}.hada_w2_b"
            
            if not all(k in weights for k in [w1_a_key, w1_b_key, w2_a_key, w2_b_key]):
                continue
            
            w1_a = weights[w1_a_key].to(self.device)
            w1_b = weights[w1_b_key].to(self.device)
            w2_a = weights[w2_a_key].to(self.device)
            w2_b = weights[w2_b_key].to(self.device)
            
            # 计算LoHA权重: (w1_a @ w1_b) * (w2_a @ w2_b)
            # 对于2D矩阵，使用矩阵乘法；对于1D或元素级，使用逐元素乘法
            if len(w1_a.shape) == 2 and len(w1_b.shape) == 2:
                w1 = torch.matmul(w1_a, w1_b)
            else:
                w1 = w1_a * w1_b
            
            if len(w2_a.shape) == 2 and len(w2_b.shape) == 2:
                w2 = torch.matmul(w2_a, w2_b)
            else:
                w2 = w2_a * w2_b
            
            # Hadamard积
            delta_w = w1 * w2
            
            # 获取alpha值并计算缩放
            alpha = alpha_values.get(base_key, 1.0)
            if alpha <= 0:
                alpha = 1.0
            
            # 计算缩放因子
            if len(delta_w.shape) > 0:
                scale = (alpha / delta_w.shape[0]) * weight if delta_w.shape[0] > 0 else weight
            else:
                scale = alpha * weight
            
            # 应用权重更新
            original_weight = unet_state_dict[unet_key]
            if delta_w.shape != original_weight.shape:
                # 尝试调整形状
                try:
                    delta_w = delta_w.view(original_weight.shape)
                except:
                    logger.warning(f"形状不匹配，跳过: {base_key} -> {unet_key}")
                    continue
            
            with torch.no_grad():
                unet_state_dict[unet_key] = original_weight + delta_w.to(original_weight.dtype) * scale
            
            applied_count += 1
        
        # 加载更新后的权重
        unet.load_state_dict(unet_state_dict, strict=False)
        
        # 同样更新img2img pipeline
        self.img2img_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        
        logger.info(f"LoHA权重加载完成: {lora_path}, 成功应用 {applied_count} 个模块")
    
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
        
        # 检查是否使用Compel处理prompt权重（如果prompt中包含权重语法）
        use_compel = self.compel is not None and ("(" in prompt or "[" in prompt or ":" in prompt)
        
        # 准备参数
        generate_kwargs = {
            "num_images_per_prompt": num_images,
        }
        
        # 使用Compel处理prompt权重（如果可用）
        if use_compel:
            try:
                # 处理正提示词
                prompt_embeds = self.compel(prompt)
                generate_kwargs["prompt_embeds"] = prompt_embeds
                
                # 处理负面提示词（如果提供）
                if negative_prompt:
                    negative_prompt_embeds = self.compel_negative(negative_prompt)
                    generate_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
                # 如果没有负面提示词，不设置negative_prompt_embeds，让pipeline使用默认值
                
                logger.debug(f"使用Compel处理prompt权重: {prompt[:50]}...")
            except Exception as e:
                logger.warning(f"Compel处理prompt失败，回退到普通模式: {e}")
                # 回退到普通模式
                generate_kwargs["prompt"] = prompt
                if negative_prompt:
                    generate_kwargs["negative_prompt"] = negative_prompt
        else:
            # 普通模式，直接使用prompt字符串
            generate_kwargs["prompt"] = prompt
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
        
        # 确保图片是RGB格式
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")
        
        # 记录原图尺寸
        original_size = init_image.size
        logger.info(f"图生图 - 原图尺寸: {original_size}, 模式: {init_image.mode}")
        
        # 设置scheduler
        if scheduler:
            new_scheduler = self._get_scheduler(scheduler)
            if new_scheduler:
                self.img2img_pipeline.scheduler = new_scheduler
        
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 检查是否使用Compel处理prompt权重（如果prompt中包含权重语法）
        use_compel = self.compel is not None and ("(" in prompt or "[" in prompt or ":" in prompt)
        
        # 准备参数
        generate_kwargs = {
            "image": init_image,
            "num_images_per_prompt": num_images,
        }
        
        # 使用Compel处理prompt权重（如果可用）
        if use_compel:
            try:
                # 处理正提示词
                prompt_embeds = self.compel(prompt)
                generate_kwargs["prompt_embeds"] = prompt_embeds
                
                # 处理负面提示词（如果提供）
                if negative_prompt:
                    negative_prompt_embeds = self.compel_negative(negative_prompt)
                    generate_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
                # 如果没有负面提示词，不设置negative_prompt_embeds，让pipeline使用默认值
                
                logger.debug(f"使用Compel处理prompt权重: {prompt[:50]}...")
            except Exception as e:
                logger.warning(f"Compel处理prompt失败，回退到普通模式: {e}")
                # 回退到普通模式
                generate_kwargs["prompt"] = prompt
                if negative_prompt:
                    generate_kwargs["negative_prompt"] = negative_prompt
        else:
            # 普通模式，直接使用prompt字符串
            generate_kwargs["prompt"] = prompt
            if negative_prompt:
                generate_kwargs["negative_prompt"] = negative_prompt
        
        if num_inference_steps:
            generate_kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale:
            generate_kwargs["guidance_scale"] = guidance_scale
        # 修复：strength 参数处理 - 如果为 None，使用默认值 0.75，但允许显式设置为 0.0
        if strength is not None:
            # 确保 strength 在有效范围内
            strength = max(0.0, min(1.0, float(strength)))
            generate_kwargs["strength"] = strength
            logger.info(f"图生图 - 使用 strength: {strength}")
        else:
            # 如果没有设置 strength，使用默认值 0.75（这是 diffusers 的默认值）
            generate_kwargs["strength"] = 0.75
            logger.info(f"图生图 - 使用默认 strength: 0.75（未指定时）")
        if generator:
            generate_kwargs["generator"] = generator
        
        # 合并额外参数
        generate_kwargs.update(kwargs)
        
        # 记录生成参数（用于调试）
        logger.info(f"图生图参数 - prompt长度: {len(prompt)}, strength: {generate_kwargs.get('strength')}, "
                   f"guidance_scale: {generate_kwargs.get('guidance_scale')}, "
                   f"num_inference_steps: {generate_kwargs.get('num_inference_steps')}, "
                   f"原图尺寸: {original_size}")
        
        # 生成图片
        result = self.img2img_pipeline(**generate_kwargs)
        
        # 记录生成结果尺寸
        if result.images:
            result_size = result.images[0].size if isinstance(result.images, list) else result.images.size
            logger.info(f"图生图完成 - 生成图片尺寸: {result_size}, 数量: {len(result.images) if isinstance(result.images, list) else 1}")
        
        images = result.images
        
        if num_images == 1:
            return images[0]
        return images

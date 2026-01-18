"""
LLM服务模块：将自然语言转换为Stable Diffusion提示词
"""
import httpx
import json
from typing import Tuple, Optional
from app.config import Config
from app.utils.logger import logger


class LLMService:
    """LLM服务类，用于将自然语言转换为提示词"""
    
    def __init__(self):
        """初始化LLM服务"""
        self.api_base = Config.LLM_API_BASE
        self.api_key = Config.LLM_API_KEY
        self.model = Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.timeout = Config.LLM_TIMEOUT
        self.provider = Config.LLM_PROVIDER
        
        # 图生图VL模型配置（可选）
        self.vision_model = Config.LLM_VISION_MODEL
        self.vision_api_base = Config.LLM_VISION_API_BASE
        self.vision_api_key = Config.LLM_VISION_API_KEY
    
    def _get_vision_config(self):
        """获取VL模型配置，如果未配置则使用普通模型配置"""
        return {
            "api_base": self.vision_api_base or self.api_base,
            "api_key": self.vision_api_key or self.api_key,
            "model": self.vision_model or self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "timeout": self.timeout * 2  # VL模型需要更长时间
        }
    
    def convert_img2img_prompts_with_image(
        self, 
        natural_language: str, 
        init_image_base64: str
    ) -> Tuple[str, str]:
        """
        基于原图内容识别，将自然语言转换为贴近原图的Stable Diffusion图生图提示词
        
        Args:
            natural_language: 用户的修改要求（如"将衣服变成白色，去掉反光效果"）
            init_image_base64: 原图的base64编码
            
        Returns:
            (prompt, negative_prompt) 元组
        """
        if not self.api_key or not self.api_base:
            raise ValueError("LLM服务未配置，请在config.yaml中配置llm.api_key和llm.api_base")
        
        # 获取VL模型配置
        vision_config = self._get_vision_config()
        vision_model = vision_config["model"]
        
        # 检查模型是否支持视觉输入（GPT-4V, Claude Vision, GLM, Qwen-VL等）
        vision_models = [
            # OpenAI
            "gpt-4-vision-preview", "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
            # Anthropic Claude
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet",
            # GLM (智谱AI)
            "glm-4v", "glm-4.6v", "glm-4-vision", "glm-4-flash-vision",
            # Qwen (通义千问)
            "qwen-vl", "qwen-vl-plus", "qwen-vl-max", "qwen2-vl",
            # Gemini (Google)
            "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash",
            # 其他常见视觉模型
            "llava", "minigpt", "blip", "instructblip"
        ]
        # 检查模型名称是否包含任何视觉模型关键词
        vision_model_lower = vision_model.lower()
        is_vision_model = any(vm in vision_model_lower for vm in vision_models)
        
        # 额外检查：如果模型名称包含 "vision" 或 "vl" 或 "visual"，也认为是视觉模型
        if not is_vision_model:
            vision_keywords = ["vision", "vl", "visual", "image", "multimodal"]
            is_vision_model = any(keyword in vision_model_lower for keyword in vision_keywords)
        
        if not is_vision_model:
            # 如果不支持视觉，回退到普通模式
            logger.warning(f"VL模型 {vision_model} 不支持视觉输入，回退到普通图生图模式")
            return self.convert_to_prompts(natural_language, is_img2img=True)
        
        # 使用VL模型配置
        logger.info(f"使用VL模型进行原图识别: {vision_model}")
        
        try:
            # 从配置文件加载提示词
            system_prompt = Config.get_prompt("img2img_with_vision", "system_prompt")
            if not system_prompt:
                raise ValueError(
                    f"提示词配置缺失：prompts.yaml 中未找到 img2img_with_vision.system_prompt。"
                    f"请确保配置文件 {Config.PROMPTS_FILE} 存在且包含完整的提示词配置。"
                )
            
            user_message_template = Config.get_prompt("img2img_with_vision", "user_message_template")
            if not user_message_template:
                raise ValueError(
                    f"提示词配置缺失：prompts.yaml 中未找到 img2img_with_vision.user_message_template。"
                    f"请确保配置文件 {Config.PROMPTS_FILE} 存在且包含完整的提示词配置。"
                )

            # 处理base64图片（移除data:image前缀）
            image_data = init_image_base64
            if "," in init_image_base64:
                image_data = init_image_base64.split(",")[1]
            
            # 构建消息（使用模板格式化）
            user_message_text = user_message_template.format(natural_language=natural_language)
            
            # 构建消息
            user_message = [
                {
                    "type": "text",
                    "text": user_message_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
            
            # 调用LLM API（支持视觉的模型）
            response = self._call_llm_api_with_vision(system_prompt, user_message, vision_config)
            
            # 解析响应
            prompt, negative_prompt = self._parse_response(response)
            
            # 添加配置的前缀
            if Config.LLM_PROMPT_PREFIX:
                prefix = Config.LLM_PROMPT_PREFIX.strip()
                if prefix:
                    prompt = f"{prefix}, {prompt}"
            
            logger.info(f"基于原图识别的LLM转换成功: 用户要求='{natural_language[:50]}...' -> prompt长度={len(prompt)}")
            return prompt, negative_prompt
            
        except Exception as e:
            logger.error(f"基于原图识别的LLM转换失败: {str(e)}", exc_info=True)
            # 如果失败，回退到普通模式
            logger.warning("回退到普通图生图模式")
            return self.convert_to_prompts(natural_language, is_img2img=True)
    
    def convert_to_prompts(self, natural_language: str, is_img2img: bool = False) -> Tuple[str, str]:
        """
        将自然语言转换为Stable Diffusion提示词
        
        Args:
            natural_language: 自然语言描述
            is_img2img: 是否为图生图任务，默认为False（文生图）
            
        Returns:
            (prompt, negative_prompt) 元组
        """
        if not self.api_key or not self.api_base:
            raise ValueError("LLM服务未配置，请在config.yaml中配置llm.api_key和llm.api_base")
        
        try:
            # 根据任务类型构建不同的system_prompt
            if is_img2img:
                # 图生图任务的system_prompt（从配置文件加载）
                system_prompt = Config.get_prompt("img2img", "system_prompt")
                if not system_prompt:
                    raise ValueError(
                        f"提示词配置缺失：prompts.yaml 中未找到 img2img.system_prompt。"
                        f"请确保配置文件 {Config.PROMPTS_FILE} 存在且包含完整的提示词配置。"
                    )
                
                user_message_template = Config.get_prompt("img2img", "user_message_template")
                if not user_message_template:
                    raise ValueError(
                        f"提示词配置缺失：prompts.yaml 中未找到 img2img.user_message_template。"
                        f"请确保配置文件 {Config.PROMPTS_FILE} 存在且包含完整的提示词配置。"
                    )
                
                user_message = user_message_template.format(natural_language=natural_language)
            else:
                # 文生图任务的system_prompt（从配置文件加载）
                system_prompt = Config.get_prompt("text2img", "system_prompt")
                if not system_prompt:
                    raise ValueError(
                        f"提示词配置缺失：prompts.yaml 中未找到 text2img.system_prompt。"
                        f"请确保配置文件 {Config.PROMPTS_FILE} 存在且包含完整的提示词配置。"
                    )
                
                user_message_template = Config.get_prompt("text2img", "user_message_template")
                if not user_message_template:
                    raise ValueError(
                        f"提示词配置缺失：prompts.yaml 中未找到 text2img.user_message_template。"
                        f"请确保配置文件 {Config.PROMPTS_FILE} 存在且包含完整的提示词配置。"
                    )
                
                user_message = user_message_template.format(natural_language=natural_language)
            
            # 调用LLM API
            response = self._call_llm_api(system_prompt, user_message)
            
            # 解析响应
            prompt, negative_prompt = self._parse_response(response)
            
            # 添加配置的前缀
            if Config.LLM_PROMPT_PREFIX:
                prefix = Config.LLM_PROMPT_PREFIX.strip()
                if prefix:
                    prompt = f"{prefix}, {prompt}"
            
            logger.info(f"LLM转换成功: 自然语言='{natural_language[:50]}...' -> prompt长度={len(prompt)}")
            return prompt, negative_prompt
            
        except Exception as e:
            logger.error(f"LLM转换失败: {str(e)}", exc_info=True)
            raise
    
    def _call_llm_api_with_vision(self, system_prompt: str, user_message: list, vision_config: dict = None) -> str:
        """
        调用支持视觉的LLM API（用于图片内容识别）
        
        Args:
            system_prompt: 系统提示词
            user_message: 用户消息（包含文本和图片）
            vision_config: VL模型配置（如果为None，使用默认配置）
            
        Returns:
            LLM响应的文本内容
        """
        # 使用VL模型配置
        if vision_config is None:
            vision_config = self._get_vision_config()
        
        api_base = vision_config["api_base"]
        api_key = vision_config["api_key"]
        model = vision_config["model"]
        temperature = vision_config["temperature"]
        timeout = vision_config["timeout"]
        provider = vision_config["provider"]
        
        # 构建请求URL
        if provider == "openai" or api_base.endswith("/v1"):
            url = f"{api_base}/chat/completions"
        else:
            url = f"{api_base}/chat/completions"
        
        # 构建请求体
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}  # 强制JSON格式
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 发送请求
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # 提取内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                raise ValueError(f"LLM API响应格式异常: {result}")
    
    def _call_llm_api(self, system_prompt: str, user_message: str) -> str:
        """
        调用LLM API
        
        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            
        Returns:
            LLM响应的文本内容
        """
        # 构建请求URL
        if self.provider == "openai" or self.api_base.endswith("/v1"):
            url = f"{self.api_base}/chat/completions"
        else:
            # 兼容其他OpenAI兼容的API
            url = f"{self.api_base}/chat/completions"
        
        # 构建请求体
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}  # 强制JSON格式
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 发送请求
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # 提取内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                raise ValueError(f"LLM API响应格式异常: {result}")
    
    def _parse_response(self, response_text: str) -> Tuple[str, str]:
        """
        解析LLM响应，提取prompt和negative_prompt
        
        Args:
            response_text: LLM响应的文本内容
            
        Returns:
            (prompt, negative_prompt) 元组
        """
        try:
            # 尝试解析JSON
            # 如果响应包含markdown代码块，先提取JSON部分
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            data = json.loads(response_text)
            
            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")
            
            if not prompt:
                raise ValueError("LLM返回的prompt为空")
            
            # 如果没有negative_prompt，使用默认值（更全面的负面提示词）
            if not negative_prompt:
                negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, ugly, disfigured, bad proportions, malformed, mutated, extra limbs, missing limbs, extra arms, extra legs, unnatural, unrealistic, distorted, out of focus, grainy, noise, oversaturated, undersaturated, compression artifacts"
            
            return prompt, negative_prompt
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM响应不是有效的JSON，尝试提取文本: {e}")
            # 如果JSON解析失败，尝试简单提取
            # 假设格式是 "prompt: ..." 或 "negative_prompt: ..."
            prompt = response_text
            negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
            return prompt, negative_prompt
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
            raise ValueError(f"无法解析LLM响应: {str(e)}")


# 全局LLM服务实例
_llm_service: Optional[LLMService] = None


def get_llm_service() -> Optional[LLMService]:
    """获取LLM服务实例（单例）"""
    global _llm_service
    
    # 检查是否配置了LLM
    if not Config.LLM_API_KEY or not Config.LLM_API_BASE:
        return None
    
    if _llm_service is None:
        try:
            _llm_service = LLMService()
            logger.info("LLM服务初始化成功")
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {e}")
            return None
    
    return _llm_service

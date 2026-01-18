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
    
    def convert_to_prompts(self, natural_language: str) -> Tuple[str, str]:
        """
        将自然语言转换为Stable Diffusion提示词
        
        Args:
            natural_language: 自然语言描述
            
        Returns:
            (prompt, negative_prompt) 元组
        """
        if not self.api_key or not self.api_base:
            raise ValueError("LLM服务未配置，请在config.yaml中配置llm.api_key和llm.api_base")
        
        try:
            # 构建转换提示词 - 基于Stable Diffusion最佳实践
            system_prompt = """你是一个专业的Stable Diffusion提示词生成专家。你的任务是将用户的自然语言描述转换为高质量、结构化的Stable Diffusion提示词。

## Stable Diffusion提示词规则

### 1. 提示词结构顺序（按重要性排序）
必须按照以下顺序组织提示词，用逗号分隔：
1. **主体/焦点元素** (Subject/Focal Element)
   - 人物、角色、物体、生物等主要元素
   - 包含：年龄、性别、姿态、表情、服装等细节
   
2. **环境/背景** (Environment/Setting)
   - 地点、场景、时间、天气、氛围
   - 例如：mountain pass, urban street, forest, beach, indoor room等
   
3. **风格/媒介** (Style/Medium)
   - 艺术风格：photorealistic, digital painting, anime style, oil painting, watercolor等
   - 摄影风格：film photography, cinematic, portrait photography等
   
4. **光照和颜色** (Lighting & Color)
   - 对于写实风格/摄影风格：golden hour, rim lighting, soft lighting, dramatic lighting, cinematic lighting
   - 对于中国传统画/水墨画/工笔画等艺术风格：natural ink wash, traditional Chinese painting lighting, soft brush strokes, ink and wash effect（避免使用摄影类光照术语）
   - 颜色：vibrant colors, muted tones, warm colors, cool palette, ink colors, traditional Chinese colors等
   
5. **构图和视角** (Composition & Perspective)
   - 视角：close-up, wide shot, bird's eye view, low angle, eye level
   - 构图：rule of thirds, centered composition, dynamic angle等
   
6. **质量增强标签** (Quality Enhancers)
   - 分辨率：8k, 4k, ultra high resolution, high quality
   - 细节：ultra-detailed, highly detailed, sharp focus, intricate details
   - 技术：masterpiece, best quality, professional, award winning
   
7. **情绪/氛围** (Mood/Atmosphere)
   - 情绪：melancholic, joyful, mysterious, serene, dramatic
   - 氛围：foggy, ethereal, dreamy, intense等

### 2. 格式要求
- **使用英文**，所有关键词都用英文
- **逗号分隔**：用逗号和空格分隔不同元素，例如："a cat, sitting on windowsill, golden hour, soft lighting, 8k, highly detailed"
- **简洁明了**：每个元素用1-3个词描述，避免冗长句子
- **关键词优先**：使用名词、形容词，避免动词和完整句子
- **长度控制**：正向提示词控制在50-150个英文单词，不要过长

### 3. 权重控制（可选）
- 使用括号增强权重：`(word:1.2)` 表示1.2倍权重
- 使用方括号减弱权重：`[word]` 表示减弱
- 嵌套括号：`((word:1.3))` 表示更强权重

### 4. 负面提示词规则
负面提示词应包含以下类别的问题：
- **质量问题**：lowres, worst quality, low quality, normal quality, jpeg artifacts, blurry, pixelated
- **解剖问题**：bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, ugly
- **肢体问题**：bad hands, extra fingers, missing fingers, extra limbs, missing limbs, extra arms, extra legs
- **文本和水印**：text, watermark, signature, username, copyright, trademark
- **不自然元素**：unnatural, unrealistic, distorted, out of focus, grainy, noise
- **其他常见问题**：cropped, jpeg artifacts, compression artifacts, oversaturated, undersaturated

### 5. 生成要求
- 正向提示词必须包含：主体、环境、风格、质量标签
- **根据风格类型调整光照描述**：
  - 如果是中国传统画、水墨画、工笔画、国画等风格，使用传统绘画的光照描述，避免使用摄影类光照术语
  - 如果是写实、摄影风格，使用摄影类光照术语
  - 如果是其他艺术风格，根据风格特点选择合适的光照描述
- 根据用户描述自动补充合适的风格、光照、质量标签
- 如果用户描述是人物，自动添加合适的质量标签如"portrait, detailed face, beautiful"
- 如果用户描述是风景，自动添加"landscape, scenic, detailed environment"
- 负面提示词必须包含至少10个常见问题关键词

### 6. 输出格式
必须以JSON格式返回，格式如下：
{
  "prompt": "完整的正向提示词，按照规则顺序组织",
  "negative_prompt": "完整的负面提示词，包含常见问题"
}

## 示例

用户输入："一只可爱的小猫坐在窗台上"
输出：
{
  "prompt": "a cute cat, sitting on windowsill, indoor setting, natural lighting, soft daylight, photorealistic, 8k, ultra-detailed, sharp focus, beautiful, high quality, masterpiece",
  "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, ugly, disfigured"
}

现在请根据以上规则，将用户的自然语言描述转换为专业的Stable Diffusion提示词。"""

            user_message = f"""请将以下自然语言描述转换为专业的Stable Diffusion提示词。

用户描述：
{natural_language}

请严格按照上述规则生成提示词，确保：
1. 正向提示词按照规定的7个部分顺序组织
2. 自动补充合适的风格、光照、质量标签
3. 使用逗号分隔，简洁明了
4. 负面提示词包含至少10个常见问题关键词
5. 返回有效的JSON格式"""
            
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

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
        
        # 检查模型是否支持视觉输入（GPT-4V, Claude Vision等）
        vision_models = ["gpt-4-vision-preview", "gpt-4o", "gpt-4-turbo", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        is_vision_model = any(vm in vision_model.lower() for vm in vision_models)
        
        if not is_vision_model:
            # 如果不支持视觉，回退到普通模式
            logger.warning(f"VL模型 {vision_model} 不支持视觉输入，回退到普通图生图模式")
            return self.convert_to_prompts(natural_language, is_img2img=True)
        
        # 使用VL模型配置
        logger.info(f"使用VL模型进行原图识别: {vision_model}")
        
        try:
            system_prompt = """你是一个专业的Stable Diffusion图生图提示词生成专家。你的任务是分析用户提供的原图，识别原图的内容和特征，然后根据用户的修改要求，生成一个"保留原图主要特征+应用用户修改"的提示词。

## 任务流程

1. **识别原图内容**：仔细分析原图，识别以下内容：
   - 主体元素：人物、物体、动物等主要元素
   - 人物特征：年龄、性别、姿态、表情、服装、发型等
   - 环境背景：场景、地点、时间、天气等
   - 视觉特征：颜色、光照、风格、构图、视角等
   - 细节特征：材质、纹理、反光、阴影等

2. **理解用户要求**：分析用户想要进行的修改
   - 颜色修改（如：衣服变白、背景变暗）
   - 效果修改（如：去掉反光、增加阴影）
   - 风格修改（如：转换为动漫风格）
   - 局部修改（如：只修改某个部分）

3. **生成提示词**：生成一个既保留原图主要特征，又应用用户修改的提示词

## Stable Diffusion图生图提示词规则

### 1. 提示词结构顺序（按重要性排序）
必须按照以下顺序组织提示词，用逗号分隔：

1. **保留的主体/焦点元素** (Preserved Subject/Focal Element)
   - 保留原图的主要元素（人物、物体等）
   - 如果用户只修改局部，完整保留其他部分
   - 例如：same person, same pose, same character, keep original subject

2. **应用的修改** (Applied Modifications)
   - 用户要求的修改内容
   - 例如：white clothing, no reflections, matte surface, changed color scheme
   - 这是最重要的部分，必须明确描述用户要求的修改

3. **保留的环境/背景** (Preserved Environment/Setting)
   - 如果用户没有要求修改背景，保留原背景
   - 例如：keep original background, same setting, preserve scene
   - 如果用户要求修改背景，描述新的背景

4. **保留的构图和视角** (Preserved Composition & Perspective)
   - 保持原图的构图和视角
   - 例如：same composition, keep original angle, preserve perspective

5. **目标风格/媒介** (Target Style/Medium)
   - 如果用户要求风格转换，描述目标风格
   - 如果不需要风格转换，保持原图风格
   - 例如：photorealistic, anime style, digital painting

6. **光照和颜色调整** (Lighting & Color Adjustments)
   - 根据用户要求调整光照和颜色
   - 例如：no reflections, matte finish, soft lighting, adjusted lighting

7. **质量增强标签** (Quality Enhancers)
   - 8k, 4k, ultra high resolution, high quality
   - ultra-detailed, highly detailed, sharp focus, intricate details
   - masterpiece, best quality, professional

### 2. 格式要求
- **使用英文**，所有关键词都用英文
- **逗号分隔**：用逗号和空格分隔不同元素
- **简洁明了**：每个元素用1-3个词描述
- **关键词优先**：使用名词、形容词，避免动词和完整句子
- **长度控制**：正向提示词控制在80-200个英文单词

### 3. 生成原则
- **保留原则**：如果用户没有明确要求修改某个部分，在提示词中明确保留该部分
- **修改原则**：用户明确要求的修改，必须在提示词中明确描述
- **平衡原则**：在保留原图特征和应用修改之间找到平衡
- **明确原则**：使用"keep", "preserve", "maintain", "same"等词来明确保留的内容
- **修改原则**：使用"changed", "modified", "no", "without", "different"等词来明确修改的内容

### 4. 负面提示词规则
负面提示词应包含：
- **质量问题**：lowres, worst quality, low quality, normal quality, jpeg artifacts, blurry, pixelated
- **解剖问题**：bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, ugly
- **肢体问题**：bad hands, extra fingers, missing fingers, extra limbs, missing limbs
- **文本和水印**：text, watermark, signature, username, copyright, trademark
- **不自然元素**：unnatural, unrealistic, distorted, out of focus, grainy, noise
- **图生图特定问题**：completely different image, unrelated image, lost original composition, unrecognizable from original, changed subject identity, different person
- **其他常见问题**：cropped, compression artifacts, oversaturated, undersaturated

### 5. 输出格式
必须以JSON格式返回，格式如下：
{
  "prompt": "完整的正向提示词，明确保留原图特征并应用用户修改",
  "negative_prompt": "完整的负面提示词，包含常见问题和图生图特定问题"
}

## 示例

用户上传一张人物照片（穿红色衣服，有反光），要求："将衣服变成白色，去掉反光效果"

分析原图：
- 主体：一个人物，特定姿态和表情
- 服装：红色衣服
- 效果：有反光
- 背景：特定场景
- 风格：写实照片

输出：
{
  "prompt": "same person, same pose, same expression, white clothing, no reflections, matte fabric, keep original background, preserve original scene, same composition, keep original angle, photorealistic, adjusted lighting, no reflective surfaces, 8k, ultra-detailed, highly detailed, sharp focus, masterpiece, best quality",
  "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, ugly, disfigured, reflections, shiny surfaces, completely different image, unrelated image, lost original composition, unrecognizable from original, changed subject identity, different person"
}

现在请分析用户提供的原图，识别原图内容，然后根据用户的修改要求生成提示词。"""

            # 处理base64图片（移除data:image前缀）
            image_data = init_image_base64
            if "," in init_image_base64:
                image_data = init_image_base64.split(",")[1]
            
            # 构建消息
            user_message = [
                {
                    "type": "text",
                    "text": f"""请分析这张原图，识别原图的内容和特征，然后根据用户的修改要求生成一个"保留原图主要特征+应用用户修改"的Stable Diffusion图生图提示词。

**用户的修改要求**：
{natural_language}

请按照以下步骤：
1. 仔细分析原图，识别原图的主要内容、特征、风格等
2. 理解用户想要进行的修改
3. 生成一个既保留原图主要特征，又应用用户修改的提示词
4. 确保提示词明确描述要保留的内容和要修改的内容

请返回JSON格式的提示词。"""
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
                # 图生图任务的system_prompt
                system_prompt = """你是一个专业的Stable Diffusion提示词生成专家。你的任务是将用户的自然语言描述转换为高质量、结构化的Stable Diffusion图生图提示词。

## 图生图（Image-to-Image）任务说明

**重要**：这是图生图任务，用户已经提供了一张初始图像。你的提示词应该描述"基于现有图像要做什么改变"，而不是从零开始描述整个场景。

图生图提示词应该：
1. **关注变化和修改**：描述想要对原图进行的改变、修改、转换
2. **保留原图特征**：如果需要保留原图的主体、构图等，在提示词中体现
3. **描述目标效果**：明确说明最终希望达到的效果，但不需要重复描述原图中已经存在的明显特征

## Stable Diffusion图生图提示词规则

### 1. 提示词结构顺序（按重要性排序）
必须按照以下顺序组织提示词，用逗号分隔：
1. **变化的主体/焦点元素** (Modified Subject/Focal Element)
   - 要修改或改变的主要元素
   - 例如：changed appearance, different style, new color scheme, transformed character
   - 如果描述的是风格转换，强调目标风格
   
2. **变化的环境/背景** (Modified Environment/Setting)
   - 如果需要改变背景、场景、地点，在这里描述
   - 例如：different background, new setting, transformed scene
   - 如果不需要改变背景，可以省略或描述为"keep original background"
   
3. **目标风格/媒介** (Target Style/Medium)
   - 转换后的艺术风格：photorealistic, digital painting, anime style, oil painting, watercolor等
   - 摄影风格：film photography, cinematic, portrait photography等
   - 如果是风格转换任务，这是最重要的部分
   
4. **光照和颜色变化** (Lighting & Color Changes)
   - 光照变化：changed lighting, different time of day, new lighting conditions
   - 颜色调整：color shift, color grading, new color palette
   - 根据风格类型调整光照描述（同文生图规则）
   
5. **构图和视角** (Composition & Perspective)
   - 如果不需要改变构图：keep original composition
   - 如果需要改变：new angle, different perspective等
   
6. **质量增强标签** (Quality Enhancers)
   - 分辨率：8k, 4k, ultra high resolution, high quality
   - 细节：ultra-detailed, highly detailed, sharp focus, intricate details
   - 技术：masterpiece, best quality, professional, award winning
   
7. **情绪/氛围变化** (Mood/Atmosphere Changes)
   - 如果改变情绪：different mood, new atmosphere
   - 如果需要保持：maintain original mood

### 2. 格式要求
- **使用英文**，所有关键词都用英文
- **逗号分隔**：用逗号和空格分隔不同元素
- **简洁明了**：每个元素用1-3个词描述，避免冗长句子
- **关键词优先**：使用名词、形容词，避免动词和完整句子
- **长度控制**：正向提示词控制在50-150个英文单词，不要过长
- **强调变化**：使用"changed", "transformed", "modified", "different", "new"等词来描述变化

### 3. 图生图特殊提示词
- 如果用户描述的是风格转换，强调目标风格
- 如果用户描述的是局部修改，描述具体的修改内容
- 如果用户描述的是整体变换，描述变换方向和效果
- 不要重复描述原图中已经明确存在的特征（除非需要特别强调保留）

### 4. 权重控制（可选）
- 使用括号增强权重：`(word:1.2)` 表示1.2倍权重
- 使用方括号减弱权重：`[word]` 表示减弱
- 嵌套括号：`((word:1.3))` 表示更强权重

### 5. 负面提示词规则
负面提示词应包含以下类别的问题：
- **质量问题**：lowres, worst quality, low quality, normal quality, jpeg artifacts, blurry, pixelated
- **解剖问题**：bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, ugly
- **肢体问题**：bad hands, extra fingers, missing fingers, extra limbs, missing limbs, extra arms, extra legs
- **文本和水印**：text, watermark, signature, username, copyright, trademark
- **不自然元素**：unnatural, unrealistic, distorted, out of focus, grainy, noise
- **图生图特殊问题**：completely different image, unrelated image, lost original composition, unrecognizable from original
- **其他常见问题**：cropped, jpeg artifacts, compression artifacts, oversaturated, undersaturated

### 6. 生成要求
- 正向提示词必须关注"变化"和"修改"，而不是完全重新描述场景
- **根据风格类型调整光照描述**（同文生图规则）
- 自动补充合适的风格、光照、质量标签
- 负面提示词必须包含至少10个常见问题关键词，特别是图生图相关的问题

### 7. 输出格式
必须以JSON格式返回，格式如下：
{
  "prompt": "完整的正向提示词，按照规则顺序组织，强调变化",
  "negative_prompt": "完整的负面提示词，包含常见问题和图生图特定问题"
}

## 图生图示例

用户输入："将这张图片转换成动漫风格"
输出：
{
  "prompt": "anime style, manga style, anime art, transformed to anime, colorful anime illustration, cel shading, vibrant colors, detailed anime character, high quality anime art, 8k, ultra-detailed, sharp focus, masterpiece, best quality",
  "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, ugly, disfigured, photorealistic, realistic, unrelated image, lost original composition"
}

用户输入："改变这张照片的颜色为暖色调"
输出：
{
  "prompt": "warm color palette, warm tones, golden hour lighting, warm color grading, color shifted to warm, keep original composition, maintain original subject, warm color scheme, 8k, ultra-detailed, high quality, professional color grading",
  "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, ugly, disfigured, cool tones, blue color cast, completely different image"
}

现在请根据以上规则，将用户的自然语言描述转换为专业的Stable Diffusion图生图提示词。"""
                
                user_message = f"""请将以下自然语言描述转换为专业的Stable Diffusion图生图提示词。

**任务类型**：图生图（Image-to-Image）- 基于现有图像进行修改

用户描述：
{natural_language}

请严格按照上述规则生成提示词，确保：
1. 提示词关注"变化"和"修改"，而不是完全重新描述场景
2. 正向提示词按照规定的7个部分顺序组织
3. 自动补充合适的风格、光照、质量标签
4. 使用逗号分隔，简洁明了
5. 负面提示词包含至少10个常见问题关键词，特别是图生图特定问题
6. 返回有效的JSON格式"""
            else:
                # 文生图任务的system_prompt（原有逻辑）
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

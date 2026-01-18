# SD模型API接口文档

## 目录

- [服务概述](#服务概述)
- [认证方式](#认证方式)
- [基础URL](#基础url)
- [接口列表](#接口列表)
- [接口详细说明](#接口详细说明)
  - [1. 健康检查](#1-健康检查)
  - [2. 图片生成](#2-图片生成)
  - [3. 查询任务状态](#3-查询任务状态)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [错误码说明](#错误码说明)

---

## 服务概述

本服务是基于 Stable Diffusion 的图片生成API服务，支持以下功能：

- **文生图（Text-to-Image）**：根据文本提示词生成图片
- **图生图（Image-to-Image）**：基于输入图片和提示词生成新图片
- **异步任务处理**：支持异步生成，通过任务ID查询结果
- **回调通知**：支持配置回调URL，生成完成后自动推送结果
- **批量生成**：支持一次生成多张图片（最多10张）
- **LoRA/LoHA支持**：支持加载LoRA模型进行风格定制

---

## 认证方式

所有接口（除健康检查外）都需要在请求头中携带API Key进行认证。

### 请求头

```
X-API-Key: your_api_key
```

### 配置方式

API Key需要在 `config.yaml` 文件中配置：

```yaml
api:
  keys:
    - "your_api_key_1"
    - "your_api_key_2"
  key_header: "X-API-Key"
```

---

## 基础URL

```
http://localhost:8000
```

生产环境请替换为实际的服务器地址。

---

## 接口列表

| 接口 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/health` 或 `/api/health` | 检查服务状态 |
| 图片生成 | POST | `/api/v1/generate` | 生成图片（文生图/图生图） |
| 查询任务状态 | GET | `/api/v1/tasks/{task_id}` | 查询生成任务状态 |

---

## 接口详细说明

### 1. 健康检查

检查服务状态和模型加载情况。

#### 请求

```http
GET /health
GET /api/health
```

#### 响应

```json
{
  "status": "ok",
  "model_loaded": true
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态，正常为 "ok" |
| model_loaded | boolean | 模型是否已加载完成 |

#### 示例

```bash
curl -X GET http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 2. 图片生成

生成图片接口，支持文生图和图生图两种模式。

- **文生图**：不提供 `init_image` 参数时，执行文生图
- **图生图**：提供 `init_image` 参数时，执行图生图

#### 请求

```http
POST /api/v1/generate
Content-Type: application/json
X-API-Key: your_api_key
```

#### 请求参数

**请求体（JSON）：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| prompt | string | 是 | - | 提示词，描述要生成的图片内容 |
| init_image | string | 否 | null | 初始图片的base64编码（包含data:image前缀）。提供此参数时执行图生图 |
| negative_prompt | string | 否 | null | 负面提示词，描述不希望在图片中出现的内容 |
| width | integer | 否 | null | 图片宽度（像素）。仅文生图有效 |
| height | integer | 否 | null | 图片高度（像素）。仅文生图有效 |
| num_images | integer | 否 | 1 | 生成图片张数，范围：1-10 |
| num_inference_steps | integer | 否 | null | 推理步数。步数越多质量越好但速度越慢，建议20-50 |
| guidance_scale | float | 否 | null | 引导强度（CFG Scale）。控制模型对提示词的遵循程度，建议7.0-12.0 |
| strength | float | 否 | null | 图生图强度。仅图生图有效，范围0.0-1.0，控制对原图的修改程度 |
| scheduler | string | 否 | null | 采样方法名称，详见[采样方法列表](#采样方法列表) |
| seed | integer | 否 | null | 随机种子。使用相同seed可生成相同图片，用于结果复现 |
| output_format | string | 否 | "png" | 输出图片格式，可选：`png`、`jpg`、`jpeg` |
| callback_url | string | 否 | null | 回调URL。生成完成后将结果通过POST请求推送到该URL |

#### 响应

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "任务已创建，正在处理中"
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID，用于查询任务状态 |
| status | string | 任务状态：`pending`（待处理）、`processing`（处理中）、`completed`（已完成）、`failed`（失败） |
| message | string | 响应消息 |

#### 示例

**文生图示例：**

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "a beautiful landscape with mountains and lakes, sunset, highly detailed, 4k",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "output_format": "png"
  }'
```

**图生图示例：**

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "anime style, vibrant colors",
    "init_image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "strength": 0.7,
    "guidance_scale": 7.5,
    "num_inference_steps": 30
  }'
```

---

### 3. 查询任务状态

查询图片生成任务的状态和结果。

#### 请求

```http
GET /api/v1/tasks/{task_id}
X-API-Key: your_api_key
```

#### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID（由生成接口返回） |

#### 响应

**成功时（单张图片）：**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result_url": "http://192.168.31.40:9000/sd-images/abc123.png",
  "result_urls": null,
  "error_message": null,
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:30"
}
```

**成功时（多张图片）：**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result_url": null,
  "result_urls": [
    "http://192.168.31.40:9000/sd-images/abc123.png",
    "http://192.168.31.40:9000/sd-images/def456.png"
  ],
  "error_message": null,
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:35"
}
```

**失败时：**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "result_url": null,
  "result_urls": null,
  "error_message": "生成过程中发生错误",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:10"
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID |
| status | string | 任务状态：`pending`（待处理）、`processing`（处理中）、`completed`（已完成）、`failed`（失败） |
| result_url | string | 单张图片URL（`num_images=1` 时有效） |
| result_urls | array | 多张图片URL列表（`num_images>1` 时有效） |
| error_message | string | 错误信息（失败时） |
| created_at | string | 任务创建时间（ISO格式） |
| updated_at | string | 任务更新时间（ISO格式） |

#### 示例

```bash
curl -X GET http://localhost:8000/api/v1/tasks/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-API-Key: your_api_key"
```

---

## 参数说明

### guidance_scale（引导强度）

- **含义**：控制模型对提示词的遵循程度（CFG Scale）
- **取值范围**：建议 7.0-12.0
- **说明**：
  - 值越高，生成的图片越符合提示词描述，但可能过度饱和或失真
  - 值越低，生成结果更随机，但可能不够符合提示词
  - 默认值通常为 7.5
- **推荐值**：
  - 一般场景：7.5-9.0
  - 需要高精度：10.0-12.0
  - 需要创意多样性：5.0-7.0

### strength（图生图强度）

- **含义**：控制对初始图片的修改程度
- **取值范围**：0.0-1.0
- **说明**：
  - `0.0`：几乎不修改，输出接近原图
  - `0.1-0.3`：轻微调整，适合修复或微调
  - `0.4-0.6`：中等变化，适合风格转换
  - `0.7-0.9`：较大变化，保留构图但内容大幅改变
  - `1.0`：接近重新生成
- **推荐值**：
  - 微调/修复：0.2-0.4
  - 风格转换：0.5-0.7
  - 大幅改造：0.7-0.9
  - 默认值：0.75

### num_inference_steps（推理步数）

- **含义**：生成图片时的迭代步数
- **说明**：
  - 步数越多，质量越好但生成时间越长
  - 步数过少可能导致质量下降
- **推荐值**：
  - 快速生成：20-30步
  - 平衡：30-40步
  - 高质量：40-50步

### seed（随机种子）

- **含义**：控制生成结果的随机性
- **说明**：
  - 使用相同的 `seed` 和其他参数，可生成相同或相似的图片
  - 不指定时每次生成结果都不同
  - 用于结果复现和对比实验

### 采样方法列表

支持的采样方法（scheduler）：

| 方法名称 | 说明 |
|----------|------|
| `DPMSolverMultistepScheduler` | DPM多步求解器，质量好速度快（推荐） |
| `DDIMScheduler` | DDIM采样，稳定但较慢 |
| `EulerDiscreteScheduler` | Euler离散采样，快速 |
| `EulerAncestralDiscreteScheduler` | Euler祖先采样，快速且随机性强 |
| `PNDMScheduler` | PNDM采样，经典方法 |
| `LMSDiscreteScheduler` | LMS线性多步采样 |
| `HeunDiscreteScheduler` | Heun离散采样 |
| `KDPM2DiscreteScheduler` | KDPM2离散采样 |
| `KDPM2AncestralDiscreteScheduler` | KDPM2祖先采样 |

**推荐**：使用 `DPMSolverMultistepScheduler` 或 `EulerDiscreteScheduler`，在质量和速度间取得良好平衡。

---

## 使用示例

### 示例1：基础文生图

```python
import requests
import time

# API配置
api_url = "http://localhost:8000/api/v1/generate"
api_key = "your_api_key"

# 请求参数
payload = {
    "prompt": "a beautiful sunset over the ocean, peaceful, serene, 4k",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "output_format": "png"
}

# 发送生成请求
headers = {
    "Content-Type": "application/json",
    "X-API-Key": api_key
}

response = requests.post(api_url, json=payload, headers=headers)
result = response.json()

print(f"任务ID: {result['task_id']}")
task_id = result['task_id']

# 轮询查询任务状态
while True:
    status_response = requests.get(
        f"http://localhost:8000/api/v1/tasks/{task_id}",
        headers={"X-API-Key": api_key}
    )
    status_data = status_response.json()
    
    if status_data['status'] == 'completed':
        print(f"生成成功！图片URL: {status_data['result_url']}")
        break
    elif status_data['status'] == 'failed':
        print(f"生成失败: {status_data['error_message']}")
        break
    else:
        print(f"任务状态: {status_data['status']}，等待中...")
        time.sleep(2)
```

### 示例2：图生图（风格转换）

```python
import requests
import base64

# 读取本地图片并转换为base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_str = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"

# 准备参数
payload = {
    "prompt": "anime style, vibrant colors, detailed",
    "init_image": image_to_base64("input_image.png"),
    "strength": 0.7,  # 较大变化，风格转换
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "output_format": "png"
}

# 发送请求（同上）
```

### 示例3：批量生成多张图片

```python
payload = {
    "prompt": "a cute cat playing in the garden",
    "width": 512,
    "height": 512,
    "num_images": 3,  # 生成3张不同的图片
    "guidance_scale": 7.5,
    "num_inference_steps": 30
}

# ... 发送请求并查询状态 ...

# 完成后，result_urls 包含3个URL
print(f"生成完成，图片URL列表: {status_data['result_urls']}")
```

### 示例4：使用回调URL（推荐）

```python
# 配置回调URL，生成完成后自动推送结果到你的服务器
payload = {
    "prompt": "a futuristic cityscape at night",
    "width": 1024,
    "height": 1024,
    "callback_url": "https://your-server.com/api/callback",  # 你的回调接收地址
    "guidance_scale": 7.5,
    "num_inference_steps": 30
}

# 发送请求后，不需要轮询
# 生成完成后，服务器会自动POST结果到callback_url
```

**回调接收示例（你的服务器端）：**

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/api/callback', methods=['POST'])
def callback():
    data = request.json
    task_id = data['task_id']
    status = data['status']
    
    if status == 'completed':
        # 单张图片
        if 'image_url' in data:
            image_url = data['image_url']
            print(f"任务 {task_id} 完成，图片URL: {image_url}")
        # 多张图片
        elif 'image_urls' in data:
            image_urls = data['image_urls']
            print(f"任务 {task_id} 完成，图片URLs: {image_urls}")
    elif status == 'failed':
        error_message = data['error_message']
        print(f"任务 {task_id} 失败: {error_message}")
    
    return {'status': 'ok'}
```

### 示例5：使用seed实现结果复现

```python
# 第一次生成
payload1 = {
    "prompt": "a magical forest with glowing mushrooms",
    "seed": 12345,  # 固定seed
    "guidance_scale": 7.5,
    "num_inference_steps": 30
}

# 第二次生成（使用相同的seed和参数）
payload2 = {
    "prompt": "a magical forest with glowing mushrooms",
    "seed": 12345,  # 相同的seed
    "guidance_scale": 7.5,
    "num_inference_steps": 30
}

# 两次生成的结果应该相同或非常相似
```

### 示例6：JavaScript/Node.js示例

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000/api/v1';
const API_KEY = 'your_api_key';

// 生成图片
async function generateImage() {
    try {
        // 1. 提交生成任务
        const generateResponse = await axios.post(
            `${API_URL}/generate`,
            {
                prompt: 'a beautiful landscape, sunset, 4k',
                width: 512,
                height: 512,
                guidance_scale: 7.5,
                num_inference_steps: 30
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                }
            }
        );
        
        const { task_id } = generateResponse.data;
        console.log(`任务ID: ${task_id}`);
        
        // 2. 轮询查询状态
        let status = 'pending';
        while (status !== 'completed' && status !== 'failed') {
            await new Promise(resolve => setTimeout(resolve, 2000)); // 等待2秒
            
            const statusResponse = await axios.get(
                `${API_URL}/tasks/${task_id}`,
                {
                    headers: {
                        'X-API-Key': API_KEY
                    }
                }
            );
            
            const taskData = statusResponse.data;
            status = taskData.status;
            
            if (status === 'completed') {
                console.log('生成成功！');
                console.log('图片URL:', taskData.result_url || taskData.result_urls);
            } else if (status === 'failed') {
                console.error('生成失败:', taskData.error_message);
            } else {
                console.log(`状态: ${status}，等待中...`);
            }
        }
    } catch (error) {
        console.error('错误:', error.response?.data || error.message);
    }
}

generateImage();
```

### 示例7：使用curl命令

```bash
# 1. 生成图片
TASK_ID=$(curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "a cat sitting on a windowsill",
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5
  }' | jq -r '.task_id')

echo "任务ID: $TASK_ID"

# 2. 查询状态（循环直到完成）
while true; do
  RESPONSE=$(curl -X GET "http://localhost:8000/api/v1/tasks/$TASK_ID" \
    -H "X-API-Key: your_api_key")
  
  STATUS=$(echo $RESPONSE | jq -r '.status')
  
  if [ "$STATUS" == "completed" ]; then
    URL=$(echo $RESPONSE | jq -r '.result_url')
    echo "生成成功！图片URL: $URL"
    break
  elif [ "$STATUS" == "failed" ]; then
    ERROR=$(echo $RESPONSE | jq -r '.error_message')
    echo "生成失败: $ERROR"
    break
  else
    echo "状态: $STATUS，等待中..."
    sleep 2
  fi
done
```

---

## 错误码说明

### HTTP状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 401 | 认证失败（API Key无效或缺失） |
| 404 | 资源不存在（如任务ID不存在） |
| 422 | 参数验证失败 |
| 500 | 服务器内部错误 |

### 错误响应示例

```json
{
  "detail": "任务不存在"
}
```

或

```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## 注意事项

1. **图片格式**：
   - `init_image` 必须为base64编码，且包含 `data:image/png;base64,` 或 `data:image/jpeg;base64,` 前缀
   - 支持的输入格式：PNG、JPEG

2. **生成时间**：
   - 图片生成是耗时操作，通常需要10-60秒（取决于参数和硬件）
   - 建议使用回调URL或异步轮询，避免阻塞

3. **图片尺寸**：
   - 建议使用512x512、768x768、1024x1024等常见尺寸
   - 尺寸越大，生成时间越长，显存占用越大

4. **并发限制**：
   - 由于模型资源限制，建议控制并发请求数量
   - 大量请求时可能排队等待

5. **存储**：
   - 生成的图片会存储到MinIO OSS
   - 图片URL是公开访问的，注意安全配置

---

## 交互式API文档

服务启动后，可以访问以下地址查看交互式API文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

这些文档提供了完整的接口说明和在线测试功能。

---

## 技术支持

如有问题或建议，请查看项目文档或联系技术支持。

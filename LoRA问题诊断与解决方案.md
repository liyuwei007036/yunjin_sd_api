# LoRA 效果不一致问题诊断与解决方案

## 问题描述
使用 LoRA 后生成的图片效果与在 Kohya 中生成的完全不一致。

## 常见原因分析

### 1. **触发词（Trigger Words）缺失** ⚠️ **最常见原因**

**问题**：Kohya 训练 LoRA 时通常会使用特定的触发词（trigger words）来激活 LoRA。如果提示词中没有包含这些触发词，LoRA 效果可能不明显或完全无效。

**解决方案**：
1. **查找触发词**：
   - 检查 LoRA 训练时的配置文件（通常是 `training_config.json` 或训练日志）
   - 查看 LoRA 文件的元数据（如果保存了）
   - 在 Kohya 中查看训练时使用的触发词

2. **在提示词中使用触发词**：
   - 将触发词放在提示词的开头（最重要位置）
   - 可以使用权重增强：`(trigger_word:1.2)` 或 `((trigger_word))`
   - 例如：如果触发词是 `yunjin_style`，提示词应该是：`yunjin_style, [其他描述]`

3. **配置提示词前缀**：
   - 在 `config.yaml` 中配置 `llm.prompt_prefix`，自动在所有生成的提示词前添加触发词
   - 当前配置：`prompt_prefix: "nanjing yunjin style"`
   - 如果触发词不是 "nanjing yunjin style"，请修改为正确的触发词

### 2. **LoRA 权重不合适**

**问题**：当前配置的权重是 `0.8`，但实际使用时可能需要调整。

**解决方案**：
1. **尝试不同的权重值**：
   - 权重范围通常在 `0.5 - 1.5` 之间
   - 如果效果太弱，尝试增加到 `1.0` 或 `1.2`
   - 如果效果过强或出现异常，尝试降低到 `0.6` 或 `0.5`

2. **修改配置**：
   ```yaml
   lora:
     models:
       - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
         weight: 1.0  # 尝试不同的值：0.5, 0.6, 0.8, 1.0, 1.2, 1.5
   ```

3. **重启服务**：修改权重后需要重启服务才能生效

### 3. **采样器/调度器差异**

**问题**：Kohya 和当前代码使用的采样器可能不同，导致生成效果差异。

**解决方案**：
1. **确认 Kohya 中使用的采样器**：
   - 查看 Kohya 生成图片时使用的采样器名称
   - 常见采样器：`Euler a`, `DPM++ 2M Karras`, `DDIM`, `DPMSolverMultistepScheduler` 等

2. **在 API 调用时指定相同的采样器**：
   - 查看 `sd_service.py` 中支持的采样器列表
   - 在 API 请求中指定 `scheduler` 参数
   - 例如：`"scheduler": "DPMSolverMultistepScheduler"`

3. **配置默认采样器**：
   ```yaml
   default_scheduler: "DPMSolverMultistepScheduler"  # 在 config.yaml 中配置
   ```

### 4. **CFG Scale（引导强度）差异**

**问题**：CFG Scale 值不同会导致生成效果差异。

**解决方案**：
1. **确认 Kohya 中使用的 CFG Scale**：
   - 通常在 `7.0 - 12.0` 之间
   - 默认值通常是 `7.0` 或 `7.5`

2. **在 API 调用时指定相同的 CFG Scale**：
   - 在 API 请求中指定 `guidance_scale` 参数
   - 例如：`"guidance_scale": 7.0`

### 5. **推理步数差异**

**问题**：推理步数不同会影响生成质量和效果。

**解决方案**：
1. **确认 Kohya 中使用的步数**：
   - 通常在 `20 - 50` 步之间
   - 默认值通常是 `20` 或 `30`

2. **在 API 调用时指定相同的步数**：
   - 在 API 请求中指定 `num_inference_steps` 参数
   - 例如：`"num_inference_steps": 30`

### 6. **提示词格式差异**

**问题**：提示词的结构、顺序、权重标记等可能不同。

**解决方案**：
1. **使用与 Kohya 相同的提示词格式**：
   - 保持提示词结构一致
   - 使用相同的权重标记（如果有）
   - 保持触发词位置一致

2. **检查提示词前缀**：
   - 当前配置：`prompt_prefix: "nanjing yunjin style"`
   - 确保这个前缀与 Kohya 中使用的触发词一致

### 7. **LoRA 加载方式问题**

**问题**：虽然代码使用了标准的 `load_lora_weights` 方法，但某些特殊格式的 LoRA 可能需要特殊处理。

**解决方案**：
1. **检查 LoRA 文件格式**：
   - 确认是 `.safetensors` 格式（推荐）
   - 如果是 `.ckpt` 或 `.pt` 格式，可能需要转换

2. **查看日志**：
   - 检查服务启动时的日志，确认 LoRA 是否成功加载
   - 查找是否有错误或警告信息

## 诊断步骤

### 步骤 1：确认触发词
1. 检查 LoRA 训练配置，找到触发词
2. 确认当前提示词中是否包含触发词
3. 如果不包含，添加触发词到提示词开头

### 步骤 2：对比参数
1. 记录 Kohya 中使用的所有参数：
   - 采样器
   - CFG Scale
   - 推理步数
   - 提示词（包括触发词）
   - LoRA 权重

2. 在 API 调用时使用相同的参数

### 步骤 3：测试不同权重
1. 从 `0.5` 开始，逐步增加到 `1.5`
2. 每次测试使用相同的提示词和参数
3. 找到效果最好的权重值

### 步骤 4：检查日志
1. 查看服务启动日志，确认 LoRA 加载成功
2. 查看生成时的日志，确认参数正确传递

## 快速检查清单

- [ ] 提示词中包含触发词（最重要！）
- [ ] LoRA 权重设置合理（0.5-1.5）
- [ ] 使用相同的采样器
- [ ] 使用相同的 CFG Scale
- [ ] 使用相同的推理步数
- [ ] 提示词格式一致
- [ ] LoRA 文件加载成功（查看日志）
- [ ] 服务重启后生效

## 示例：正确的配置和使用

### config.yaml 配置
```yaml
lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0  # 根据测试结果调整

llm:
  prompt_prefix: "yunjin_style"  # 使用正确的触发词

default_scheduler: "DPMSolverMultistepScheduler"  # 与 Kohya 保持一致
```

### API 调用示例
```json
{
  "prompt": "yunjin_style, a beautiful landscape, traditional Chinese patterns",
  "negative_prompt": "lowres, bad quality",
  "num_inference_steps": 30,
  "guidance_scale": 7.0,
  "scheduler": "DPMSolverMultistepScheduler",
  "seed": 42
}
```

## 如果问题仍然存在

1. **检查 LoRA 文件**：
   - 确认 LoRA 文件没有损坏
   - 尝试重新训练或下载 LoRA 文件

2. **对比基础模型**：
   - 确认使用相同的基础 SD 模型
   - 不同版本的 SD 模型可能导致效果差异

3. **查看详细日志**：
   - 启用详细日志模式
   - 检查是否有错误或警告

4. **联系支持**：
   - 提供详细的参数对比
   - 提供 LoRA 训练配置信息
   - 提供生成结果的对比图

## 常见触发词示例

根据你的配置，可能的触发词包括：
- `yunjin_style`
- `nanjing yunjin style`
- `yunjin brocade`
- `yunjin`
- 或其他训练时使用的特定词汇

**重要提示**：触发词必须与训练时使用的完全一致，包括大小写和格式。

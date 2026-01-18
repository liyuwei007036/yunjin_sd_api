# LoRA 效果不一致 - 快速修复指南

## 🚨 最常见原因：缺少触发词

**90% 的问题都是因为提示词中没有包含 LoRA 的触发词！**

## 快速修复步骤

### 1. 找到触发词（最重要！）

触发词是训练 LoRA 时使用的特殊关键词。你需要：

1. **查看训练配置**：
   - 打开 Kohya 训练时的配置文件
   - 查找 `trigger_words` 或 `activation_words` 字段
   - 或者在训练日志中查找

2. **常见触发词格式**：
   - 单个词：`yunjin_style`
   - 多个词：`nanjing yunjin style`
   - 带下划线：`yunjin_style_v1`

### 2. 配置触发词

在 `config.yaml` 中为你的 LoRA 添加触发词：

```yaml
lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0
      trigger_words: ["yunjin_style"]  # ⬅️ 添加这一行！使用你的实际触发词
```

**或者使用逗号分隔的字符串**：
```yaml
trigger_words: "yunjin_style, nanjing yunjin"
```

### 3. 确保提示词包含触发词

触发词必须出现在提示词中才能激活 LoRA。有两种方式：

**方式 1：使用 prompt_prefix（自动添加）**
```yaml
llm:
  prompt_prefix: "yunjin_style"  # 自动在所有提示词前添加
```

**方式 2：手动在提示词中包含**
```
提示词：yunjin_style, a beautiful landscape, traditional patterns
```

### 4. 调整权重（如果需要）

如果添加触发词后效果仍然不理想，尝试调整权重：

```yaml
lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0  # 尝试：0.5, 0.6, 0.8, 1.0, 1.2, 1.5
      trigger_words: ["yunjin_style"]
```

### 5. 重启服务

修改配置后，**必须重启服务**才能生效。

## 检查清单

- [ ] ✅ 找到了正确的触发词
- [ ] ✅ 在 `config.yaml` 中配置了 `trigger_words`
- [ ] ✅ 提示词中包含触发词（或使用了 `prompt_prefix`）
- [ ] ✅ 重启了服务
- [ ] ✅ 查看日志确认 LoRA 加载成功

## 查看日志确认

启动服务时，你应该看到类似这样的日志：

```
[LoRA 1/1] 加载LoRA: D:\yunjin_train\SD1.5\output\yunjin_style.safetensors
  - 权重: 1.0
  - 触发词: yunjin_style
  ✓ LoRA加载成功: ...
```

**如果看到警告**：
```
⚠️ 未配置触发词！这可能导致LoRA效果不明显。
```

说明你需要在配置中添加 `trigger_words`。

## 如果还是不行

1. **对比参数**：确保使用与 Kohya 相同的：
   - 采样器（scheduler）
   - CFG Scale（guidance_scale）
   - 推理步数（num_inference_steps）

2. **检查基础模型**：确保使用相同版本的 SD 模型

3. **查看详细文档**：参考 `LoRA问题诊断与解决方案.md` 获取更详细的帮助

## 示例配置

```yaml
# config.yaml

lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0
      trigger_words: ["yunjin_style"]  # ⬅️ 关键配置

llm:
  prompt_prefix: "yunjin_style"  # 自动添加触发词到所有提示词

default_scheduler: "DPMSolverMultistepScheduler"  # 与 Kohya 保持一致
```

## 常见问题

**Q: 我不知道触发词是什么？**
A: 查看训练配置或训练日志。如果找不到，尝试常见的触发词格式（如文件名、训练时使用的关键词等）。

**Q: 可以配置多个触发词吗？**
A: 可以，使用列表：`trigger_words: ["word1", "word2"]` 或字符串：`trigger_words: "word1, word2"`

**Q: 触发词必须完全匹配吗？**
A: 是的，必须与训练时使用的触发词完全一致（包括大小写和格式）。

**Q: 权重应该设置多少？**
A: 通常从 1.0 开始，如果效果太弱增加到 1.2-1.5，如果效果过强降低到 0.6-0.8。

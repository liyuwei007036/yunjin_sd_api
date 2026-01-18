# trigger_words 说明文档

## 什么是 trigger_words（触发词）？

**trigger_words** 是 LoRA 训练时使用的特殊关键词，用于激活 LoRA 模型的效果。

### 为什么需要触发词？

1. **LoRA 训练机制**：
   - 在 Kohya 等工具训练 LoRA 时，通常会使用特定的触发词（trigger words）
   - 这些触发词与训练数据中的标签相关联
   - 模型学会了"当看到这些触发词时，应用 LoRA 的风格/特征"

2. **激活 LoRA**：
   - 如果提示词中**没有**包含触发词，LoRA 可能不会生效或效果很弱
   - 触发词必须出现在提示词中，LoRA 才能被正确激活

### 示例

假设你训练了一个云锦风格的 LoRA，训练时使用的触发词是 `yunjin_style`：

- ❌ **错误**：`"a beautiful landscape, traditional Chinese patterns"`
  - 没有触发词，LoRA 可能不会生效

- ✅ **正确**：`"yunjin_style, a beautiful landscape, traditional Chinese patterns"`
  - 包含触发词，LoRA 会被激活

## 如何配置 trigger_words？

### 方式 1：为每个 LoRA 单独配置（推荐）

在 `config.yaml` 中：

```yaml
lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0
      trigger_words: ["yunjin_style"]  # ⬅️ 单个触发词
```

或多个触发词：

```yaml
lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0
      trigger_words: ["yunjin_style", "nanjing yunjin"]  # ⬅️ 多个触发词
```

或使用逗号分隔的字符串：

```yaml
trigger_words: "yunjin_style, nanjing yunjin"
```

### 方式 2：全局配置（所有 LoRA 共用）

如果多个 LoRA 使用相同的触发词：

```yaml
lora:
  models:
    - path: "./models/lora/model1.safetensors"
      weight: 1.0
      # 这个 LoRA 没有配置 trigger_words
    - path: "./models/lora/model2.safetensors"
      weight: 0.8
      # 这个 LoRA 也没有配置 trigger_words
  trigger_words: ["global_trigger"]  # ⬅️ 全局触发词，所有 LoRA 共用
```

## trigger_words 的实际使用

### ✅ 已实现的功能

系统会**自动**将触发词添加到提示词中：

1. **使用自然语言输入时**：
   - LLM 转换自然语言为提示词后，自动在提示词开头添加触发词
   - 例如：用户输入 `"一幅美丽的风景画"` → 系统生成 `"yunjin_style, a beautiful landscape painting"`

2. **手动提供 prompt 时**：
   - 如果提示词中不包含触发词，系统会自动添加
   - 例如：用户提供 `"a beautiful landscape"` → 系统自动变为 `"yunjin_style, a beautiful landscape"`

3. **避免重复**：
   - 如果提示词中已经包含触发词，不会重复添加
   - 例如：用户提供 `"yunjin_style, a beautiful landscape"` → 保持不变

### 触发词添加位置

触发词会被添加到提示词的**最前面**（最重要位置）：

```
[触发词], [其他提示词内容]
```

例如：
```
yunjin_style, a beautiful landscape, traditional Chinese patterns, 8k, masterpiece
```

## 如何找到正确的触发词？

### 方法 1：查看训练配置

1. 打开 Kohya 训练时的配置文件
2. 查找 `trigger_words` 或 `activation_words` 字段
3. 或者在训练日志中查找

### 方法 2：查看训练数据

1. 查看训练时使用的标签文件
2. 触发词通常是标签中的第一个词或特殊标记

### 方法 3：尝试常见格式

如果找不到，可以尝试：
- LoRA 文件名（去掉扩展名）
- 训练时使用的关键词
- 风格名称（如 `yunjin_style`, `yunjin`）

## 配置示例

### 完整配置示例

```yaml
lora:
  models:
    - path: "D:\\yunjin_train\\SD1.5\\output\\yunjin_style.safetensors"
      weight: 1.0
      trigger_words: ["yunjin_style"]  # ⬅️ 关键配置

llm:
  prompt_prefix: "nanjing yunjin style"  # 这个也会自动添加，但触发词优先级更高
```

### 多个 LoRA 配置示例

```yaml
lora:
  models:
    - path: "./models/lora/style1.safetensors"
      weight: 1.0
      trigger_words: ["style1_trigger"]
    - path: "./models/lora/style2.safetensors"
      weight: 0.8
      trigger_words: ["style2_trigger"]
```

系统会自动将所有触发词添加到提示词中：
```
style1_trigger, style2_trigger, [其他提示词]
```

## 验证配置

启动服务时，查看日志确认触发词已配置：

```
[LoRA 1/1] 加载LoRA: D:\yunjin_train\SD1.5\output\yunjin_style.safetensors
  - 权重: 1.0
  - 触发词: yunjin_style  ⬅️ 确认这里显示了触发词
  ✓ LoRA加载成功: ...
```

如果看到警告：
```
⚠️ 未配置触发词！这可能导致LoRA效果不明显。
```

说明需要在配置中添加 `trigger_words`。

## 常见问题

### Q: 触发词必须完全匹配吗？

A: 是的，必须与训练时使用的触发词**完全一致**（包括大小写和格式）。

### Q: 可以配置多个触发词吗？

A: 可以，使用列表：`trigger_words: ["word1", "word2"]` 或字符串：`trigger_words: "word1, word2"`

### Q: 触发词会被添加到提示词的哪里？

A: 自动添加到提示词的**最前面**，这是最重要的位置。

### Q: 如果提示词中已经有触发词了，还会添加吗？

A: 不会，系统会检查提示词中是否已包含触发词，避免重复添加。

### Q: trigger_words 和 prompt_prefix 有什么区别？

A: 
- `trigger_words`：LoRA 专用的触发词，**必须**与训练时使用的触发词一致，优先级最高
- `prompt_prefix`：通用的提示词前缀，可以添加任何内容（如质量标签）

系统会先添加 `trigger_words`，再添加 `prompt_prefix`。

## 总结

- ✅ **trigger_words 是必需的**：如果 LoRA 训练时使用了触发词，必须在配置中设置
- ✅ **自动添加**：系统会自动将触发词添加到提示词中，无需手动添加
- ✅ **避免重复**：如果提示词中已包含触发词，不会重复添加
- ✅ **优先级最高**：触发词会被添加到提示词的最前面

配置好 `trigger_words` 后，重启服务即可生效！

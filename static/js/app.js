/**
 * 主应用逻辑
 */

// 全局变量
let currentTaskId = null;
let pollInterval = null;
let taskHistory = [];

// DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // 加载保存的API Key
    loadApiKey();
    
    // 绑定事件
    setupEventListeners();
    
    // 加载预设列表
    loadPresets();

    // 设置图片上传
    setupImageUpload();
}

function setupEventListeners() {
    // 标签页切换
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });

    // 表单提交
    document.getElementById('text2imgForm').addEventListener('submit', handleText2ImgSubmit);
    document.getElementById('img2imgForm').addEventListener('submit', handleImg2ImgSubmit);

    // 保存预设
    document.getElementById('savePresetBtn').addEventListener('click', savePreset);
    
    // 加载预设
    document.getElementById('loadPresetSelect').addEventListener('change', loadPreset);
    
    // 保存API Key
    document.getElementById('saveApiKeyBtn').addEventListener('click', saveApiKey);
    
    // API Key输入框回车保存
    document.getElementById('apiKeyInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            saveApiKey();
        }
    });
}

function switchMode(mode) {
    // 更新标签页状态
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // 切换表单面板
    document.querySelectorAll('.form-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    if (mode === 'text2img') {
        document.getElementById('text2imgForm').classList.add('active');
    } else {
        document.getElementById('img2imgForm').classList.add('active');
    }
}

function toggleCollapsible(btn) {
    const collapsible = btn.closest('.collapsible');
    collapsible.classList.toggle('active');
}

function setSize(width, height) {
    const mode = document.querySelector('.tab-btn.active').dataset.mode;
    if (mode === 'text2img') {
        document.getElementById('text2imgWidth').value = width;
        document.getElementById('text2imgHeight').value = height;
    }
}

function setupImageUpload() {
    const uploadArea = document.getElementById('img2imgUploadArea');
    const fileInput = document.getElementById('img2imgFileInput');
    const preview = document.getElementById('img2imgPreview');

    // 点击上传区域
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#ccc';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        }
    });
}

function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('img2imgPreview');
        const placeholder = document.querySelector('.upload-placeholder');
        
        preview.src = e.target.result;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function handleText2ImgSubmit(e) {
    e.preventDefault();

    const formData = {
        natural_language: document.getElementById('text2imgNaturalLanguage').value.trim(),
        prompt: document.getElementById('text2imgPrompt').value.trim() || null,
        negative_prompt: document.getElementById('text2imgNegativePrompt').value.trim() || null,
        width: parseInt(document.getElementById('text2imgWidth').value) || null,
        height: parseInt(document.getElementById('text2imgHeight').value) || null,
        num_images: parseInt(document.getElementById('text2imgNumImages').value) || 1,
        num_inference_steps: parseInt(document.getElementById('text2imgSteps').value) || null,
        guidance_scale: parseFloat(document.getElementById('text2imgGuidance').value) || null,
        scheduler: document.getElementById('text2imgScheduler').value || null,
        seed: document.getElementById('text2imgSeed').value ? parseInt(document.getElementById('text2imgSeed').value) : null,
        output_format: document.querySelector('input[name="text2imgFormat"]:checked').value || 'png',
        callback_url: document.getElementById('text2imgCallback').value.trim() || null
    };

    // 如果没有自然语言且没有prompt，报错
    if (!formData.natural_language && !formData.prompt) {
        alert('请填写自然语言描述或Prompt');
        return;
    }

    // 如果填写了prompt，优先使用prompt，移除natural_language（避免AI转换）
    if (formData.prompt) {
        delete formData.natural_language;
        // 如果negative_prompt为空，也删除
        if (!formData.negative_prompt) delete formData.negative_prompt;
    } else if (formData.natural_language) {
        // 如果只有natural_language，移除空的prompt和negative_prompt
        if (!formData.prompt) delete formData.prompt;
        if (!formData.negative_prompt) delete formData.negative_prompt;
    }

    try {
        await submitGenerateRequest(formData);
    } catch (error) {
        showError(error.message);
    }
}

async function handleImg2ImgSubmit(e) {
    e.preventDefault();

    const fileInput = document.getElementById('img2imgFileInput');
    if (!fileInput.files || !fileInput.files[0]) {
        alert('请上传图片');
        return;
    }

    // 读取图片为base64
    const file = fileInput.files[0];
    const base64Image = await readFileAsBase64(file);

    const formData = {
        natural_language: document.getElementById('img2imgNaturalLanguage').value.trim(),
        init_image: base64Image,
        prompt: document.getElementById('img2imgPrompt').value.trim() || null,
        negative_prompt: document.getElementById('img2imgNegativePrompt').value.trim() || null,
        num_images: parseInt(document.getElementById('img2imgNumImages').value) || 1,
        num_inference_steps: parseInt(document.getElementById('img2imgSteps').value) || null,
        guidance_scale: parseFloat(document.getElementById('img2imgGuidance').value) || null,
        strength: (() => {
            const val = document.getElementById('img2imgStrength').value;
            return val !== '' ? parseFloat(val) : null;
        })(),
        scheduler: document.getElementById('img2imgScheduler').value || null,
        seed: document.getElementById('img2imgSeed').value ? parseInt(document.getElementById('img2imgSeed').value) : null,
        output_format: document.querySelector('input[name="img2imgFormat"]:checked').value || 'png',
        callback_url: document.getElementById('img2imgCallback').value.trim() || null
    };

    // 如果没有自然语言且没有prompt，报错
    if (!formData.natural_language && !formData.prompt) {
        alert('请填写自然语言描述或Prompt');
        return;
    }

    // 如果填写了prompt，优先使用prompt，移除natural_language（避免AI转换）
    if (formData.prompt) {
        delete formData.natural_language;
        // 如果negative_prompt为空，也删除
        if (!formData.negative_prompt) delete formData.negative_prompt;
    } else if (formData.natural_language) {
        // 如果只有natural_language，移除空的prompt和negative_prompt
        if (!formData.prompt) delete formData.prompt;
        if (!formData.negative_prompt) delete formData.negative_prompt;
    }

    try {
        await submitGenerateRequest(formData);
    } catch (error) {
        showError(error.message);
    }
}

function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function submitGenerateRequest(formData) {
    // 检查API Key是否已设置
    if (!api.apiKey || !api.apiKey.trim()) {
        const apiKey = prompt('请先设置API Key（从服务端config.yaml中获取）:');
        if (!apiKey || !apiKey.trim()) {
            showError('API Key未设置，无法生成图片');
            return;
        }
        api.setApiKey(apiKey.trim());
        // 保存到localStorage
        localStorage.setItem('api_key', apiKey.trim());
        // 更新输入框
        document.getElementById('apiKeyInput').value = apiKey.trim();
    }
    
    // 显示加载动画
    showLoadingAnimation();
    
    // 禁用所有生成按钮并显示加载状态
    const generateButtons = document.querySelectorAll('.btn-primary.btn-large');
    generateButtons.forEach(btn => {
        const originalText = btn.textContent;
        btn.dataset.originalText = originalText;
        btn.disabled = true;
        btn.style.opacity = '0.6';
        btn.style.cursor = 'not-allowed';
        btn.innerHTML = '🔄 生成中...';
    });
    
    // 清除null和空字符串的字段
    Object.keys(formData).forEach(key => {
        if (formData[key] === null || formData[key] === '') {
            delete formData[key];
        }
    });

    try {
        const response = await api.generateImage(formData);
        currentTaskId = response.task_id;

        // 添加到历史记录
        addToHistory({
            taskId: response.task_id,
            status: response.status,
            timestamp: new Date().toISOString(),
            params: formData
        });

        // 开始轮询任务状态
        startPolling(response.task_id);
    } catch (error) {
        // 隐藏加载动画
        hideLoadingAnimation();
        // 恢复按钮状态
        generateButtons.forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
        });
        throw error;
    }
}

function startPolling(taskId) {
    // 清除之前的轮询
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    // 显示任务状态
    showTaskStatus(taskId, 'pending');

    // 立即查询一次
    checkTaskStatus(taskId);

    // 每2秒查询一次
    pollInterval = setInterval(() => {
        checkTaskStatus(taskId);
    }, 2000);
}

async function checkTaskStatus(taskId) {
    try {
        const status = await api.getTaskStatus(taskId);
        updateTaskStatus(status);

        if (status.status === 'completed' || status.status === 'failed') {
            // 停止轮询
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }

            // 更新历史记录
            updateHistoryItem(taskId, status);

            // 如果完成，显示结果
            if (status.status === 'completed') {
                showResult(status);
            } else if (status.status === 'failed') {
                // 失败时也隐藏加载动画
                hideLoadingAnimation();
                const generateButtons = document.querySelectorAll('.btn-primary.btn-large');
                generateButtons.forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    btn.style.cursor = 'pointer';
                });
                showError(status.error_message || '任务执行失败');
            }
        }
    } catch (error) {
        console.error('查询任务状态失败:', error);
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
        showError('查询任务状态失败: ' + error.message);
    }
}

function showTaskStatus(taskId, status) {
    const taskStatusDiv = document.getElementById('taskStatus');
    const statusText = document.getElementById('statusText');
    const taskProgress = document.getElementById('taskProgress');

    taskStatusDiv.style.display = 'block';
    statusText.textContent = status;
    statusText.className = 'status-text ' + status;

    if (status === 'pending' || status === 'processing') {
        taskProgress.style.display = 'block';
    } else {
        taskProgress.style.display = 'none';
    }
}

function updateTaskStatus(status) {
    const statusText = document.getElementById('statusText');
    statusText.textContent = status.status;
    statusText.className = 'status-text ' + status.status;
}

function showResult(status) {
    // 隐藏加载动画
    hideLoadingAnimation();
    
    // 恢复按钮状态
    const generateButtons = document.querySelectorAll('.btn-primary.btn-large');
    generateButtons.forEach(btn => {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        if (btn.dataset.originalText) {
            btn.textContent = btn.dataset.originalText;
            delete btn.dataset.originalText;
        }
    });
    
    const resultImagesDiv = document.getElementById('resultImages');
    resultImagesDiv.innerHTML = '';

    // 显示prompt信息
    if (status.prompt || status.negative_prompt) {
        const promptDiv = document.createElement('div');
        promptDiv.className = 'prompt-info';
        let promptHTML = '<div class="prompt-info-header">使用的提示词：</div>';
        
        if (status.prompt) {
            promptHTML += `<div class="prompt-section"><strong>Prompt:</strong><div class="prompt-text">${escapeHtml(status.prompt)}</div></div>`;
        }
        
        if (status.negative_prompt) {
            promptHTML += `<div class="prompt-section"><strong>Negative Prompt:</strong><div class="prompt-text">${escapeHtml(status.negative_prompt)}</div></div>`;
        }
        
        promptDiv.innerHTML = promptHTML;
        resultImagesDiv.appendChild(promptDiv);
    }

    const urls = status.result_url ? [status.result_url] : (status.result_urls || []);

    urls.forEach((url, index) => {
        const item = document.createElement('div');
        item.className = 'result-image-item';
        item.innerHTML = `
            <img src="${url}" alt="Generated image ${index + 1}" loading="lazy">
            <div class="result-image-url">${url}</div>
            <div class="result-image-actions">
                <button class="btn btn-small" onclick="copyUrl('${url}')">复制URL</button>
                <button class="btn btn-small" onclick="downloadImage('${url}', '${index + 1}')">下载</button>
            </div>
        `;
        resultImagesDiv.appendChild(item);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyUrl(url) {
    navigator.clipboard.writeText(url).then(() => {
        alert('URL已复制到剪贴板');
    }).catch(() => {
        // 降级方案
        const textarea = document.createElement('textarea');
        textarea.value = url;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        alert('URL已复制到剪贴板');
    });
}

function downloadImage(url, index) {
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated-image-${index}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function showError(message) {
    // 隐藏加载动画
    hideLoadingAnimation();
    
    // 恢复按钮状态
    const generateButtons = document.querySelectorAll('.btn-primary.btn-large');
    generateButtons.forEach(btn => {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        if (btn.dataset.originalText) {
            btn.textContent = btn.dataset.originalText;
            delete btn.dataset.originalText;
        }
    });
    
    const resultImagesDiv = document.getElementById('resultImages');
    resultImagesDiv.innerHTML = `<div class="error-message">错误: ${message}</div>`;
}

// 显示加载动画
function showLoadingAnimation() {
    const resultImagesDiv = document.getElementById('resultImages');
    resultImagesDiv.innerHTML = `
        <div class="loading-container">
            <div class="loading-spinner">
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
            </div>
            <div class="loading-text">正在生成图片，请稍候...</div>
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
}

// 隐藏加载动画
function hideLoadingAnimation() {
    // 加载动画会在showResult或showError中被清除，这里不需要额外操作
}

// 生成随机种子
function generateRandomSeed(seedId) {
    const seedInput = document.getElementById(seedId);
    // 生成一个0到2^31-1之间的随机整数
    const randomSeed = Math.floor(Math.random() * 2147483647);
    seedInput.value = randomSeed;
}

// 预设管理
function savePreset() {
    const mode = document.querySelector('.tab-btn.active').dataset.mode;
    const presetName = prompt('请输入预设名称:');
    if (!presetName) return;

    const preset = {
        mode: mode,
        timestamp: new Date().toISOString()
    };

    if (mode === 'text2img') {
        preset.natural_language = document.getElementById('text2imgNaturalLanguage').value;
        preset.prompt = document.getElementById('text2imgPrompt').value;
        preset.negative_prompt = document.getElementById('text2imgNegativePrompt').value;
        preset.width = document.getElementById('text2imgWidth').value;
        preset.height = document.getElementById('text2imgHeight').value;
        preset.num_images = document.getElementById('text2imgNumImages').value;
        preset.num_inference_steps = document.getElementById('text2imgSteps').value;
        preset.guidance_scale = document.getElementById('text2imgGuidance').value;
        preset.scheduler = document.getElementById('text2imgScheduler').value;
        preset.seed = document.getElementById('text2imgSeed').value;
        preset.output_format = document.querySelector('input[name="text2imgFormat"]:checked').value;
    } else {
        preset.natural_language = document.getElementById('img2imgNaturalLanguage').value;
        preset.prompt = document.getElementById('img2imgPrompt').value;
        preset.negative_prompt = document.getElementById('img2imgNegativePrompt').value;
        preset.num_images = document.getElementById('img2imgNumImages').value;
        preset.num_inference_steps = document.getElementById('img2imgSteps').value;
        preset.guidance_scale = document.getElementById('img2imgGuidance').value;
        preset.strength = document.getElementById('img2imgStrength').value;
        preset.scheduler = document.getElementById('img2imgScheduler').value;
        preset.seed = document.getElementById('img2imgSeed').value;
        preset.output_format = document.querySelector('input[name="img2imgFormat"]:checked').value;
    }

    const presets = JSON.parse(localStorage.getItem('presets') || '{}');
    presets[presetName] = preset;
    localStorage.setItem('presets', JSON.stringify(presets));

    loadPresets();
    alert('预设已保存');
}

function loadPresets() {
    const presets = JSON.parse(localStorage.getItem('presets') || '{}');
    const select = document.getElementById('loadPresetSelect');
    
    // 清除现有选项（保留默认选项）
    select.innerHTML = '<option value="">加载预设...</option>';
    
    Object.keys(presets).forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
    });
}

function loadPreset() {
    const select = document.getElementById('loadPresetSelect');
    const presetName = select.value;
    if (!presetName) return;

    const presets = JSON.parse(localStorage.getItem('presets') || '{}');
    const preset = presets[presetName];
    if (!preset) return;

    // 切换到对应模式
    switchMode(preset.mode);

    // 延迟加载值，等待表单切换完成
    setTimeout(() => {
        if (preset.mode === 'text2img') {
            if (preset.natural_language) document.getElementById('text2imgNaturalLanguage').value = preset.natural_language;
            if (preset.prompt) document.getElementById('text2imgPrompt').value = preset.prompt;
            if (preset.negative_prompt) document.getElementById('text2imgNegativePrompt').value = preset.negative_prompt;
            if (preset.width) document.getElementById('text2imgWidth').value = preset.width;
            if (preset.height) document.getElementById('text2imgHeight').value = preset.height;
            if (preset.num_images) document.getElementById('text2imgNumImages').value = preset.num_images;
            if (preset.num_inference_steps) document.getElementById('text2imgSteps').value = preset.num_inference_steps;
            if (preset.guidance_scale) document.getElementById('text2imgGuidance').value = preset.guidance_scale;
            if (preset.scheduler) document.getElementById('text2imgScheduler').value = preset.scheduler;
            if (preset.seed) document.getElementById('text2imgSeed').value = preset.seed;
            if (preset.output_format) document.querySelector(`input[name="text2imgFormat"][value="${preset.output_format}"]`).checked = true;
        } else {
            if (preset.natural_language) document.getElementById('img2imgNaturalLanguage').value = preset.natural_language;
            if (preset.prompt) document.getElementById('img2imgPrompt').value = preset.prompt;
            if (preset.negative_prompt) document.getElementById('img2imgNegativePrompt').value = preset.negative_prompt;
            if (preset.num_images) document.getElementById('img2imgNumImages').value = preset.num_images;
            if (preset.num_inference_steps) document.getElementById('img2imgSteps').value = preset.num_inference_steps;
            if (preset.guidance_scale) document.getElementById('img2imgGuidance').value = preset.guidance_scale;
            if (preset.strength) document.getElementById('img2imgStrength').value = preset.strength;
            if (preset.scheduler) document.getElementById('img2imgScheduler').value = preset.scheduler;
            if (preset.seed) document.getElementById('img2imgSeed').value = preset.seed;
            if (preset.output_format) document.querySelector(`input[name="img2imgFormat"][value="${preset.output_format}"]`).checked = true;
        }

        // 重置选择框
        select.value = '';
    }, 100);
}

// API Key管理
function loadApiKey() {
    const savedApiKey = localStorage.getItem('api_key');
    if (savedApiKey) {
        api.setApiKey(savedApiKey);
        document.getElementById('apiKeyInput').value = savedApiKey;
    }
}

function saveApiKey() {
    const apiKeyInput = document.getElementById('apiKeyInput');
    const apiKey = apiKeyInput.value.trim();
    
    if (!apiKey) {
        alert('请输入API Key');
        return;
    }
    
    api.setApiKey(apiKey);
    localStorage.setItem('api_key', apiKey);
    alert('API Key已保存');
}

// 历史记录管理
function addToHistory(task) {
    taskHistory.unshift(task);
    if (taskHistory.length > 20) {
        taskHistory = taskHistory.slice(0, 20);
    }
    updateHistoryList();
}

function updateHistoryItem(taskId, status) {
    const index = taskHistory.findIndex(t => t.taskId === taskId);
    if (index !== -1) {
        taskHistory[index].status = status.status;
        taskHistory[index].result = status.result_url || status.result_urls;
        updateHistoryList();
    }
}

function updateHistoryList() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';

    taskHistory.forEach(task => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const time = new Date(task.timestamp).toLocaleString('zh-CN');
        item.innerHTML = `
            <div class="history-item-header">
                <span>任务 ${task.taskId.substring(0, 8)}...</span>
                <span class="history-item-time">${time}</span>
            </div>
            <div class="status-text ${task.status}">${task.status}</div>
        `;
        item.addEventListener('click', () => {
            if (task.status === 'completed' && task.result) {
                const result = Array.isArray(task.result) ? task.result : [task.result];
                showResult({ result_url: null, result_urls: result });
            }
            startPolling(task.taskId);
        });
        historyList.appendChild(item);
    });
}

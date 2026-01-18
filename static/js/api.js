/**
 * API调用封装
 */

const API_BASE = '/api/v1';

class API {
    constructor() {
        this.apiKey = '';
    }

    setApiKey(key) {
        this.apiKey = key;
    }

    getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
        };
        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }
        return headers;
    }

    /**
     * 生成图片
     */
    async generateImage(params) {
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * 查询任务状态
     */
    async getTaskStatus(taskId) {
        const response = await fetch(`${API_BASE}/tasks/${taskId}`, {
            method: 'GET',
            headers: this.getHeaders()
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    }
}

// 导出全局API实例
window.api = new API();

"""
FastAPI应用入口
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routers import image
from app.config import Config
from app.models.schemas import HealthResponse

# 创建FastAPI应用
app = FastAPI(
    title="SD模型API服务",
    description="基于Stable Diffusion的图片生成API服务，支持文生图和图生图",
    version="1.0.0",
)

# 配置CORS（如果需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(image.router)


@app.get("/health", response_model=HealthResponse, summary="健康检查")
@app.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    健康检查接口
    
    返回服务状态和模型加载状态
    默认免认证（可通过config.yaml中的health_check.no_auth配置）
    """
    # 检查模型是否已加载
    model_loaded = False
    try:
        # 尝试获取SD服务实例来判断模型是否已加载
        from app.routers.image import get_sd_service
        sd_service = get_sd_service()
        model_loaded = sd_service is not None and sd_service.text2img_pipeline is not None
    except Exception:
        pass
    
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded
    )


@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("=" * 50)
    print("SD模型API服务启动中...")
    print("=" * 50)
    
    # 验证配置
    errors = Config.validate()
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        print("请检查config.yaml配置文件")
    else:
        print("配置验证通过")
    
    # 预加载模型（通过访问服务实例触发）
    print("预加载SD模型...")
    try:
        from app.routers.image import get_sd_service
        sd_service = get_sd_service()
        print("SD模型预加载完成")
    except Exception as e:
        print(f"SD模型预加载失败: {e}")
    
    print("=" * 50)
    print("服务启动完成！")
    print("API文档地址: http://localhost:8000/docs")
    print("=" * 50)


@app.get("/", summary="根路径")
async def root():
    """根路径"""
    return {
        "message": "SD模型API服务",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

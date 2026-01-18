"""
FastAPI应用入口
"""
import sys
from pathlib import Path

# 添加项目根目录到sys.path，确保可以导入app模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尽早初始化日志系统，抑制第三方库的详细日志
from app.utils.logger import logger

from contextlib import asynccontextmanager
import gc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from app.routers import image
from app.config import Config
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 50)
    logger.info("SD模型API服务启动中...")
    logger.info("=" * 50)

    # 验证配置
    errors = Config.validate()
    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.error("请检查config.yaml配置文件")
    else:
        logger.info("配置验证通过")

    # 预加载模型（通过访问服务实例触发）
    logger.info("预加载SD模型...")
    try:
        from app.routers.image import get_sd_service
        sd_service = get_sd_service()
        logger.info("SD模型预加载完成")
    except Exception as e:
        logger.error(f"SD模型预加载失败: {e}", exc_info=True)

    logger.info("=" * 50)
    logger.info("服务启动完成！")
    logger.info("API文档地址: http://localhost:8000/docs")
    logger.info("=" * 50)

    yield

    # 关闭时执行资源清理
    logger.info("服务正在关闭...")
    try:
        # 清理路由中的服务实例
        from app.routers.image import cleanup_services
        cleanup_services()

        # 清理LLM服务（如果有）
        from app.services.llm_service import _llm_service
        if _llm_service is not None:
            # LLM服务通常不需要特殊清理，主要是HTTP客户端会自动关闭
            gc.collect()
            logger.info("LLM服务已清理")

        # 最后强制垃圾回收
        gc.collect()

        # 清理PyTorch CUDA缓存（如果有CUDA）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("PyTorch CUDA缓存已清理")
        except Exception:
            pass

        logger.info("资源清理完成")
    except Exception as e:
        logger.error(f"关闭服务时出错: {e}", exc_info=True)


# 创建FastAPI应用
app = FastAPI(
    title="SD模型API服务",
    description="基于Stable Diffusion的图片生成API服务，支持文生图和图生图",
    version="1.0.0",
    lifespan=lifespan,
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

# 配置静态文件服务
static_dir = project_root / "static"
if static_dir.exists():
    # 挂载静态文件目录（用于CSS、JS等资源）
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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


@app.get("/", summary="根路径")
async def root():
    """根路径，返回前端页面"""
    static_dir = project_root / "static"
    index_file = static_dir / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        return {
            "message": "SD模型API服务",
            "docs": "/docs",
            "health": "/health"
        }


@app.get("/api/v1/images/{bucket}/{filename:path}", summary="图片代理接口")
async def proxy_image(bucket: str, filename: str):
    """
    图片代理接口：从MinIO获取图片并返回给客户端
    
    这样可以将MinIO的内部URL转换为可通过服务端访问的URL
    """
    try:
        # 使用单例OSS服务实例
        from app.routers.image import get_oss_service
        oss_service = get_oss_service()

        # 从MinIO获取图片对象
        response = oss_service.client.get_object(bucket, filename)

        # 确定内容类型
        content_type = "image/png"  # 默认
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            content_type = "image/jpeg"
        elif filename.lower().endswith('.png'):
            content_type = "image/png"

        # 返回图片流
        return StreamingResponse(
            response.stream(32 * 1024),  # 32KB chunks
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=31536000",  # 缓存1年
            }
        )
    except Exception as e:
        logger.error(f"获取图片失败: bucket={bucket}, filename={filename}, error={str(e)}")
        raise HTTPException(status_code=404, detail=f"图片不存在: {filename}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

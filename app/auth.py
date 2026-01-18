"""
API Key认证中间件
"""
from typing import Optional
from fastapi import HTTPException, Security, Depends, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import Config


# HTTP Bearer安全方案
security = HTTPBearer(auto_error=False)


async def get_api_key_from_header(
    authorization: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    从Header中获取API Key
    
    支持格式：
    - X-API-Key: <api_key>
    - Authorization: Bearer <api_key>
    """
    # 方式1: 从X-API-Key获取
    if x_api_key:
        return x_api_key
    
    # 方式2: 从Authorization Bearer获取
    if authorization:
        return authorization.credentials
    
    return None


async def get_api_key_from_query(
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> Optional[str]:
    """
    从Query参数中获取API Key
    
    格式: ?api_key=xxx
    """
    return api_key


async def verify_api_key(
    api_key_header: Optional[str] = Depends(get_api_key_from_header),
    api_key_query: Optional[str] = Depends(get_api_key_from_query),
) -> str:
    """
    验证API Key
    
    Args:
        api_key_header: 从Header获取的API Key
        api_key_query: 从Query参数获取的API Key
    
    Returns:
        API Key字符串
    
    Raises:
        HTTPException: 如果API Key无效
    """
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="缺少API Key。请在Header中提供X-API-Key或Authorization: Bearer <api_key>"
        )
    
    if api_key not in Config.API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="无效的API Key"
        )
    
    return api_key


# 依赖项：用于需要认证的路由
def require_auth(api_key: str = Depends(verify_api_key)) -> str:
    """认证依赖项"""
    return api_key

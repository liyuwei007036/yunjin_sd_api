"""
MinIO OSS上传服务
"""
import io
import uuid
from typing import List, Union, Literal
from PIL import Image
from minio import Minio
from minio.error import S3Error
from app.config import Config
from app.utils.logger import logger


class OSSService:
    """OSS上传服务类"""
    
    def __init__(self):
        """初始化MinIO客户端"""
        self.client = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            secure=False  # 如果使用HTTPS，设置为True
        )
        self.bucket = Config.MINIO_BUCKET
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """确保bucket存在，如果不存在则创建"""
        # 首先尝试直接创建bucket（如果不存在）
        # 这样可以避免 bucket_exists() 的权限检查问题
        try:
            self.client.make_bucket(self.bucket)
            logger.info(f"创建bucket: {self.bucket}")
        except S3Error as e:
            # BucketAlreadyOwnedByYou 或 BucketAlreadyExists 表示bucket已存在，这是正常情况
            if e.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                logger.info(f"Bucket {self.bucket} 已存在")
            # AccessDenied 可能表示：
            # 1. bucket已存在但没有检查权限（可以继续使用）
            # 2. 没有创建权限（需要检查用户权限）
            elif e.code == "AccessDenied":
                # 尝试检查bucket是否存在（即使没有ListBucket权限，也可能有GetBucketLocation权限）
                try:
                    # 尝试列出bucket中的对象（即使为空）来验证bucket是否存在
                    # 如果bucket不存在，会返回NoSuchBucket
                    objects = list(self.client.list_objects(self.bucket, max_keys=1))
                    logger.info(f"Bucket {self.bucket} 已存在（通过列表对象验证）")
                except S3Error as list_error:
                    if list_error.code == "NoSuchBucket":
                        logger.error(f"Bucket {self.bucket} 不存在且无法创建，请检查用户权限")
                        raise
                    elif list_error.code == "AccessDenied":
                        # 既不能创建也不能检查，可能是权限不足
                        logger.warning(
                            f"无法检查或创建bucket {self.bucket}，权限可能不足。"
                            f"请确保用户具有 s3:CreateBucket, s3:ListBucket, s3:GetBucketLocation 权限。"
                            f"服务将继续运行，但可能在上传时遇到问题"
                        )
                    else:
                        logger.warning(f"检查bucket时出错: {list_error}，服务将继续运行")
            else:
                # 其他错误，记录但不阻止初始化（可能是网络问题等临时性错误）
                logger.warning(f"创建bucket时出错: {e}。服务将继续运行，但可能在上传时遇到问题")
    
    def _generate_filename(self, output_format: str = "png") -> str:
        """生成唯一文件名"""
        file_ext = output_format.lower()
        if file_ext == "jpeg":
            file_ext = "jpg"
        return f"{uuid.uuid4().hex}.{file_ext}"
    
    def _image_to_bytes(self, image: Image.Image, output_format: str = "png") -> bytes:
        """将PIL Image转换为字节流"""
        buffer = io.BytesIO()
        
        if output_format.lower() in ["jpg", "jpeg"]:
            # JPEG格式始终使用100%质量
            image = image.convert("RGB")  # 确保是RGB模式
            image.save(buffer, format="JPEG", quality=100)
        else:
            # PNG格式
            image.save(buffer, format="PNG")
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def upload_image(self, image: Image.Image, output_format: str = "png") -> str:
        """
        上传单张图片到MinIO，返回服务端代理URL
        
        Args:
            image: PIL Image对象
            output_format: 输出格式，png/jpg/jpeg
        
        Returns:
            图片的服务端代理URL
        """
        filename = self._generate_filename(output_format)
        image_bytes = self._image_to_bytes(image, output_format)
        content_type = f"image/{output_format}" if output_format != "jpg" else "image/jpeg"
        
        try:
            self.client.put_object(self.bucket, filename, io.BytesIO(image_bytes), length=len(image_bytes), content_type=content_type)
            # 返回服务端代理URL，而不是MinIO直接URL
            # 这样可以通过服务端代理访问图片，避免直接访问MinIO的权限问题
            url = f"/api/v1/images/{self.bucket}/{filename}"
            return url
        except S3Error as e:
            logger.error(f"上传图片失败: {e}")
            raise
    
    def upload_images(self, images: List[Image.Image], output_format: str = "png") -> List[str]:
        """
        批量上传多张图片到MinIO，返回公开URL列表
        
        Args:
            images: PIL Image对象列表
            output_format: 输出格式，png/jpg/jpeg
        
        Returns:
            图片URL列表
        """
        urls = []
        logger.info(f"开始批量上传图片: count={len(images)}, format={output_format}")
        for idx, image in enumerate(images, 1):
            url = self.upload_image(image, output_format)
            urls.append(url)
            logger.debug(f"批量上传进度: {idx}/{len(images)}, url={url}")
        logger.info(f"批量上传完成: count={len(urls)}, urls={urls}")
        return urls

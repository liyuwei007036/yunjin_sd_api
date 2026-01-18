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
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"创建bucket: {self.bucket}")
        except S3Error as e:
            # 如果是权限错误，尝试直接创建bucket（可能没有检查权限但有创建权限）
            if e.code == "AccessDenied":
                try:
                    self.client.make_bucket(self.bucket)
                    logger.info(f"创建bucket: {self.bucket}")
                except S3Error as create_error:
                    # 如果创建也失败，可能是bucket已存在但没有访问权限，记录警告但不阻止初始化
                    # 实际使用时可能仍然可以上传文件
                    if create_error.code == "BucketAlreadyOwnedByYou" or create_error.code == "BucketAlreadyExists":
                        logger.warning(f"Bucket {self.bucket} 已存在但无法检查，可能在后续使用中正常工作")
                    else:
                        logger.warning(f"无法创建或检查bucket {self.bucket}: {create_error}。如果bucket已存在，服务可能仍可正常工作")
            else:
                # 其他错误，记录但不阻止初始化（可能是网络问题等临时性错误）
                logger.warning(f"检查bucket失败: {e}。服务将继续运行，但可能在上传时遇到问题")
    
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
        上传单张图片到MinIO，返回公开URL
        
        Args:
            image: PIL Image对象
            output_format: 输出格式，png/jpg/jpeg
        
        Returns:
            图片的公开URL
        """
        filename = self._generate_filename(output_format)
        image_bytes = self._image_to_bytes(image, output_format)
        content_type = f"image/{output_format}" if output_format != "jpg" else "image/jpeg"
        
        try:
            self.client.put_object(self.bucket, filename, io.BytesIO(image_bytes), length=len(image_bytes), content_type=content_type)
            # 生成公开URL（根据MinIO配置调整）
            # 如果配置了公开访问，可以直接使用endpoint + bucket + filename
            url = f"http://{Config.MINIO_ENDPOINT}/{self.bucket}/{filename}"
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
        for image in images:
            url = self.upload_image(image, output_format)
            urls.append(url)
        return urls

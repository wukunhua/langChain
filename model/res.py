from typing import Any, Optional, Generic, TypeVar
from enum import Enum
from pydantic import BaseModel

# 定义泛型，用于支持不同 data 类型
T = TypeVar('T')

# 定义状态码枚举
class StatusCode(Enum):
    """业务状态码"""
    # 成功状态 (200-299)
    SUCCESS = 200
    CREATED = 2001
    ACCEPTED = 2002
    
    # 客户端错误 (4000-4999)
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    VALIDATION_ERROR = 422
    
    # 服务器错误 (5000-5999)
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503
    TIMEOUT = 504
    
    # 业务自定义错误 (6000-6999)
    BUSINESS_ERROR = 6000
    DATA_NOT_EXIST = 6001
    DUPLICATE_DATA = 6002
    PERMISSION_DENIED = 6003


class Response(BaseModel):
    """统一 API 响应格式"""
    success: bool          # 是否成功
    code: int              # 业务状态码
    message: str           # 提示信息
    data: Optional[Any] = None   # 返回数据
    timestamp: int = None  # 时间戳
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "code": 2000,
                "message": "操作成功",
                "data": {},
                "timestamp": 1705123456
            }
        }
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = int(time.time())
        super().__init__(**data)
    
    @classmethod
    def ok(cls, data: Any = None, message: str = "操作成功"):
        """成功响应"""
        return cls(
            success=True,
            code=StatusCode.SUCCESS.value,
            message=message,
            data=data
        )
    
    @classmethod
    def error(cls, code: int = None, message: str = "操作失败", data: Any = None):
        """错误响应"""
        if code is None:
            code = StatusCode.BUSINESS_ERROR.value
        return cls(
            success=False,
            code=code,
            message=message,
            data=data
        )
    
    @classmethod
    def bad_request(cls, message: str = "请求参数错误"):
        """400 错误"""
        return cls(
            success=False,
            code=StatusCode.BAD_REQUEST.value,
            message=message
        )
    
    @classmethod
    def not_found(cls, message: str = "资源不存在"):
        """404 错误"""
        return cls(
            success=False,
            code=StatusCode.NOT_FOUND.value,
            message=message
        )
    
    @classmethod
    def unauthorized(cls, message: str = "未授权访问"):
        """401 错误"""
        return cls(
            success=False,
            code=StatusCode.UNAUTHORIZED.value,
            message=message
        )
    
    @classmethod
    def forbidden(cls, message: str = "无权限访问"):
        """403 错误"""
        return cls(
            success=False,
            code=StatusCode.FORBIDDEN.value,
            message=message
        )
    
    @classmethod
    def internal_error(cls, message: str = "服务器内部错误"):
        """500 错误"""
        return cls(
            success=False,
            code=StatusCode.INTERNAL_ERROR.value,
            message=message
        )


# 导入 time 模块
import time
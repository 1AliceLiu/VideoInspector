"""控制面 API 的公共异常类型。"""

from __future__ import annotations


class ApiError(Exception):
    """带 HTTP 状态码和稳定错误码的 API 异常。"""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

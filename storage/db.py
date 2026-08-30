"""引擎与会话工厂 — DATABASE_URL 驱动（仿 Dify 的 12-factor 配置方式）。

- 主后端: mysql+pymysql（utf8mb4），库不存在时自动 CREATE DATABASE
- 降级:   MySQL 连接失败时回退 SQLite（沿用原 data/cisp_qa.db），仅告警不崩溃
"""
from __future__ import annotations

import logging
import os
import threading
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

import config

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_backend = None  # "mysql" | "sqlite"
_lock = threading.Lock()


def _mysql_url() -> str:
    user = os.getenv("MYSQL_USER", "root")
    password = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    name = os.getenv("MYSQL_DB", "cisp_qa")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


def _create_database_if_missing(url: str) -> None:
    """连接不含库名的 URL，CREATE DATABASE IF NOT EXISTS。"""
    base = url.rsplit("/", 1)[0]
    name = url.rsplit("/", 1)[1].split("?")[0]
    server_engine = create_engine(base + "/", pool_pre_ping=True)
    with server_engine.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{name}` "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        conn.commit()
    server_engine.dispose()


def _init() -> tuple:
    global _engine, _SessionLocal, _backend
    with _lock:
        if _engine is not None:
            return _engine, _SessionLocal, _backend

        mysql_url = config.MYSQL_URL_OVERRIDE or _mysql_url()
        if os.getenv("MYSQL_PASSWORD") or config.MYSQL_URL_OVERRIDE:
            try:
                _create_database_if_missing(mysql_url)
                _engine = create_engine(
                    mysql_url, pool_pre_ping=True, poolclass=QueuePool,
                    pool_size=5, pool_recycle=1800,
                )
                with _engine.connect():
                    pass  # 真连一次才算数
                _backend = "mysql"
                logger.info("存储后端: MySQL (%s)", mysql_url.rsplit("@", 1)[1])
                _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
                return _engine, _SessionLocal, _backend
            except Exception:
                logger.warning("MySQL 连接失败，降级 SQLite: %s", str(os.getenv("MYSQL_PASSWORD", "")) and "凭据已配置但不正确或服务不可达")

        sqlite_url = f"sqlite:///{config.DB_PATH}"
        _engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
        _backend = "sqlite"
        logger.info("存储后端: SQLite (%s)", config.DB_PATH)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
        return _engine, _SessionLocal, _backend


def backend() -> str:
    return _init()[2]


def get_engine():
    return _init()[0]


def session() -> Session:
    return _init()[1]()


def create_all() -> None:
    """建表（Alembic 之外的开发便捷路径；等价于 Dify 的 db.create_all）。"""
    from storage.models import Base
    Base.metadata.create_all(get_engine())


def override_engine(engine) -> None:
    """测试钩子：替换全局引擎/会话工厂为指定实例。"""
    global _engine, _SessionLocal, _backend
    with _lock:
        _engine = engine
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        _backend = "sqlite"

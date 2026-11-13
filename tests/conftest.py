"""共享 fixtures：隔离的临时库、可注入参数的配置、建库与种子数据辅助函数。"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import aiosqlite
import pytest


@pytest.fixture
def collector_env(tmp_path, monkeypatch):
    """把数据目录指向临时路径，并按需覆盖限速参数；重新加载 config 使其生效。

    返回一个可调用对象，用于在 import app.* 之前追加环境变量并重新加载配置。
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("COLLECTOR_DB", str(data_dir / "collector.db"))

    def _reload(**overrides: str):
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        import app.config as config

        importlib.reload(config)
        return config

    yield _reload, data_dir


@pytest.fixture
def config(collector_env):
    reload_config, _ = collector_env
    cfg = reload_config(
        # 测试默认加速：轮询与心跳更密，租约更短
        POLL_INTERVAL_SECONDS="0.02",
        HEARTBEAT_INTERVAL_SECONDS="0.1",
        LEASE_SECONDS="2",
        SHUTDOWN_GRACE_SECONDS="3",
        HTTP_TIMEOUT_SECONDS="10",
        WORKER_CONCURRENCY="5",
    )
    return cfg


async def init_database(db_path: str | Path) -> None:
    """在独立连接上建表（等价于 API/worker 启动时的 init_db）。"""
    from app import db

    conn = await db.connect(str(db_path))
    try:
        await db.init_db(conn)
    finally:
        await conn.close()


async def seed_job(db_path: str | Path, source_base: str, items: list[tuple[str, str]]):
    """直接写入一个任务与其 pending 条目，返回 (job_id, item_ids)。"""
    now = time.time()
    conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    try:
        cur = await conn.execute(
            "INSERT INTO jobs(source_base, created_at) VALUES (?, ?)",
            (source_base, now),
        )
        job_id = cur.lastrowid
        await conn.executemany(
            "INSERT INTO items(job_id, station_id, date, status, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?)",
            [(job_id, station, date, now) for station, date in items],
        )
        await conn.execute(
            "INSERT INTO source_states(source_base) VALUES (?) "
            "ON CONFLICT(source_base) DO NOTHING",
            (source_base,),
        )
        rows = await conn.execute_fetchall(
            "SELECT id FROM items WHERE job_id=? ORDER BY id", (job_id,)
        )
        return job_id, [r[0] for r in rows]
    finally:
        await conn.close()


async def fetch_source_state(db_path: str | Path, source_base: str):
    conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    conn.row_factory = aiosqlite.Row
    try:
        row = await conn.execute_fetchall(
            "SELECT * FROM source_states WHERE source_base=?", (source_base,)
        )
        slots = await conn.execute_fetchall(
            "SELECT * FROM inflight_slots WHERE source_base=?", (source_base,)
        )
        items = await conn.execute_fetchall(
            "SELECT id, status, attempts, precipitation, error_code, error_message FROM items "
            "ORDER BY id"
        )
        return dict(row[0]) if row else None, [dict(s) for s in slots], [
            dict(i) for i in items
        ]
    finally:
        await conn.close()


@pytest.fixture
def helpers():
    return type(
        "H",
        (),
        {
            "init_database": staticmethod(init_database),
            "seed_job": staticmethod(seed_job),
            "fetch_source_state": staticmethod(fetch_source_state),
        },
    )

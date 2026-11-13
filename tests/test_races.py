"""两类发送竞态的确定性回归测试。

测试一 test_request_gap_survives_write_lock_contention：
    另一连接持有 BEGIN IMMEDIATE 写锁约 1.3 秒，两个 worker 的认领动作被挤在
    写锁释放后的同一时刻——修复前实际发包间隔会小于 1 秒；修复后发送闸门
    在锁内用新鲜时间戳裁决，实际开始间隔仍 ≥ 1 秒。

测试二 test_claimed_unsent_items_wait_for_cooldown：
    条目已认领但尚未发送时，另一条请求收到 429 写入冷却——修复前前者仍会
    发出；修复后所有未发送条目在发送闸门前等待冷却结束，且冷却按
    “收到响应时刻 + Retry-After”计算，后续响应不能缩短。

另含冷却的 max 语义/重启持久化、源隔离与“等待不占写事务”、端到端可靠性测试。
"""
from __future__ import annotations

import asyncio
import time

import aiosqlite
import httpx
import pytest

from app.worker import Worker


# ---------------------------------------------------------------- 测试辅助
def make_transport(handler, sent: list):
    """构造脚本化 transport，并记录每次请求真正发出的墙钟时刻。

    直接把 async handler 交给 httpx.MockTransport：其 handle_async_request
    会识别协程并 await（见 httpx 源码）。记录逻辑在 handler 外层完成，
    包装器必须返回 Response 而非协程，因此记录与调用合并进同一个 async 函数。
    """

    async def recording_handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        sent.append((time.time(), params.get("station_id"), params.get("date")))
        return await handler(request)

    return httpx.MockTransport(recording_handler)


async def ok_response(request: httpx.Request) -> httpx.Response:
    params = request.url.params
    return httpx.Response(
        200,
        json={
            "station_id": params["station_id"],
            "date": params["date"],
            "precipitation": 2.5,
            "unit": "mm",
        },
    )


async def stop_worker(w: Worker) -> None:
    """关闭测试 worker：只做库内清理（在途项回 pending、释放名额、注销心跳），
    不经过 _shutdown——测试里直接创建的游离任务由调用方自行 await。"""
    w.shutdown_event.set()
    assert w.client is not None and w.conn is not None and w.worker_pk is not None
    now = time.time()
    async with w.write_lock:
        await w.conn.execute("BEGIN IMMEDIATE")
        try:
            await w.conn.execute(
                "UPDATE items SET status='pending', worker_id=NULL, "
                "lease_until=NULL, updated_at=? "
                "WHERE worker_id=? AND status='running'",
                (now, w.worker_pk),
            )
            await w.conn.execute(
                "DELETE FROM inflight_slots WHERE worker_id=?", (w.worker_pk,)
            )
            await w.conn.execute(
                "UPDATE workers SET active=0, last_heartbeat=? WHERE id=?",
                (now, w.worker_pk),
            )
            await w.conn.execute("COMMIT")
        except Exception:
            await w.conn.execute("ROLLBACK")
            raise
    await w.client.aclose()
    await w.conn.close()


# ==================================================== 竞态一：写锁挤压下的 1 秒间隔
async def test_request_gap_survives_write_lock_contention(
    config, tmp_path, helpers, monkeypatch
):
    monkeypatch.setattr(config, "MIN_REQUEST_GAP_SECONDS", 1.0)
    db_path = config.COLLECTOR_DB
    await helpers.init_database(db_path)
    source = "http://src-gap"
    _, item_ids = await helpers.seed_job(
        db_path, source, [("S1", "2026-11-01"), ("S2", "2026-11-01")]
    )

    sent: list = []
    transport = make_transport(ok_response, sent)
    w1 = Worker(transport=transport, hostname="w1")
    w2 = Worker(transport=transport, hostname="w2")
    await w1.start()
    await w2.start()

    # 第三个连接模拟“另一采集进程”持写锁约 1.3 秒（题目给出的时长）
    locker = await aiosqlite.connect(str(db_path), isolation_level=None)
    await locker.execute("PRAGMA busy_timeout=10000")
    await locker.execute("BEGIN IMMEDIATE")
    try:
        # 两个 worker 的认领并发排队在写锁上；持锁 1.3 秒后释放，
        # 两者会在锁释放后的同一窗口内先后完成认领
        t_claim_start = time.time()
        acquire_task = asyncio.gather(w1._acquire(source), w2._acquire(source))
        await asyncio.sleep(1.3)
        await locker.execute("ROLLBACK")
        state1, state2 = await asyncio.wait_for(acquire_task, timeout=8)
        t_claim_done = time.time()

        # 前置条件：两次认领确实被写锁挤压，几乎同时发生（间隔远小于 1 秒）
        assert state1 is not None and state2 is not None
        assert t_claim_done - t_claim_start >= 1.2  # 确实经历了写锁等待

        # 两条都已认领、占名额，但此时一个请求都还没发出
        _, slots, _ = await helpers.fetch_source_state(db_path, source)
        assert len(slots) == 2
        assert sent == []

        task1 = asyncio.create_task(w1._process_item(source, *state1[:3]))
        task2 = asyncio.create_task(w2._process_item(source, *state2[:3]))
        await asyncio.gather(task1, task2)

        # 核心断言：尽管认领被挤压，实际请求开始间隔仍 ≥ 1 秒
        assert len(sent) == 2
        real_gap = sent[1][0] - sent[0][0]
        assert real_gap + 1e-9 >= config.MIN_REQUEST_GAP_SECONDS
        # 也不应过度等待（闸门只补够 1 秒，不含额外长等）
        assert real_gap < config.MIN_REQUEST_GAP_SECONDS + 0.6

        state, slots, items = await helpers.fetch_source_state(db_path, source)
        # last_started_at 记录的是真正发包时刻，而不是认领时刻
        assert abs(state["last_started_at"] - sent[1][0]) < 0.3
        # 不残留名额、不丢条目、不重复保存
        assert slots == []
        assert len(items) == 2
        assert all(i["status"] == "succeeded" for i in items)
        assert len({i["id"] for i in items}) == len(item_ids)
    finally:
        await locker.close()
        await stop_worker(w1)
        await stop_worker(w2)


# ==================================================== 竞态二：429 冷却拦住已认领未发送条目
async def test_claimed_unsent_items_wait_for_cooldown(
    config, tmp_path, helpers, monkeypatch
):
    # 隔离冷却语义：关闭 1 秒间隔干扰，放宽在途上限以便同时持有 3 个名额
    monkeypatch.setattr(config, "MIN_REQUEST_GAP_SECONDS", 0.0)
    monkeypatch.setattr(config, "MAX_INFLIGHT_PER_SOURCE", 3)

    db_path = config.COLLECTOR_DB
    await helpers.init_database(db_path)
    source = "http://src-429"
    stations = ["A", "B", "C"]
    await helpers.seed_job(
        db_path, source, [(s, "2026-11-02") for s in stations]
    )

    b_429_done = asyncio.Event()
    b_received_at: dict[str, float] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        station = request.url.params["station_id"]
        if station == "B" and not b_429_done.is_set():
            await asyncio.sleep(0.4)  # 让其他条目先完成“认领但尚未发送”的布局
            b_received_at["t"] = time.time()
            b_429_done.set()
            return httpx.Response(429, headers={"Retry-After": "1"})
        return await ok_response(request)

    sent: list = []
    w = Worker(transport=make_transport(handler, sent), hostname="w-429")
    await w.start()
    try:
        # 阶段 1：三条全部认领，没有任何请求发出
        claimed = []
        for _ in range(3):
            state = await w._acquire(source)
            assert state is not None
            claimed.append(state)
        assert sent == []
        _, slots_before, _ = await helpers.fetch_source_state(db_path, source)
        assert len(slots_before) == 3

        # 阶段 2：只放 B 出去，B 收到 429 并写入冷却、回 pending、释放名额
        b_state = next(s for s in claimed if s[1] == "B")
        tb = asyncio.create_task(w._process_item(source, *b_state[:3]))
        await asyncio.wait_for(tb, timeout=5)

        state_row, slots_mid, items_mid = await helpers.fetch_source_state(
            db_path, source
        )
        cooldown_until = state_row["cooldown_until"]
        # 冷却 = 收到 429 响应的时刻 + Retry-After
        assert cooldown_until is not None
        assert abs(cooldown_until - (b_received_at["t"] + 1.0)) < 0.2
        # B 已回 pending 且名额释放；A、C 仍在 running（已认领未发送）
        statuses = {i["id"]: i["status"] for i in items_mid}
        b_item = b_state[0]
        assert statuses[b_item] == "pending"
        assert len(slots_mid) == 2
        # 此刻只有 B 的请求真正发出过
        assert [s[1] for s in sent] == ["B"]

        # 阶段 3：A、C 此刻才开始跑——它们在 429 之前就已认领，
        # 修复前会无视冷却直接发出；修复后必须在发送闸门前等到冷却结束
        other_states = [s for s in claimed if s[1] in ("A", "C")]
        tasks = [asyncio.create_task(w._process_item(source, *s[:3])) for s in other_states]

        # 冷却窗口内确认它们一直没发出（等待不占写事务，见下条断言）
        await asyncio.sleep(0.4)
        assert [s[1] for s in sent] == ["B"]

        # 冷却等待不占库写事务：另一连接可立刻拿到写锁并完成写入
        locker = await aiosqlite.connect(str(db_path), isolation_level=None)
        t0 = time.time()
        await locker.execute("BEGIN IMMEDIATE")
        await locker.execute(
            "UPDATE workers SET last_heartbeat=last_heartbeat WHERE id=?",
            (w.worker_pk,),
        )
        await locker.execute("COMMIT")
        await locker.close()
        assert time.time() - t0 < 0.3

        await asyncio.gather(*tasks)

        # A、C 的实际请求开始时刻都不得早于冷却结束时刻
        for send_t, station, _ in sent:
            if station in ("A", "C"):
                assert send_t + 1e-9 >= cooldown_until

        # 阶段 4：B 重新认领并完成（冷却结束后允许重试成功）
        b_again = await w._acquire(source)
        assert b_again is not None and b_again[0] == b_item
        await w._process_item(source, *b_again[:3])
    finally:
        await stop_worker(w)

    # 最终：三条全部成功、无残留名额、无丢失、无重复保存
    state_row, slots, items = await helpers.fetch_source_state(db_path, source)
    assert slots == []
    assert len(items) == 3
    assert all(i["status"] == "succeeded" for i in items)
    sent_stations = [s[1] for s in sent]
    assert sorted(set(sent_stations)) == ["A", "B", "C"]
    assert sent_stations.count("B") == 2  # B 经历一次 429 后重试
    b_item_row = next(i for i in items if i["id"] == b_item)
    assert b_item_row["attempts"] == 2


# ==================================================== 冷却只延长不缩短 + 重启持久化
async def test_cooldown_max_semantics_and_restart_persistence(
    config, tmp_path, helpers, monkeypatch
):
    monkeypatch.setattr(config, "MIN_REQUEST_GAP_SECONDS", 0.0)
    db_path = config.COLLECTOR_DB
    await helpers.init_database(db_path)
    source = "http://src-persist"
    await helpers.seed_job(db_path, source, [("P1", "2026-11-03")])

    w = Worker(transport=make_transport(ok_response, []), hostname="w-p1")
    await w.start()
    try:
        state = await w._acquire(source)
        assert state is not None
        item_id = state[0]

        # 第一次 429：Retry-After=1
        r1 = time.time()
        await w._on_rate_limited(source, item_id, r1, 1)
        row, _, _ = await helpers.fetch_source_state(db_path, source)
        first_until = row["cooldown_until"]
        assert abs(first_until - (r1 + 1.0)) < 0.05

        # 紧接着第二次 429：Retry-After=0，不能把冷却缩短
        await w._on_rate_limited(source, item_id, time.time(), 0)
        row, _, _ = await helpers.fetch_source_state(db_path, source)
        assert row["cooldown_until"] == first_until

        # 第三次 429：Retry-After=5，必须延长（取 max）
        r3 = time.time()
        await w._on_rate_limited(source, item_id, r3, 5)
        row, _, _ = await helpers.fetch_source_state(db_path, source)
        assert abs(row["cooldown_until"] - (r3 + 5.0)) < 0.05
        assert row["cooldown_until"] > first_until
    finally:
        await stop_worker(w)

    # 重启：全新 worker 进程读同一个库，冷却仍是库内绝对值，内存无法重置
    sent: list = []
    w2 = Worker(transport=make_transport(ok_response, sent), hostname="w-p2")
    await w2.start()
    try:
        row, _, _ = await helpers.fetch_source_state(db_path, source)
        assert row["cooldown_until"] > time.time()  # 冷却仍在生效
        # 冷却结束前新进程领不到该源的条目（认领阶段也会读库内冷却）
        assert await w2._acquire(source) is None
        # 等库中冷却结束（绝对值判定，非内存计时），随后认领并处理
        await asyncio.sleep(max(0.0, row["cooldown_until"] - time.time()) + 0.05)
        state2 = await w2._acquire(source)
        assert state2 is not None and state2[0] == item_id
        t0 = time.time()
        await w2._process_item(source, *state2[:3])
        # 重启后的首次发包不得早于库中冷却结束时刻，且没有额外的长等待
        assert sent[0][0] + 1e-9 >= row["cooldown_until"]
        assert sent[0][0] - t0 < 0.5

        _, slots, items = await helpers.fetch_source_state(db_path, source)
        assert slots == []
        assert all(i["status"] == "succeeded" for i in items)
    finally:
        await stop_worker(w2)


# ==================================================== 源隔离 + 等待不堵查询/其他源
async def test_cooldown_isolated_per_source_and_reads_unblocked(
    config, tmp_path, helpers, monkeypatch
):
    monkeypatch.setattr(config, "MIN_REQUEST_GAP_SECONDS", 0.0)
    db_path = config.COLLECTOR_DB
    await helpers.init_database(db_path)
    hot = "http://src-hot"
    other = "http://src-other"
    await helpers.seed_job(db_path, hot, [("H1", "2026-11-04")])
    await helpers.seed_job(db_path, other, [("O1", "2026-11-04")])

    sent: list = []

    async def handler(request: httpx.Request) -> httpx.Response:
        return await ok_response(request)

    w = Worker(transport=make_transport(handler, sent), hostname="w-iso")
    await w.start()
    try:
        hot_state = await w._acquire(hot)
        other_state = await w._acquire(other)
        assert hot_state is not None and other_state is not None

        # 两个条目均已认领但尚未发送；此时给 hot 源写入 1 秒冷却（模拟另一条
        # 进行中的请求刚收到 429），hot 条目必须在发送闸门前等冷却结束
        cooldown_until = time.time() + 1.0
        async with w.write_lock:
            await w.conn.execute("BEGIN IMMEDIATE")
            await w.conn.execute(
                "UPDATE source_states SET cooldown_until=? WHERE source_base=?",
                (cooldown_until, hot),
            )
            await w.conn.execute("COMMIT")
        w.mem_cooldown[hot] = cooldown_until

        hot_task = asyncio.create_task(w._process_item(hot, *hot_state[:3]))
        other_task = asyncio.create_task(w._process_item(other, *other_state[:3]))

        # other 源立即发出，不被 hot 的冷却拖累
        await asyncio.wait_for(other_task, timeout=2)
        assert [s[1] for s in sent] == ["O1"]

        # 冷却窗口内 hot 一直发不出去（等待不占写事务，下面同时验证只读不堵）
        ro = await aiosqlite.connect(
            f"file:{db_path}?mode=ro", uri=True, isolation_level=None
        )
        for _ in range(5):
            t0 = time.time()
            await ro.execute("SELECT COUNT(*) FROM items")
            assert time.time() - t0 < 0.2
            await asyncio.sleep(0.1)
        await ro.close()
        assert [s[1] for s in sent] == ["O1"]

        await asyncio.wait_for(hot_task, timeout=5)
        hot_row, _, _ = await helpers.fetch_source_state(db_path, hot)
        hot_send = next(t for t, s, _ in sent if s == "H1")
        assert hot_send + 1e-9 >= hot_row["cooldown_until"]
    finally:
        await stop_worker(w)

    _, slots, items = await helpers.fetch_source_state(db_path, hot)
    assert slots == []
    assert all(i["status"] == "succeeded" for i in items)


# ==================================================== 端到端：主循环下不丢、不重、名额不残留
async def test_end_to_end_main_loop_with_429_and_inflight_cap(
    config, tmp_path, helpers, monkeypatch
):
    monkeypatch.setattr(config, "MIN_REQUEST_GAP_SECONDS", 0.3)
    db_path = config.COLLECTOR_DB
    await helpers.init_database(db_path)
    source = "http://src-e2e"
    n_items = 6
    await helpers.seed_job(
        db_path, source, [(f"S{i}", "2026-11-05") for i in range(n_items)]
    )

    call_count = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] % 3 == 0:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return await ok_response(request)

    sent: list = []
    w = Worker(transport=make_transport(handler, sent), hostname="w-e2e")

    # 采样在途峰值，断言同源全局在途始终 ≤ 2
    observed_max = {"v": 0}
    stop_sampler = asyncio.Event()

    async def sampler():
        conn = await aiosqlite.connect(str(db_path), isolation_level=None)
        try:
            while not stop_sampler.is_set():
                cur = await conn.execute(
                    "SELECT COUNT(*) AS c FROM inflight_slots WHERE source_base=?",
                    (source,),
                )
                observed_max["v"] = max(observed_max["v"], (await cur.fetchone())[0])
                await asyncio.sleep(0.02)
        finally:
            await conn.close()

    sampler_task = asyncio.create_task(sampler())
    run_task = asyncio.create_task(w.run())

    conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    conn.row_factory = aiosqlite.Row
    try:
        deadline = time.time() + 25
        while time.time() < deadline:
            rows = await conn.execute_fetchall(
                "SELECT status, COUNT(*) AS c FROM items GROUP BY status"
            )
            counts = {r["status"]: r["c"] for r in rows}
            if counts.get("succeeded", 0) == n_items:
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("任务未在时限内全部完成")
    finally:
        stop_sampler.set()
        await sampler_task
        w.shutdown_event.set()
        await asyncio.wait_for(run_task, timeout=10)
        await conn.close()

    # 全部成功；每个条目恰好一行成功结果，无重复保存
    _, slots, items = await helpers.fetch_source_state(db_path, source)
    assert slots == []
    assert len(items) == n_items
    assert all(i["status"] == "succeeded" for i in items)
    assert observed_max["v"] <= config.MAX_INFLIGHT_PER_SOURCE
    assert any(i["attempts"] >= 2 for i in items)  # 429 重试痕迹

    # 所有相邻实际请求开始间隔 ≥ 配置值
    times = sorted(t for t, _, _ in sent)
    for a, b in zip(times, times[1:]):
        assert b - a + 1e-9 >= config.MIN_REQUEST_GAP_SECONDS

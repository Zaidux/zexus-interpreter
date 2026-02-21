"""
Zexus Blockchain — Production Monitoring & Metrics
====================================================

Real-time metrics collection for blockchain node operators.

Tracks:
* **Block metrics** — block time, height, size, gas per block
* **Transaction metrics** — TPS, latency, success / fail rates
* **Peer metrics** — peer count, message rates, banned peers
* **Resource metrics** — mempool depth, storage size, cache hits
* **Consensus metrics** — round time, finality latency

Exposes metrics in two ways:

1. **Programmatic** — ``NodeMetrics.snapshot()`` returns a dict.
2. **HTTP endpoint** — ``MetricsServer`` serves ``/metrics`` in
   Prometheus-compatible text format and ``/metrics/json`` as JSON.

Usage::

    from zexus.blockchain.monitoring import NodeMetrics, MetricsServer

    metrics = NodeMetrics()
    metrics.record_block(block)
    metrics.record_transaction(receipt)

    # Optional: start HTTP metrics endpoint
    server = MetricsServer(metrics, port=9100)
    await server.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("zexus.blockchain.monitoring")


# ── Core Metrics ───────────────────────────────────────────────────────

class NodeMetrics:
    """Lightweight, thread-safe metrics collector for a blockchain node.

    All public methods are safe to call from any thread.
    """

    def __init__(self, window: int = 100):
        """
        Parameters
        ----------
        window : int
            Number of recent data-points to keep for rolling averages.
        """
        self._lock = threading.Lock()
        self._window = window

        # Block metrics
        self.blocks_produced: int = 0
        self.current_height: int = 0
        self._block_times: Deque[float] = deque(maxlen=window)
        self._block_gas: Deque[int] = deque(maxlen=window)
        self._block_sizes: Deque[int] = deque(maxlen=window)
        self._last_block_time: float = 0.0

        # Transaction metrics
        self.tx_total: int = 0
        self.tx_succeeded: int = 0
        self.tx_failed: int = 0
        self._tx_latencies: Deque[float] = deque(maxlen=window)

        # Peer metrics
        self.peer_count: int = 0
        self.peers_banned: int = 0
        self.messages_in: int = 0
        self.messages_out: int = 0

        # Mempool
        self.mempool_depth: int = 0
        self.mempool_bytes: int = 0

        # Consensus
        self._consensus_rounds: Deque[float] = deque(maxlen=window)
        self._finality_latencies: Deque[float] = deque(maxlen=window)

        # Uptime
        self._start_time: float = time.time()

    # ── Recording methods ──────────────────────────────────────────

    def record_block(self, block: Any) -> None:
        """Record a newly produced/received block."""
        now = time.time()
        with self._lock:
            self.blocks_produced += 1
            height = getattr(block, "header", None)
            if height is not None:
                height = getattr(height, "height", 0)
            else:
                height = 0
            self.current_height = max(self.current_height, height)

            if self._last_block_time > 0:
                self._block_times.append(now - self._last_block_time)
            self._last_block_time = now

            # Gas / size
            txs = getattr(block, "transactions", [])
            tx_count = len(txs) if isinstance(txs, list) else 0
            self._block_sizes.append(tx_count)
            gas_total = sum(
                getattr(tx, "gas_limit", 0) for tx in (txs or [])
            )
            self._block_gas.append(gas_total)

    def record_transaction(self, receipt: Any) -> None:
        """Record a transaction result."""
        with self._lock:
            self.tx_total += 1
            success = False
            if isinstance(receipt, dict):
                success = receipt.get("success", False)
                latency = receipt.get("latency", 0.0)
            else:
                success = getattr(receipt, "success", False)
                latency = getattr(receipt, "latency", 0.0)

            if success:
                self.tx_succeeded += 1
            else:
                self.tx_failed += 1

            if latency > 0:
                self._tx_latencies.append(latency)

    def record_peer_count(self, count: int, banned: int = 0) -> None:
        with self._lock:
            self.peer_count = count
            self.peers_banned = banned

    def record_message(self, direction: str = "in") -> None:
        with self._lock:
            if direction == "in":
                self.messages_in += 1
            else:
                self.messages_out += 1

    def record_mempool(self, depth: int, size_bytes: int = 0) -> None:
        with self._lock:
            self.mempool_depth = depth
            self.mempool_bytes = size_bytes

    def record_consensus_round(self, duration: float) -> None:
        with self._lock:
            self._consensus_rounds.append(duration)

    def record_finality(self, latency: float) -> None:
        with self._lock:
            self._finality_latencies.append(latency)

    # ── Snapshot ───────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            avg_block_time = (
                sum(self._block_times) / len(self._block_times)
                if self._block_times else 0.0
            )
            avg_block_gas = (
                sum(self._block_gas) / len(self._block_gas)
                if self._block_gas else 0.0
            )
            avg_block_size = (
                sum(self._block_sizes) / len(self._block_sizes)
                if self._block_sizes else 0.0
            )
            avg_tx_latency = (
                sum(self._tx_latencies) / len(self._tx_latencies)
                if self._tx_latencies else 0.0
            )
            tps = self.tx_total / uptime if uptime > 0 else 0.0
            avg_consensus = (
                sum(self._consensus_rounds) / len(self._consensus_rounds)
                if self._consensus_rounds else 0.0
            )
            avg_finality = (
                sum(self._finality_latencies) / len(self._finality_latencies)
                if self._finality_latencies else 0.0
            )

            return {
                "uptime_seconds": round(uptime, 2),
                "block": {
                    "height": self.current_height,
                    "total_produced": self.blocks_produced,
                    "avg_block_time": round(avg_block_time, 3),
                    "avg_gas_per_block": round(avg_block_gas, 1),
                    "avg_txs_per_block": round(avg_block_size, 1),
                },
                "transaction": {
                    "total": self.tx_total,
                    "succeeded": self.tx_succeeded,
                    "failed": self.tx_failed,
                    "success_rate": round(
                        self.tx_succeeded / self.tx_total * 100, 1
                    ) if self.tx_total > 0 else 100.0,
                    "avg_latency_ms": round(avg_tx_latency * 1000, 2),
                    "tps": round(tps, 2),
                },
                "peer": {
                    "connected": self.peer_count,
                    "banned": self.peers_banned,
                    "messages_in": self.messages_in,
                    "messages_out": self.messages_out,
                },
                "mempool": {
                    "pending_txs": self.mempool_depth,
                    "size_bytes": self.mempool_bytes,
                },
                "consensus": {
                    "avg_round_time": round(avg_consensus, 3),
                    "avg_finality_latency": round(avg_finality, 3),
                },
            }

    # ── Prometheus-compatible text format ──────────────────────────

    def prometheus_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        s = self.snapshot()
        lines: List[str] = []

        def _g(name: str, value: Any, help_text: str = ""):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        _g("zexus_block_height", s["block"]["height"], "Current chain height")
        _g("zexus_blocks_produced_total", s["block"]["total_produced"], "Total blocks produced")
        _g("zexus_avg_block_time_seconds", s["block"]["avg_block_time"], "Average block time")
        _g("zexus_avg_gas_per_block", s["block"]["avg_gas_per_block"], "Avg gas usage per block")
        _g("zexus_tx_total", s["transaction"]["total"], "Total transactions processed")
        _g("zexus_tx_succeeded", s["transaction"]["succeeded"], "Successful transactions")
        _g("zexus_tx_failed", s["transaction"]["failed"], "Failed transactions")
        _g("zexus_tx_success_rate", s["transaction"]["success_rate"], "Tx success rate %")
        _g("zexus_tps", s["transaction"]["tps"], "Current transactions per second")
        _g("zexus_avg_tx_latency_ms", s["transaction"]["avg_latency_ms"], "Avg tx latency ms")
        _g("zexus_peers_connected", s["peer"]["connected"], "Connected peers")
        _g("zexus_peers_banned", s["peer"]["banned"], "Banned peers")
        _g("zexus_messages_in_total", s["peer"]["messages_in"], "Inbound messages")
        _g("zexus_messages_out_total", s["peer"]["messages_out"], "Outbound messages")
        _g("zexus_mempool_depth", s["mempool"]["pending_txs"], "Pending mempool txs")
        _g("zexus_mempool_bytes", s["mempool"]["size_bytes"], "Mempool size bytes")
        _g("zexus_avg_consensus_round_seconds", s["consensus"]["avg_round_time"], "Avg consensus round")
        _g("zexus_avg_finality_latency_seconds", s["consensus"]["avg_finality_latency"], "Avg finality latency")
        _g("zexus_uptime_seconds", s["uptime_seconds"], "Node uptime")

        return "\n".join(lines) + "\n"


# ── HTTP Metrics Server ───────────────────────────────────────────────

class MetricsServer:
    """Lightweight HTTP server that exposes node metrics.

    Endpoints:
    * ``GET /metrics``       — Prometheus text format
    * ``GET /metrics/json``  — JSON snapshot
    * ``GET /health``        — Simple health check

    Usage::

        server = MetricsServer(metrics, port=9100)
        await server.start()
        # ...
        await server.stop()
    """

    def __init__(self, metrics: NodeMetrics, host: str = "0.0.0.0", port: int = 9100):
        self.metrics = metrics
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_request, self.host, self.port
        )
        logger.info("Metrics server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Metrics server stopped")

    async def _handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            request = data.decode("utf-8", errors="replace")
            first_line = request.split("\r\n")[0] if request else ""
            path = first_line.split(" ")[1] if len(first_line.split(" ")) > 1 else "/"
        except Exception:
            writer.close()
            return

        if path == "/metrics":
            body = self.metrics.prometheus_text()
            content_type = "text/plain; version=0.0.4; charset=utf-8"
        elif path == "/metrics/json":
            body = json.dumps(self.metrics.snapshot(), indent=2)
            content_type = "application/json"
        elif path == "/health":
            body = json.dumps({"status": "ok"})
            content_type = "application/json"
        else:
            body = "Not Found"
            content_type = "text/plain"
            response = (
                f"HTTP/1.1 404 Not Found\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n{body}"
            )
            writer.write(response.encode())
            await writer.drain()
            writer.close()
            return

        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n{body}"
        )
        writer.write(response.encode())
        await writer.drain()
        writer.close()

"""Simple monitoring dashboard for hosted-tinker.

Serves a web UI at /dashboard showing:
- GPU utilization and memory
- API request counts and latency
- Active models and training status
- Server health

No JS framework — just HTML + htmx for auto-refresh.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

# In-memory metrics (reset on server restart)
_metrics = {
    "requests": defaultdict(int),       # endpoint -> count
    "latency_sum": defaultdict(float),   # endpoint -> total seconds
    "errors": defaultdict(int),          # endpoint -> error count
    "start_time": time.time(),
    "last_requests": [],                 # last 20 requests: (time, endpoint, latency, status)
}
_MAX_RECENT = 50


def record_request(endpoint: str, latency: float, status: int):
    """Record a request metric. Call from API middleware."""
    _metrics["requests"][endpoint] += 1
    _metrics["latency_sum"][endpoint] += latency
    if status >= 400:
        _metrics["errors"][endpoint] += 1
    _metrics["last_requests"].append((time.time(), endpoint, latency, status))
    if len(_metrics["last_requests"]) > _MAX_RECENT:
        _metrics["last_requests"] = _metrics["last_requests"][-_MAX_RECENT:]


def _get_gpu_info() -> list[dict]:
    """Get GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": parts[0], "name": parts[1],
                    "mem_used": int(parts[2]), "mem_total": int(parts[3]),
                    "util": int(parts[4]), "temp": int(parts[5]),
                })
        return gpus
    except Exception:
        return []


def _uptime() -> str:
    secs = int(time.time() - _metrics["start_time"])
    hours, secs = divmod(secs, 3600)
    mins, secs = divmod(secs, 60)
    return f"{hours}h {mins}m {secs}s"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    gpus = _get_gpu_info()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime = _uptime()

    total_requests = sum(_metrics["requests"].values())
    total_errors = sum(_metrics["errors"].values())

    # GPU cards
    gpu_html = ""
    for gpu in gpus:
        mem_pct = int(gpu["mem_used"] / gpu["mem_total"] * 100) if gpu["mem_total"] > 0 else 0
        bar_color = "#4caf50" if mem_pct < 50 else "#ff9800" if mem_pct < 80 else "#f44336"
        util_color = "#4caf50" if gpu["util"] < 50 else "#ff9800" if gpu["util"] < 80 else "#f44336"
        gpu_html += f"""
        <div class="card">
            <h3>GPU {gpu['index']} — {gpu['name']}</h3>
            <div class="metric">
                <span>Memory</span>
                <span>{gpu['mem_used']} / {gpu['mem_total']} MiB ({mem_pct}%)</span>
            </div>
            <div class="bar"><div class="bar-fill" style="width:{mem_pct}%;background:{bar_color}"></div></div>
            <div class="metric">
                <span>Utilization</span>
                <span style="color:{util_color}">{gpu['util']}%</span>
            </div>
            <div class="bar"><div class="bar-fill" style="width:{gpu['util']}%;background:{util_color}"></div></div>
            <div class="metric"><span>Temperature</span><span>{gpu['temp']}°C</span></div>
        </div>"""

    # Request stats
    req_rows = ""
    for endpoint in sorted(_metrics["requests"].keys()):
        count = _metrics["requests"][endpoint]
        avg_lat = _metrics["latency_sum"][endpoint] / count if count > 0 else 0
        errors = _metrics["errors"][endpoint]
        req_rows += f"""
        <tr>
            <td>{endpoint}</td>
            <td>{count}</td>
            <td>{avg_lat*1000:.0f}ms</td>
            <td style="color:{'#f44336' if errors > 0 else '#4caf50'}">{errors}</td>
        </tr>"""

    # Recent requests
    recent_html = ""
    for ts, endpoint, latency, status in reversed(_metrics["last_requests"][-20:]):
        t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        color = "#4caf50" if status < 400 else "#f44336"
        recent_html += f"""
        <tr>
            <td>{t}</td>
            <td>{endpoint}</td>
            <td>{latency*1000:.0f}ms</td>
            <td style="color:{color}">{status}</td>
        </tr>"""

    # Check health
    try:
        import httpx
        health = httpx.get("http://localhost:8000/api/v1/healthz", timeout=2).json()
        health_status = "Healthy" if health.get("status") == "ok" else "Unhealthy"
        health_color = "#4caf50" if health_status == "Healthy" else "#f44336"
    except Exception:
        health_status = "Unknown"
        health_color = "#999"

    # Check vLLM
    try:
        import httpx
        models = httpx.get("http://localhost:8001/v1/models", timeout=2).json()
        vllm_status = f"{len(models.get('data', []))} model(s)"
        vllm_color = "#4caf50"
    except Exception:
        vllm_status = "Not running"
        vllm_color = "#999"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hosted Tinker Dashboard</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #fff; margin-bottom: 5px; }}
        h2 {{ color: #aaa; font-size: 14px; margin-bottom: 20px; }}
        h3 {{ color: #ddd; margin-bottom: 10px; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .card {{ background: #16213e; border-radius: 8px; padding: 15px; border: 1px solid #333; }}
        .metric {{ display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0; }}
        .bar {{ background: #333; border-radius: 4px; height: 8px; margin: 4px 0 8px; }}
        .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .status {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #aaa; font-weight: 500; }}
        .section {{ margin-bottom: 25px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 20px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: 700; }}
        .stat-label {{ font-size: 11px; color: #888; }}
    </style>
</head>
<body hx-get="/dashboard" hx-trigger="every 5s" hx-swap="innerHTML" hx-target="body">
    <div class="header">
        <div>
            <h1>Hosted Tinker</h1>
            <h2>Self-hosted training + inference server</h2>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" style="color:{health_color}">{health_status}</div>
                <div class="stat-label">Engine</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color:{vllm_color}">{vllm_status}</div>
                <div class="stat-label">vLLM Inference</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_requests}</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color:{'#f44336' if total_errors > 0 else '#4caf50'}">{total_errors}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat">
                <div class="stat-value">{uptime}</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h3>GPUs</h3>
        <div class="grid">{gpu_html if gpu_html else '<div class="card">No GPUs detected</div>'}</div>
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px;">
        <div class="card section">
            <h3>API Endpoints</h3>
            <table>
                <tr><th>Endpoint</th><th>Requests</th><th>Avg Latency</th><th>Errors</th></tr>
                {req_rows if req_rows else '<tr><td colspan="4" style="color:#666">No requests yet</td></tr>'}
            </table>
        </div>

        <div class="card section">
            <h3>Recent Requests</h3>
            <table>
                <tr><th>Time</th><th>Endpoint</th><th>Latency</th><th>Status</th></tr>
                {recent_html if recent_html else '<tr><td colspan="4" style="color:#666">No requests yet</td></tr>'}
            </table>
        </div>
    </div>

    <div style="text-align:center; color:#555; font-size:11px; margin-top:20px;">
        {now} · Auto-refreshes every 5s
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)

#!/usr/bin/env python
"""
Alpaca Bot Manager 🤖
Unified command center for your trading bot fleet.

Usage:
    python manage.py status   # Check what's running & view latest logs
    python manage.py train    # Start GPU Training (Hidden Mode)
    python manage.py stop     # Stop Training
    python manage.py clean    # Clean logs and pycache
"""
import os
import sys
import shutil
import time
import subprocess
from datetime import datetime

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
VENV_PYTHON = os.path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe')
TRAIN_SCRIPT = os.path.join(ROOT_DIR, 'scripts', 'train_multimodal.py')
TRAIN_PID_FILE = os.path.join(LOG_DIR, 'training.pid')

# Ensure log dir exists
os.makedirs(LOG_DIR, exist_ok=True)


def _read_tracked_training_pid():
    if not os.path.exists(TRAIN_PID_FILE):
        return None
    try:
        with open(TRAIN_PID_FILE, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        return int(raw) if raw else None
    except Exception:
        return None


def _write_tracked_training_pid(pid: int):
    try:
        with open(TRAIN_PID_FILE, 'w', encoding='utf-8') as f:
            f.write(str(int(pid)))
    except Exception:
        pass


def _is_pid_running(pid: int) -> bool:
    if not pid:
        return False
    pid_check = run_command(
        (
            "powershell -command \""
            f"Get-Process -Id {pid} -ErrorAction SilentlyContinue "
            "| Measure-Object | Select-Object -ExpandProperty Count"
            "\""
        )
    )
    try:
        return bool(pid_check and int(pid_check.strip()) > 0)
    except Exception:
        return False


def _find_train_gpu_pid():
    """Return PID of a python process running train_multimodal.py, if found."""
    ps = (
        "powershell -command \""
        "Get-CimInstance Win32_Process -Filter 'Name=\\\"python.exe\\\"' "
        "| Where-Object {$_.CommandLine -match 'train_multimodal\\.py'} "
        "| Select-Object -First 1 -ExpandProperty ProcessId"
        "\""
    )
    out = run_command(ps)
    try:
        return int(out.strip()) if out.strip() else None
    except Exception:
        return None


def _find_all_train_gpu_pids():
    """Return a list of PIDs for python processes running train_multimodal.py."""
    ps = (
        "powershell -command \""
        "Get-CimInstance Win32_Process -Filter 'Name=\\\"python.exe\\\"' "
        "| Where-Object {$_.CommandLine -match 'train_multimodal\\.py'} "
        "| Select-Object -ExpandProperty ProcessId"
        "\""
    )
    out = run_command(ps)
    pids = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except Exception:
            continue
    return sorted(set(pids))


def run_command(cmd, shell=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.check_output(cmd, shell=shell, stderr=subprocess.STDOUT)
        return result.decode('utf-8', errors='ignore').strip()
    except subprocess.CalledProcessError:
        return ""


def get_tail(filepath, n=3):
    """Get last n lines of a file."""
    if not os.path.exists(filepath):
        return ["(No log file)"]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Filter distinct lines to avoid tqdm spam if possible
            return [line.strip() for line in lines[-n:]]
    except Exception:
        return ["(Error reading log)"]


def show_status():
    print("\n" + "="*50)
    print("🤖 ALPACA BOT STATUS")
    print("="*50)
    
    # 1. Training Status (Local)
    print("\n🧠 TRAINING (Local GPU):")
    training_active = False
    
    # Check log freshness
    log_file = os.path.join(LOG_DIR, 'training.log')
    if os.path.exists(log_file):
        mtime = os.path.getmtime(log_file)
        params = (datetime.now().timestamp() - mtime)
        is_fresh = params < 120  # Updated in last 2 mins

        # Prefer PID tracking if available (safer than killing all python.exe).
        tracked_pid = _read_tracked_training_pid()
        pid_active = _is_pid_running(tracked_pid) if tracked_pid else False

        # If PID file looks stale but training is running, refresh the PID file.
        # Note: Do not delete PID files from status checks; this can race with
        # freshly-started processes. PID cleanup happens in `stop`.
        if tracked_pid and not pid_active:
            discovered_pid = _find_train_gpu_pid()
            if discovered_pid:
                _write_tracked_training_pid(discovered_pid)
                tracked_pid = discovered_pid
                pid_active = True
            else:
                # PID file is present but process is not running.
                # Treat as stopped; the log may still be "fresh" due to buffered writes.
                status = f"🔴 STOPPED (stale PID {tracked_pid})"
                tracked_pid = None
                pid_active = False

        # Detect duplicates (two trainers competing will crush throughput).
        all_pids = _find_all_train_gpu_pids()
        if len(all_pids) > 1:
            print(f"   ⚠️  Detected multiple trainers: {all_pids}")
            print("      Consider: python manage.py stop ; python manage.py train")

        # If we don't have a tracked PID, try to discover a running trainer.
        discovered_pid = None
        if not tracked_pid:
            discovered_pid = _find_train_gpu_pid()
            if discovered_pid:
                _write_tracked_training_pid(discovered_pid)
                tracked_pid = discovered_pid
                pid_active = True
        
        # Check if process is actually running (PID check is hard without psutil, assuming active if log moving)
        # Better: Assume active if fresh log + user knows they started it.
        # Actually, let's look for large memory python process?
        mem_check = run_command(
            (
                "powershell -command \""
                "Get-Process -Name python -ErrorAction SilentlyContinue "
                "| Where-Object {$_.WS -gt 500000000} "
                "| Measure-Object | Select-Object -ExpandProperty Count"
                "\""
            )
        )
        if pid_active:
            status = "🟢 ACTIVE (PID tracked)"
            training_active = True
        elif mem_check and int(mem_check.strip()) > 0:
            status = "🟢 ACTIVE (High Mem Process Detected)"
            training_active = True
        elif is_fresh and tracked_pid is None:
            status = "🔴 STOPPED (recent log activity, no trainer process found)"
        else:
            status = status if 'status' in locals() else "🔴 STOPPED"
    else:
        status = "🔴 STOPPED"
        
    print(f"   Status: {status}")
    print(f"   Log ({log_file}):")
    for line in get_tail(log_file, 3):
        print(f"     > {line}")

    # 2. Docker Bots
    print("\n🐳 DOCKER BOTS:")
    docker_ps = run_command("docker ps --format \"table {{.Names}}\t{{.Status}}\"")
    if "CONTAINER ID" not in docker_ps and "NAMES" not in docker_ps:
        print("   (Docker not reachable or no containers running)")
    else:
        # Parse docker output
        lines = docker_ps.split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                print(f"   {line}")
        else:
            print("   🔴 No active containers")

    print("\n" + "="*50)
    if training_active:
        print("💡 Tip: View training progress with: Get-Content logs/training.log -Wait")
    print("💡 Tip: Manage via: python manage.py [status|train|stop|clean]\n")


def start_training(follow: bool = False):
    print("🚀 Starting Training in BACKGROUND (Hidden)...")

    # Safety: stop any existing trainer(s) first to avoid duplicate runs.
    existing = _find_all_train_gpu_pids()
    if existing:
        print(f"⚠️  Found existing trainer PID(s): {existing}. Stopping them first...")
        for pid in existing:
            run_command(
                f"powershell -command \"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue\""
            )

    # Spawn directly so we can reliably track PID (safer stop/status).
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"

    out_path = os.path.join(LOG_DIR, 'training.log')
    err_path = os.path.join(LOG_DIR, 'training_err.log')

    def _start_detached_windows() -> int | None:
        """Start the trainer detached via PowerShell (more reliable in VS Code terminals).

        Many Windows terminals run commands in a job object that can terminate child
        processes when the parent exits. PowerShell Start-Process avoids that.
        """
        if os.name != 'nt':
            return None
        # Note: Start-Process redirection overwrites the target files.
        # For monitorability, we prefer a fresh log each run.
        ps = (
            "powershell -NoProfile -Command "
            "\"$p = Start-Process "
            f"-FilePath '{VENV_PYTHON}' "
            f"-ArgumentList '-u','{TRAIN_SCRIPT}' "
            f"-WorkingDirectory '{ROOT_DIR}' "
            "-WindowStyle Hidden "
            f"-RedirectStandardOutput '{out_path}' "
            f"-RedirectStandardError '{err_path}' "
            "-PassThru; "
            "$p.Id\""
        )
        out = run_command(ps)
        try:
            return int(out.strip()) if out.strip() else None
        except Exception:
            return None

    pid = _start_detached_windows()

    # Fallback: spawn directly (may be killed with parent depending on terminal/job settings)
    proc = None
    if pid is None:
        CREATE_NO_WINDOW = 0x08000000
        try:
            with open(out_path, 'a', encoding='utf-8') as out, open(err_path, 'a', encoding='utf-8') as err:
                proc = subprocess.Popen(
                    [VENV_PYTHON, TRAIN_SCRIPT],
                    stdout=out,
                    stderr=err,
                    env=env,
                    cwd=ROOT_DIR,
                    creationflags=CREATE_NO_WINDOW,
                )
            pid = proc.pid
        except Exception:
            pid = None

    if pid is None:
        print("❌ Failed to start trainer process.")
        return

    try:
        with open(TRAIN_PID_FILE, 'w', encoding='utf-8') as f:
            f.write(str(int(pid)))
    except Exception:
        pass

    # If a second trainer was started elsewhere, stop it to avoid split resources.
    # (Best-effort; safe no-op if process discovery fails.)
    time.sleep(1)
    for other_pid in _find_all_train_gpu_pids():
        if other_pid != int(pid):
            run_command(
                f"powershell -command \"Stop-Process -Id {other_pid} -Force -ErrorAction SilentlyContinue\""
            )

    print("✅ Command sent. Checking status...")
    show_status()

    if follow:
        print("\n📡 Following live training output (Ctrl+C to stop):")
        try:
            subprocess.call(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Get-Content '{out_path}' -Tail 200 -Wait",
                ],
                cwd=ROOT_DIR,
            )
        except KeyboardInterrupt:
            pass


def stop_local():
    tracked_pid = _read_tracked_training_pid()

    if tracked_pid and _is_pid_running(tracked_pid):
        print(f"🛑 Stopping training PID {tracked_pid}...")
        run_command(
            f"powershell -command \"Stop-Process -Id {tracked_pid} -Force -ErrorAction SilentlyContinue\""
        )
        try:
            os.remove(TRAIN_PID_FILE)
        except Exception:
            pass
        print("✅ Done.")
        # Also stop any other stray trainers.
        for pid in _find_all_train_gpu_pids():
            if pid != tracked_pid:
                run_command(
                    f"powershell -command \"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue\""
                )
        return

    # PID file present but process is not running: remove stale PID file.
    if tracked_pid and not _is_pid_running(tracked_pid):
        try:
            os.remove(TRAIN_PID_FILE)
        except Exception:
            pass

    # PID file missing/stale: stop any discovered trainer(s).
    discovered = _find_all_train_gpu_pids()
    if discovered:
        print(f"🛑 Stopping discovered training PID(s) {discovered}...")
        for pid in discovered:
            run_command(
                f"powershell -command \"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue\""
            )
        try:
            os.remove(TRAIN_PID_FILE)
        except Exception:
            pass
        print("✅ Done.")
        return

    print("⚠️  No active trainer found; not killing all python.exe.")
    print("   If you really need to stop everything: powershell Stop-Process -Name python -Force")


def clean_workspace():
    print("🧹 Cleaning workspace...")
    
    # 1. Clean pycache
    print("   - Removing __pycache__...")
    for root, dirs, _files in os.walk(ROOT_DIR):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
    
    # 2. Archive old logs (Optional: or just delete?)
    # Let's keep recent ones.
    
    print("✅ Cleanup complete.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    action = sys.argv[1].lower()
    
    if action == 'status':
        show_status()
    elif action == 'train':
        follow = ("--follow" in sys.argv[2:]) or (os.getenv("TRAIN_FOLLOW", "0") == "1")
        start_training(follow=follow)
    elif action == 'stop':
        stop_local()
    elif action == 'clean':
        clean_workspace()
    else:
        print(f"❌ Unknown command: {action}")
        print(__doc__)


if __name__ == "__main__":
    main()

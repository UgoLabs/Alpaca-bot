import time
import os
import argparse
import torch
import pandas as pd
import numpy as np
import subprocess
import sys

# Force UTF-8 on Windows (avoids crash when stdout is redirected or cp1252)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

import alpaca_trade_api as tradeapi
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import (
    SwingTraderCreds,
    PaperSwingTraderCreds,
    OptionsTraderCreds,
    SwingTraderConfig,
    OptionsTraderConfig,
    ALPACA_BASE_URL, OPTIONS_ALPACA_BASE_URL, PAPER_SWING_ALPACA_BASE_URL, SWING_MODEL_PATH,
    OPTIONS_MODEL_PATH, OPTIONS_BEARISH_MODEL_PATH, OPTIONS_USE_SWING_MODEL,
)
from src.execution.options_spread import OptionsSpreadBroker
from src.agents.ensemble_agent import EnsembleAgent
from src.data.pipeline import MultiModalDataPipeline
from src.core.feature_sets import get_feature_cols


class MultiModalTrader:
    """
    Live Trading Bot using the Multi-Modal Agent (Transformer + Vision + Text).
    """
    def __init__(self, watchlist_path: Optional[str] = None, mode: str = "swing", skip_data_update: bool = False):
        print(f"🤖 Initializing Multi-Modal Trader ({mode.replace('_', ' ').upper()} Mode)...")
        
        self.mode = mode.lower()
        self.watchlist_path = watchlist_path
        self._skip_data_update = skip_data_update
        
        if self.mode in ('swing', 'paper_swing'):
            self.config = SwingTraderConfig
            self.creds = (
                SwingTraderCreds if self.mode == 'swing' else PaperSwingTraderCreds
            )
        elif self.mode == 'options':
            self.creds = OptionsTraderCreds
            self.config = OptionsTraderConfig
        else:
            raise ValueError(f"Unknown mode: {mode} (use swing, paper_swing, or options)")

        api_base_url = str(ALPACA_BASE_URL)
        if self.mode == 'paper_swing':
            api_base_url = str(PAPER_SWING_ALPACA_BASE_URL)
        elif self.mode == 'options':
            api_base_url = str(OPTIONS_ALPACA_BASE_URL)
            allow_live = os.getenv("OPTIONS_ALLOW_LIVE", "").strip().lower() in (
                "1", "true", "yes",
            )
            if "paper" not in api_base_url.lower() and not allow_live:
                raise RuntimeError(
                    "Options mode requires a paper API host. "
                    "Set OPTIONS_ALPACA_BASE_URL=https://paper-api.alpaca.markets "
                    "(or OPTIONS_ALLOW_LIVE=1 for live)."
                )
        self._api_base_url = api_base_url
        if self.mode == 'paper_swing':
            print(
                "   Paper swing = live swing mirror (SwingTraderConfig, Gen7, swing_liquid); "
                "paper keys/host only.",
                flush=True,
            )

        # 1. API Connection
        self.api = tradeapi.REST(
            str(self.creds.API_KEY),
            str(self.creds.API_SECRET),
            api_base_url,
            api_version='v2'
        )
        
        # 2. Configuration
        self.symbols = self._load_watchlist()

        # 6. Swing CSV refresh — defer to pre-open window when market is closed
        if self.mode in ('swing', 'paper_swing', 'options'):
            self._maybe_initial_data_update()

        self.window_size = 60
        self.feature_cols = get_feature_cols("swing")
        self.num_features = 11
        self.vision_channels = self.num_features # Same as features for 1D ResNet
        self.action_dim = 3 # Hold, Buy, Sell
        
        # 3. Initialize Brain (Agent)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Brain on {self.device}...", flush=True)
        self.agent = EnsembleAgent(
            time_series_dim=self.num_features,
            vision_channels=self.vision_channels,
            action_dim=self.action_dim,
            device=self.device
        )
        self.bearish_agent: EnsembleAgent | None = None
        
        # Load weights based on mode
        model_path = None
        if self.mode in ('swing', 'paper_swing'):
            model_path = str(SWING_MODEL_PATH)
        elif self.mode == 'options':
            model_path = (
                str(SWING_MODEL_PATH)
                if OPTIONS_USE_SWING_MODEL
                else str(OPTIONS_MODEL_PATH)
            )

        loaded = False
        if model_path:
            prefix = model_path
            if "_balanced.pth" in model_path:
                prefix = model_path.replace("_balanced.pth", "")
            elif "_aggressive.pth" in model_path:
                prefix = model_path.replace("_aggressive.pth", "")
            elif "_conservative.pth" in model_path:
                prefix = model_path.replace("_conservative.pth", "")
            
            target_file = f"{prefix}_balanced.pth"
            if os.path.exists(target_file):
                print(f"📂 Loading Configured Model: {prefix}...", flush=True)
                self.agent.load(prefix)
                self.agent.set_eval()
                loaded = True
            else:
                # STRICT MODE: If a specific model is configured but missing, DO NOT FALL BACK.
                # This prevents accidentally loading an older, inferior model (e.g. Ep200 instead of Ep350)
                path_hint = (
                    "SWING_MODEL_PATH"
                    if self.mode in ("swing", "paper_swing")
                    else "OPTIONS_MODEL_PATH / OPTIONS_USE_SWING_MODEL"
                )
                raise FileNotFoundError(
                    f"\n❌ CRITICAL: The configured model was not found!\n"
                    f"   Expected: {target_file}\n"
                    f"   Action: Aborting startup to prevent loading wrong model.\n"
                    f"   Fix: Check {path_hint} in config/settings.py"
                )

        if self.mode == "options" and getattr(self.config, "ENABLE_BEARISH_OPENS", False):
            bear_path = str(OPTIONS_BEARISH_MODEL_PATH)
            bprefix = bear_path
            for suffix in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
                if suffix in bear_path:
                    bprefix = bear_path.replace(suffix, "")
                    break
            btarget = f"{bprefix}_balanced.pth"
            if os.path.exists(btarget):
                print(f"📂 Loading Bearish Options Model: {bprefix}...", flush=True)
                self.bearish_agent = EnsembleAgent(
                    time_series_dim=self.num_features,
                    vision_channels=self.vision_channels,
                    action_dim=self.action_dim,
                    device=self.device,
                )
                self.bearish_agent.load(bprefix)
                self.bearish_agent.set_eval()
            else:
                print(
                    f"   ⚠️ ENABLE_BEARISH_OPENS=True but bearish model missing ({btarget}). "
                    "Bearish opens disabled.",
                    flush=True,
                )
        
        if not loaded:
            # Fallback to auto-discovery ONLY if no model path was configured
            print("⚠️ No model path configured in settings. Attempting auto-discovery...")
            import glob
            checkpoints = glob.glob("models/ensemble_ep*_balanced.pth")
            if checkpoints:
                # Find latest
                ep_nums = []
                for cp in checkpoints:
                    try:
                        num = int(cp.split("ensemble_ep")[1].split("_")[0])
                        ep_nums.append(num)
                    except: pass
                
                if ep_nums:
                    latest_ep = max(ep_nums)
                    print(f"📂 Loading Ensemble Model (Episode {latest_ep})...")
                    self.agent.load(f"models/ensemble_ep{latest_ep}")
                    self.agent.set_eval()
                else:
                    print("⚠️ No ensemble checkpoints found. Starting fresh.")
            else:
                print("⚠️ No pre-trained model found. Starting fresh.")

        # 4. Initialize Eyes & Ears (Data Pipeline)
        print("📡 Initializing data pipeline (news cache + tokenizer)...", flush=True)
        feed = getattr(self.config, 'DATA_FEED', 'sip')
        self.pipeline = MultiModalDataPipeline(window_size=self.window_size, creds=self.creds, feed=feed)
        print("📡 Data pipeline ready.", flush=True)
        
        self.last_snapshot_time = None # For rate-limiting daily snapshot refresh in swing mode

        # 6. Local State Cache (To avoid API Rate Limits)
        self.cached_positions = []
        self.cached_cash = 0.0
        self.cached_buying_power = 0.0
        self.cached_equity = 0.0
        self.last_cache_update = datetime.now()

        # 7. Trailing take-profit state (swing/options).
        self.position_peaks = {}
        self.position_entry_day = {}  # symbol -> date (swing min-hold / trail)
        self.options_broker: Optional[OptionsSpreadBroker] = None
        if self.mode == 'options':
            self.options_broker = OptionsSpreadBroker(
                str(self.creds.API_KEY),
                str(self.creds.API_SECRET),
                base_url=str(self._api_base_url),
                target_dte=int(getattr(self.config, 'TARGET_DTE', 30)),
                min_dte=int(getattr(self.config, 'MIN_DTE', 14)),
                max_dte=int(getattr(self.config, 'MAX_DTE', 45)),
                spread_width=float(getattr(self.config, 'SPREAD_WIDTH', 5.0)),
                limit_slippage_pct=float(getattr(self.config, 'LIMIT_SLIPPAGE_PCT', 0.08)),
                scale_width_by_price=bool(getattr(self.config, 'SCALE_WIDTH_BY_PRICE', True)),
                min_open_interest=int(getattr(self.config, 'MIN_OPEN_INTEREST', 10)),
                max_contracts_per_slot=int(getattr(self.config, 'MAX_CONTRACTS_PER_SLOT', 10)),
                close_use_market=bool(getattr(self.config, 'CLOSE_USE_MARKET', True)),
            )
            wl = getattr(self.config, "WATCHLIST", "options_liquid_200.txt")
            print(
                f"   Options: multi-strategy spreads | watchlist={wl} | "
                f"slots={self.config.MAX_POSITIONS} | buy>={self.config.CONFIDENCE_THRESHOLD} | "
                f"model={os.path.basename(model_path) if model_path else 'none'}",
                flush=True,
            )

        # Alpaca asset metadata cache (fractionable flag for order routing).
        self._fractionable_cache: dict[str, bool] = {}

        # Pre-open scan cache (snapshots + per-symbol features); used when PREMARKET_ONLY_FETCH.
        self._premarket_scan_cache: list | None = None
        self._premarket_scan_day: date | None = None
        
        print(f"✅ Bot Ready. Watching {len(self.symbols)} symbols.")

    def _preopen_lead_seconds(self) -> int:
        return int(getattr(self.config, 'WAKE_BEFORE_OPEN_MINUTES', 30) * 60)

    def _maybe_initial_data_update(self):
        """On startup: refresh data if market is open or we're already in the pre-open window."""
        if getattr(self.config, 'SKIP_DATA_UPDATE_ON_START', False) or self._skip_data_update:
            print("⏭️ Skipping data update on startup.", flush=True)
            return
        try:
            clock = self.api.get_clock()
            if clock.is_open:
                if self.mode in ("swing", "paper_swing", "options"):
                    premarket_only = getattr(self.config, "PREMARKET_ONLY_FETCH", False)
                    if premarket_only:
                        # Mid-session restart: use disk CSV + OCC/marks; no bulk re-download.
                        if not self._premarket_cache_valid():
                            print(
                                "⏳ Market open — building scan cache from disk "
                                "(skipping OCC/CSV re-download)...",
                                flush=True,
                            )
                            self._prefetch_scan_pipeline()
                    else:
                        self._update_historical_data()
                        if self.mode == "options":
                            self._update_options_bars()
                        if not self._premarket_cache_valid():
                            print(
                                "⏳ Market open — building scan cache (missed pre-open window)...",
                                flush=True,
                            )
                            self._prefetch_scan_pipeline()
                return
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            if time_to_open <= self._preopen_lead_seconds():
                print("⏳ Within pre-open window — running full pre-market prep...", flush=True)
                self._run_premarket_routine()
            else:
                mins = self._preopen_lead_seconds() // 60
                print(
                    f"⏳ Market closed — downloads/fetch deferred until {mins} min before open.",
                    flush=True,
                )
        except Exception as e:
            print(f"⚠️ Clock check failed ({e}); running data update anyway.", flush=True)
            self._update_historical_data()

    def _update_historical_data(self):
        """Runs the external data download script to update local CSVs."""
        if getattr(self.config, 'SKIP_DATA_UPDATE_ON_START', False) or self._skip_data_update:
            print("⏭️ Skipping data update.", flush=True)
            return
        print("🔄 Checking/Updating Historical Data...", flush=True)
        try:
            workers = int(getattr(self.config, 'PREMARKET_DOWNLOAD_WORKERS', 4))
            cmd = [
                sys.executable, "-u", "scripts/download_data.py",
                "--max-age-hours", str(getattr(self.config, 'DATA_MAX_AGE_HOURS', 20)),
                "--workers", str(workers),
            ]
            if self.watchlist_path:
                cmd.extend(["--watchlist", self.watchlist_path])
            subprocess.check_call(cmd)
            print("✅ Swing CSV update complete.", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to update data: {e}. Continuing with existing data...", flush=True)

    def _update_options_bars(self):
        """Incremental Alpaca option leg + spread marks (watchlist from config)."""
        if not getattr(self.config, "PREMARKET_OPTIONS_BARS_UPDATE", True):
            return
        wl = self.watchlist_path
        if not wl:
            wl = os.path.join(
                "config", "watchlists",
                getattr(self.config, "WATCHLIST", "options_liquid_200.txt"),
            )
        if not os.path.isfile(wl):
            print(f"⚠️ Options watchlist missing: {wl}", flush=True)
            return
        print("🔄 Updating option OCC bars + spread marks (incremental)...", flush=True)
        try:
            subprocess.check_call([
                sys.executable, "-u", "scripts/download_options_bars.py",
                "--watchlist", wl,
            ])
            print("✅ Options marks update complete.", flush=True)
        except Exception as e:
            print(f"⚠️ Options bars update failed: {e}. Using cached marks...", flush=True)

    def _session_timeframe(self) -> str:
        return "1D"

    def _prefetch_live_snapshots(self) -> int:
        """Inject today's daily bars from Alpaca snapshots into the pipeline cache."""
        symbols = self._sanitize_symbols(self.symbols)
        chunk_size = 50
        chunks = [symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        self.pipeline.live_daily_candles = {}
        fetched = 0
        feed = getattr(self.config, "DATA_FEED", "iex")
        for i, chunk in enumerate(chunks):
            print(
                f"   ⏳ Snapshots {i + 1}/{len(chunks)} ({len(chunk)} syms, feed={feed})...",
                flush=True,
            )
            fetched += self._ingest_snapshot_chunk(chunk)
        if fetched > 0:
            self.last_snapshot_time = datetime.now()
            print(
                f"   📡 Live snapshots: {len(self.pipeline.live_daily_candles)} symbols",
                flush=True,
            )
        else:
            print("   ⚠️ No snapshots returned; using CSV only.", flush=True)
        return fetched

    def _prefetch_symbol_features(self, timeframe: str) -> list:
        """fetch_and_process for full watchlist; store for session scans."""
        workers = int(getattr(self.config, "PREMARKET_FETCH_WORKERS", 10))
        data_results: list = []

        def _task(symbol: str):
            try:
                data = self.pipeline.fetch_and_process(symbol, timeframe=timeframe)
                if data is None:
                    return None
                ts_state, text_ids, text_mask, current_price = data
                ts_state = ts_state.reshape(self.window_size, self.num_features)
                return (symbol, ts_state, text_ids, text_mask, current_price)
            except Exception:
                return None

        total = len(self.symbols)
        print(f"   ⏳ Pre-loading features for {total} symbols ({workers} workers)...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_task, sym): sym for sym in self.symbols}
            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"   ⏳ Pre-load {done}/{total}...", flush=True)
                res = fut.result()
                if res:
                    data_results.append(res)
        print(
            f"   ✅ Pre-load done in {time.time() - t0:.1f}s "
            f"({len(data_results)}/{total} valid)",
            flush=True,
        )
        return data_results

    def _prefetch_scan_pipeline(self) -> None:
        """All network I/O for the daily scan: snapshots + per-symbol features."""
        if self.mode not in ("swing", "paper_swing", "options"):
            return
        tf = self._session_timeframe()
        print(f"🌅 Pre-market data pipeline ({tf}, {len(self.symbols)} symbols)...", flush=True)
        self._prefetch_live_snapshots()
        self._premarket_scan_cache = self._prefetch_symbol_features(tf)
        self._premarket_scan_day = datetime.now().date()

    def _premarket_cache_valid(self) -> bool:
        if not self._premarket_scan_cache or not self._premarket_scan_day:
            return False
        return self._premarket_scan_day == datetime.now().date()

    def _run_premarket_routine(self):
        """Refresh CSVs, option marks, and scan cache before the session opens."""
        self._update_historical_data()
        if self.mode == "options":
            self._update_options_bars()
        self._prefetch_scan_pipeline()

    def _sleep_until_market_open(self):
        """When closed: sleep until pre-open window, run prep, then sleep until open."""
        pre_open_sec = self._preopen_lead_seconds()
        while True:
            clock = self.api.get_clock()
            if clock.is_open:
                return
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            if time_to_open <= 0:
                time.sleep(5)
                continue

            if time_to_open > pre_open_sec:
                sleep_sec = time_to_open - pre_open_sec
                prep = "pre-open prep (CSV + options marks + snapshots + features)" if self.mode == "options" else "pre-open prep (CSV + snapshots + features)"
                print(
                    f"🌙 Market closed. Sleeping {sleep_sec / 3600:.2f}h until "
                    f"{pre_open_sec // 60} min before open ({prep})...",
                    flush=True,
                )
                time.sleep(sleep_sec)
                continue

            print(f"🌅 Pre-market window ({pre_open_sec // 60} min before open): preparing...", flush=True)
            self._run_premarket_routine()

            clock = self.api.get_clock()
            if clock.is_open:
                return
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            if time_to_open > 45:
                print(
                    f"✅ Pre-market prep done. Sleeping {time_to_open / 60:.1f} min until market open...",
                    flush=True,
                )
                time.sleep(max(30, time_to_open - 30))
            else:
                time.sleep(max(5, time_to_open))
            return

    def _spendable_cash(self) -> float:
        """Cash available for new buys without touching margin."""
        cash = max(0.0, float(self.cached_cash))
        if not getattr(self.config, 'CASH_ONLY', True):
            return max(0.0, float(self.cached_buying_power))
        return cash

    def refresh_account_state(self):
        """Pre-fetch account and positions to cache for execution loop."""
        try:
            # 1. Get Positions
            self.cached_positions = self.api.list_positions()
            
            # 2. Get Cash & Buying Power
            account = self.api.get_account()
            self.cached_cash = float(account.cash)
            self.cached_buying_power = float(account.buying_power)
            self.cached_equity = float(account.equity)
            
            # Print Summary
            spendable = self._spendable_cash()
            cash_only = getattr(self.config, 'CASH_ONLY', True)
            mode_tag = "cash-only" if cash_only else "margin OK"
            print(f"\n💰 Equity: ${self.cached_equity:.2f} | Cash: ${self.cached_cash:.2f} | "
                  f"BP: ${self.cached_buying_power:.2f} | Spendable ({mode_tag}): ${spendable:.2f}")
            if cash_only and self.cached_cash < 0:
                print("   ⚠️ MARGIN IN USE (cash < 0). New BUYs are blocked until cash is positive.")
            if self.cached_positions:
                print(f"📦 Holding {len(self.cached_positions)} positions:")
                for p in self.cached_positions:
                    print(f"   • {p.symbol}: {p.qty}")
            else:
                print("📦 No open positions.")
                
        except Exception as e:
            print(f"⚠️ Error refreshing account state: {e}")

    
    def _load_watchlist(self):
        path = self.watchlist_path
        
        # If no path provided, try default locations based on mode
        if not path:
            default_list = getattr(self.config, "WATCHLIST", "swing_liquid.txt")
            path = os.path.join("config", "watchlists", default_list)

        # Fallback to root if not found
        if not os.path.exists(path):
            print(f"⚠️ Watchlist {path} not found. Checking root...")
            path = os.path.basename(path)
            
        if os.path.exists(path):
            try:
                self.watchlist_path = path
                with open(path, encoding="utf-8") as f:
                    raw = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
                symbols = self._sanitize_symbols(raw)
                print(f"📋 Loaded {len(symbols)} symbols from {path}")
                return symbols
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        
        print(f"⚠️ Watchlist not found or empty. Using default.")
        return ["AAPL", "TSLA", "NVDA", "AMD", "MSFT"]

    @staticmethod
    def _sanitize_symbols(symbols: List[str]) -> List[str]:
        """Uppercase, dedupe, drop blanks/comments (safe for Alpaca batch APIs)."""
        out: List[str] = []
        seen = set()
        for raw in symbols:
            s = str(raw).strip().upper()
            if not s or s.startswith("#"):
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _fetch_snapshots_batch(self, symbols: List[str]) -> Dict:
        """Batch snapshots via query params (SDK embeds symbols in path and can 400)."""
        from alpaca_trade_api.entity_v2 import SnapshotsV2

        clean = self._sanitize_symbols(symbols)
        if not clean:
            return {}
        feed = getattr(self.config, "DATA_FEED", None) or "iex"
        resp = self.api.data_get(
            "/stocks/snapshots",
            data={"symbols": ",".join(clean), "feed": feed},
            api_version="v2",
        )
        return self.api.response_wrapper(resp, SnapshotsV2)

    def _ingest_snapshot_chunk(self, chunk: List[str], depth: int = 0) -> int:
        """Fetch one chunk; on failure split in half and retry (isolates bad tickers / URL limits)."""
        chunk = self._sanitize_symbols(chunk)
        if not chunk:
            return 0
        try:
            snapshots = self._fetch_snapshots_batch(chunk)
            n = 0
            for sym, snap in snapshots.items():
                if snap and getattr(snap, "daily_bar", None):
                    self.pipeline.live_daily_candles[sym] = {
                        "date": snap.daily_bar.t,
                        "open": snap.daily_bar.o,
                        "high": snap.daily_bar.h,
                        "low": snap.daily_bar.l,
                        "close": snap.daily_bar.c,
                        "volume": snap.daily_bar.v,
                    }
                    n += 1
            time.sleep(0.1)
            return n
        except Exception as e:
            if len(chunk) <= 1:
                print(f"   ⚠️ Snapshot skip {chunk[0]}: {e}")
                return 0
            if depth >= 5:
                print(f"   ⚠️ Chunk fetch failed ({len(chunk)} symbols): {e}")
                return 0
            mid = len(chunk) // 2
            return self._ingest_snapshot_chunk(chunk[:mid], depth + 1) + self._ingest_snapshot_chunk(
                chunk[mid:], depth + 1
            )

    def _calc_buy_alloc(self) -> float:
        """Dollar amount for one new position (equal-weight, cash-only).

        Matches backtest_swing_portfolio.py: each slot gets equity/MAX_POSITIONS,
        never cash/slots_remaining (which over-concentrates when only a few buys fill).
        """
        spendable = self._spendable_cash()
        if spendable < 10.0:
            return 0.0
        equity = max(float(self.cached_equity), spendable)
        max_pos = max(1, int(self.config.MAX_POSITIONS))
        buffer = float(getattr(self.config, 'CASH_BUFFER_PCT', 0.02))
        size_pct = float(getattr(self.config, 'POSITION_SIZE_PCT', 1.0))
        per_slot = (equity * (1.0 - buffer) / max_pos) * size_pct
        return min(per_slot, spendable)

    def _get_live_price(self, symbol: str) -> Optional[float]:
        """Latest trade price from Alpaca (authoritative for order sizing)."""
        try:
            snap = self.api.get_snapshot(symbol)
            if snap:
                if getattr(snap, 'latest_trade', None) and snap.latest_trade.p:
                    return float(snap.latest_trade.p)
                if getattr(snap, 'daily_bar', None) and snap.daily_bar.c:
                    return float(snap.daily_bar.c)
        except Exception:
            pass
        try:
            trade = self.api.get_latest_trade(symbol)
            if trade and getattr(trade, 'price', None):
                return float(trade.price)
        except Exception:
            pass
        return None

    def _batch_score_symbols(self, data_results, *, agent: EnsembleAgent | None = None):
        """GPU batch inference (same batch_act path as portfolio backtest)."""
        if not data_results:
            return []
        infer_agent = agent or self.agent
        chunk_size = int(getattr(self.config, "INFER_BATCH_SIZE", 128))
        scored = []
        infer_start = time.time()

        for i in range(0, len(data_results), chunk_size):
            chunk = data_results[i : i + chunk_size]
            ts_batch = torch.stack([r[1] for r in chunk]).to(self.device)
            ids_batch = torch.stack([r[2] for r in chunk]).to(self.device)
            mask_batch = torch.stack([r[3] for r in chunk]).to(self.device)

            with torch.inference_mode():
                actions, conf = infer_agent.batch_act(
                    ts_batch, ids_batch, mask_batch, return_confidence=True,
                    confidence_temperature=float(
                        getattr(self.config, "CONFIDENCE_TEMPERATURE", 0.01)
                    ),
                )
            actions_np = actions.detach().cpu().numpy()
            conf_np = conf.detach().cpu().numpy()

            for j, (symbol, _, _, _, price) in enumerate(chunk):
                scored.append({
                    "symbol": symbol,
                    "action": int(actions_np[j]),
                    "confidence": float(conf_np[j]),
                    "price": price,
                })

        print(
            f"   Inference done in {time.time() - infer_start:.1f}s "
            f"({len(scored)} symbols, batch={chunk_size})",
            flush=True,
        )
        return scored

    def _is_fractionable(self, symbol: str) -> bool:
        """Whether Alpaca accepts notional/fractional qty for this symbol."""
        if symbol in self._fractionable_cache:
            return self._fractionable_cache[symbol]
        try:
            asset = self.api.get_asset(symbol)
            ok = bool(getattr(asset, 'fractionable', False))
        except Exception:
            ok = False
        self._fractionable_cache[symbol] = ok
        return ok

    def _swing_trail_params(self):
        return (
            getattr(self.config, 'TRAIL_ACTIVATION_PCT', 0.05),
            getattr(self.config, 'TRAIL_GIVEBACK_PCT', 0.03),
            getattr(self.config, 'MIN_HOLD_DAYS', 2),
        )

    def _options_underlyings_held(self) -> set[str]:
        if not self.options_broker:
            return set()
        held = self.options_broker.underlyings_with_open_spreads(self.cached_positions)
        return held | self.options_broker.underlyings_with_pending_opens()

    def _execute_options_trade(self, symbol, action, current_price, confidence=0.0):
        """Open/close option spreads from swing BUY/SELL signals."""
        if action == 0:
            print(f"   ⏸️ {symbol}: HOLD")
            return
        broker = self.options_broker
        if broker is None:
            return

        underlying = symbol.upper()
        held = self._options_underlyings_held()
        has_spread = underlying in held
        open_spreads = len(held)

        if action == 1:
            if has_spread:
                print(f"   ⚠️ {underlying}: already have option spread. Skipping.")
                return
            if open_spreads >= int(self.config.MAX_POSITIONS):
                print(
                    f"   ⚠️ Max option slots ({open_spreads}/{self.config.MAX_POSITIONS}). Skipping."
                )
                return
            budget = self._calc_buy_alloc()
            min_bp = float(getattr(self.config, "MIN_BUYING_POWER", 50.0))
            if budget < min_bp:
                print(f"   ⚠️ Insufficient buying power for spread slot (${budget:.2f}).")
                return "NO_BP"
            spot = self._get_live_price(underlying) or float(current_price or 0)
            if spot <= 0:
                print(f"   ⚠️ {underlying}: no live price for strike selection.")
                return
            bullish = getattr(
                self.config, "BULLISH_STRATEGIES", ("call_debit", "bull_put_credit", "long_call")
            )
            long_call_min = float(getattr(self.config, "LONG_CALL_MIN_CONFIDENCE", 0.70))
            broker.open_bullish_spread(
                underlying, spot, budget, bullish,
                confidence=confidence,
                long_call_min_confidence=long_call_min,
            )
            self.position_entry_day[underlying] = datetime.now().date()
            return

        if action == 2:
            if has_spread:
                broker.close_spread(underlying)
                self.position_peaks.pop(underlying, None)
                self.position_entry_day.pop(underlying, None)
                return True
            return

    def _execute_bearish_options_open(self, symbol, current_price, confidence=0.0):
        """Open bearish spread from the separate bearish model BUY signal."""
        broker = self.options_broker
        if broker is None:
            return
        underlying = symbol.upper()
        if underlying in self._options_underlyings_held():
            return
        if len(self._options_underlyings_held()) >= int(self.config.MAX_POSITIONS):
            return
        budget = self._calc_buy_alloc()
        if budget < float(getattr(self.config, "MIN_BUYING_POWER", 50.0)):
            return "NO_BP"
        spot = self._get_live_price(underlying) or float(current_price or 0)
        if spot <= 0:
            return
        bearish = getattr(self.config, "BEARISH_STRATEGIES", ("put_debit", "bear_call_credit"))
        broker.open_bearish_spread(underlying, spot, budget, bearish)
        self.position_entry_day[underlying] = datetime.now().date()

    def _swing_allow_agent_sell(self, symbol, avg_entry, current_price):
        """Defer soft agent SELLs on winners until trail or min-hold rules allow."""
        if self.mode not in ('swing', 'paper_swing', 'options') or not getattr(self.config, 'ENABLE_WINNER_TRAIL', False):
            return True
        if avg_entry <= 0 or current_price <= 0:
            return True
        pnl_pct = (current_price - avg_entry) / avg_entry
        peak = max(self.position_peaks.get(symbol, pnl_pct), pnl_pct)
        self.position_peaks[symbol] = peak
        act_pct, giveback_pct, min_hold = self._swing_trail_params()
        if peak >= act_pct:
            return pnl_pct <= peak - giveback_pct
        entry_day = self.position_entry_day.get(symbol)
        if entry_day is not None and pnl_pct > 0:
            days_held = (datetime.now().date() - entry_day).days
            if days_held < min_hold:
                return False
        return True

    def execute_trade(self, symbol, action, current_price, confidence=0.0):
        """
        Action Map: 0=Hold, 1=Buy, 2=Sell
        """
        if self.mode == 'options':
            return self._execute_options_trade(symbol, action, current_price, confidence)

        if action == 0:
            print(f"   ⏸️ {symbol}: HOLD")
            return

        try:
            # Use Cached Positions
            positions_dict = {p.symbol: p for p in self.cached_positions}
            has_position = symbol in positions_dict
            qty_held = float(positions_dict[symbol].qty) if has_position else 0.0
            
            my_positions = [
                p for p in self.cached_positions
                if getattr(p, 'asset_class', 'us_equity') == 'us_equity'
            ]
            tif = 'day'
            
            if action == 1: # BUY
                if has_position:
                    print(f"   ⚠️ {symbol}: Already holding {qty_held}. Skipping BUY.")
                    return
                
                # 1. Check Max Positions
                if len(my_positions) >= self.config.MAX_POSITIONS:
                    print(f"   ⚠️ Max positions reached ({len(my_positions)}/{self.config.MAX_POSITIONS}). Skipping BUY.")
                    return

                # 2. Equal-weight sizing: fixed equity/MAX_POSITIONS per slot (portfolio backtest parity).
                target_val = self._calc_buy_alloc()
                if target_val < 10.0:
                    print(f"   ⚠️ Insufficient cash for equal-weight slot (${target_val:.2f}). Skipping BUY.")
                    return 'NO_BP'

                spendable = self._spendable_cash()
                notional = round(min(target_val, spendable), 2)
                if notional < 10.0:
                    print(f"   ⚠️ Insufficient cash for equal-weight slot (${notional:.2f}). Skipping BUY.")
                    return 'NO_BP'

                live_px = self._get_live_price(symbol)
                signal_px = float(current_price) if current_price else 0.0
                if live_px and signal_px > 0:
                    drift = abs(live_px - signal_px) / live_px
                    if drift > 0.10:
                        print(f"   ⚠️ {symbol}: CSV/signal ${signal_px:.2f} vs Alpaca live ${live_px:.2f} "
                              f"({drift*100:.0f}% off) — sizing from live price only")
                elif not live_px:
                    print(f"   ⚠️ {symbol}: No live Alpaca quote yet — will use notional if fractionable.")

                eq = max(float(self.cached_equity), 1.0)
                fractionable = self._is_fractionable(symbol)

                if fractionable:
                    disp_px = live_px if live_px else signal_px
                    print(f"   🚀 {symbol}: BUY ${notional:.2f} notional "
                          f"(slot target, ~{100*notional/eq:.1f}% of equity, "
                          f"live=${(disp_px or 0):.2f})")
                    self.api.submit_order(
                        symbol=symbol,
                        notional=notional,
                        side='buy',
                        type='market',
                        time_in_force=tif,
                    )
                    est_cost = notional
                    px_est = live_px or signal_px
                    est_qty = (notional / px_est) if px_est else 0
                else:
                    # Whole-share symbols: never size from CSV/signal (QRVO-at-$4.89 bug).
                    if not live_px or live_px <= 0:
                        print(f"   ⚠️ {symbol}: Not fractionable — need live Alpaca quote to size "
                              f"shares; skipping BUY.")
                        return
                    if signal_px > 0:
                        drift = abs(live_px - signal_px) / live_px
                        if drift > 0.10:
                            print(f"   ⚠️ {symbol}: CSV/signal ${signal_px:.2f} vs live ${live_px:.2f} "
                                  f"({drift*100:.0f}% off) — skipping whole-share BUY (stale CSV).")
                            return
                    qty = int(notional // live_px)
                    if qty < 1:
                        print(f"   ⚠️ {symbol}: Not fractionable; slot ${notional:.2f} "
                              f"can't buy 1 share @ live ${live_px:.2f}. Skipping.")
                        return
                    max_qty = int(spendable // live_px)
                    qty = min(qty, max_qty)
                    if qty < 1:
                        print(f"   ⚠️ Insufficient cash for 1 whole share of {symbol} @ live ${live_px:.2f}.")
                        return 'NO_BP'
                    est_cost = qty * live_px
                    print(f"   🚀 {symbol}: BUY {qty} shares (~${est_cost:.2f}, "
                          f"not fractionable, live=${live_px:.2f})")
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force=tif,
                    )
                    est_qty = float(qty)

                self.cached_cash -= est_cost
                class MockPos:
                    def __init__(self, s, q, ac):
                        self.symbol = s
                        self.qty = q
                        self.asset_class = ac
                self.cached_positions.append(MockPos(symbol, max(est_qty, 0.0001), 'us_equity'))
                if self.mode in ('swing', 'paper_swing'):
                    self.position_entry_day[symbol] = datetime.now().date()
                    self.position_peaks[symbol] = 0.0

            elif action == 2:  # SELL
                if not has_position:
                    return

                print(f"   🔻 {symbol}: SELL {qty_held} @ ${current_price:.2f}")
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty_held,
                    side='sell',
                    type='market',
                    time_in_force=tif
                )

                # Optimistic Update
                self.cached_cash += (qty_held * current_price)
                self.cached_positions = [p for p in self.cached_positions if p.symbol != symbol]
                self.position_peaks.pop(symbol, None)
                self.position_entry_day.pop(symbol, None)
                return True  # Success
                
        except Exception as e:
            error_msg = str(e).lower()
            print(f"   ❌ Order Failed: {e}")
            if 'pattern day trading' in error_msg or '40310100' in error_msg:
                print("   ℹ️  PDT protection: same-day sell blocked (Alpaca lifts this June 4). "
                      "Hold overnight or sell tomorrow.")
                return 'PDT_BLOCKED'
            if 'insufficient' in error_msg or 'buying power' in error_msg:
                return 'NO_BP'
            if 'not fractionable' in error_msg:
                self._fractionable_cache[symbol] = False
            return False


    def liquidate_all_positions(self):
        print(f"🚨 LIQUIDATING ALL POSITIONS ({self.mode.upper()} Mode) 🚨")
        try:
            if self.mode == 'options' and self.options_broker:
                for u in sorted(self._options_underlyings_held()):
                    self.options_broker.close_spread(u)
                self.position_peaks.clear()
                self.position_entry_day.clear()
                print("✅ Option spreads close orders submitted.")
                return

            positions = self.api.list_positions()
            for pos in positions:
                if pos.asset_class == 'us_equity':
                    print(f"   📉 Closing {pos.symbol} ({pos.qty})...")
                    self.api.submit_order(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side='sell',
                        type='market',
                        time_in_force='day',
                    )
            print("✅ All equity positions liquidated.")
            # Reset trailing take-profit state (we're flat now).
            self.position_peaks = {}
        except Exception as e:
            print(f"❌ Error liquidating positions: {e}")



    def _check_market_sentiment(self):
        """
        Checks broad market sentiment using SPY.
        Returns: 'fear', 'neutral', 'greed'
        """
        try:
            snap = self.api.get_snapshot("SPY")
            if not snap or not snap.daily_bar:
                return 'neutral'

            ref_price = (
                snap.prev_daily_bar.c
                if hasattr(snap, 'prev_daily_bar') and snap.prev_daily_bar
                else snap.daily_bar.o
            )
            current_price = snap.daily_bar.c
            pct_change = (current_price - ref_price) / ref_price * 100

            print(f"   Market benchmark (SPY): {pct_change:+.2f}% vs prior close")

            thresh = float(getattr(self.config, "SPY_FEAR_BLOCK_PCT", -1.0))
            if pct_change < thresh:
                return 'fear'
            return 'neutral'
        except Exception as e:
            print(f"   Sentiment check failed: {e}")
            return 'neutral'

    def run_loop(self):
        # Determine Timeframe
        timeframe = '1D'

        print(f"\n🚀 Starting Trading Loop in {self.mode.upper()} mode ({timeframe})...")
        if self.mode == "options":
            print(
                "   📌 OPTIONS: fine-tuned spread model → Alpaca CALL DEBIT SPREADS (mleg), "
                "not stock shares. BUY opens spread; SELL closes spread.",
                flush=True,
            )
        
        while True:
            try:
                clock = self.api.get_clock()
                if not clock.is_open:
                    self._sleep_until_market_open()
                    continue
            except Exception as e:
                print(f"⚠️ Error checking clock: {e}")
                time.sleep(60)
            
            # --- Refresh Account State (CACHE) ---
            self.refresh_account_state()

            print(f"\n⏰ Scan Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # --- OPTIONS: premium stop + expiry exit ---
            if self.mode == 'options' and self.options_broker and self.cached_positions:
                prem_stop = float(getattr(self.config, 'PREMIUM_STOP_PCT', 0.40))
                profit_target = float(getattr(self.config, 'PROFIT_TARGET_PCT', 0.0))
                min_dte_exit = int(getattr(self.config, 'MIN_DTE_EXIT', 5))
                held_spreads = self.options_broker.underlyings_with_open_spreads(
                    self.cached_positions
                )
                for u in sorted(held_spreads):
                    pnl = self.options_broker.spread_unrealized_pct(self.cached_positions, u)
                    dte = self.options_broker.days_to_expiry(self.cached_positions, u)
                    print(
                        f"   📉 {u} spread P/L: {pnl*100:+.1f}%"
                        + (f" (DTE={dte})" if dte is not None else ""),
                    )
                    reason = None
                    if pnl <= -prem_stop:
                        reason = f"PREMIUM STOP ({pnl*100:.1f}%)"
                    elif profit_target > 0 and pnl >= profit_target:
                        reason = f"PROFIT TARGET (+{pnl*100:.1f}%)"
                    elif dte is not None and dte <= min_dte_exit:
                        reason = f"EXPIRY WINDOW (DTE={dte})"
                    if reason:
                        print(f"   🚨 {reason} → close spread {u}")
                        self.execute_trade(u, 2, 0.0, confidence=1.0)

            # --- SWING: trailing take-profit on winners (hold longer) ---
            if self.mode in ('swing', 'paper_swing') and getattr(self.config, 'ENABLE_WINNER_TRAIL', False) and self.cached_positions:
                act_pct, giveback_pct, _ = self._swing_trail_params()
                for pos in self.cached_positions:
                    if pos.asset_class != 'us_equity':
                        continue
                    try:
                        avg_entry = float(pos.avg_entry_price)
                        current_price = float(pos.current_price)
                        if avg_entry <= 0:
                            continue
                        pnl_pct = (current_price - avg_entry) / avg_entry
                        peak = max(self.position_peaks.get(pos.symbol, pnl_pct), pnl_pct)
                        self.position_peaks[pos.symbol] = peak
                        if peak >= act_pct and pnl_pct <= peak - giveback_pct:
                            print(
                                f"   SWING TRAILING TP {pos.symbol}: peak +{peak*100:.2f}%, "
                                f"now +{pnl_pct*100:.2f}%"
                            )
                            if self.execute_trade(pos.symbol, 2, current_price, confidence=1.0):
                                self.position_peaks.pop(pos.symbol, None)
                                self.position_entry_day.pop(pos.symbol, None)
                    except Exception as e:
                        print(f"   ⚠️ Swing trail check failed for {pos.symbol}: {e}")

            # --- RANKING LOGIC START ---
            print(f"🔍 Scanning {len(self.symbols)} symbols for RANKING...")

            premarket_only = (
                self.mode in ("swing", "paper_swing", "options")
                and getattr(self.config, "PREMARKET_ONLY_FETCH", False)
            )
            use_premarket_cache = premarket_only and self._premarket_cache_valid()

            if premarket_only and not use_premarket_cache:
                print(
                    "   ⚠️ Pre-market cache missing/stale — running one-time fetch now "
                    "(restart before open to avoid this).",
                    flush=True,
                )
                self._prefetch_scan_pipeline()
                use_premarket_cache = self._premarket_cache_valid()

            if use_premarket_cache:
                print(
                    f"   ℹ️ Using pre-market cache ({len(self._premarket_scan_cache)} symbols, "
                    f"built {self._premarket_scan_day}) — no session download/fetch.",
                    flush=True,
                )
            elif self.mode in ("swing", "paper_swing", "options"):
                fetch_interval = 900
                should_fetch = True
                if self.last_snapshot_time:
                    time_since = (datetime.now() - self.last_snapshot_time).total_seconds()
                    if time_since < fetch_interval:
                        should_fetch = False
                        print(f"   ℹ️ Using cached snapshots ({int(time_since / 60)}m old).")
                if should_fetch:
                    try:
                        self._prefetch_live_snapshots()
                    except Exception as e:
                        print(f"   ⚠️ Snapshots fetch failed: {e}. Using cached CSV data only.")
            
            data_results: list = []
            if use_premarket_cache:
                data_results = list(self._premarket_scan_cache)
                print(
                    f"   ✅ Scan data from pre-market cache "
                    f"({len(data_results)} items).",
                    flush=True,
                )
            else:
                fetch_start = time.time()

                def fetch_symbol_task(symbol):
                    try:
                        data = self.pipeline.fetch_and_process(symbol, timeframe=timeframe)
                        if data is None:
                            return None
                        ts_state, text_ids, text_mask, current_price = data
                        ts_state = ts_state.reshape(self.window_size, self.num_features)
                        return (symbol, ts_state, text_ids, text_mask, current_price)
                    except Exception:
                        return None

                workers = 10
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_symbol = {
                        executor.submit(fetch_symbol_task, sym): sym for sym in self.symbols
                    }
                    completed_count = 0
                    total = len(self.symbols)
                    for future in as_completed(future_to_symbol):
                        completed_count += 1
                        if completed_count % 10 == 0:
                            print(f"   ⏳ Fetched {completed_count}/{total}...", end="\r")
                        res = future.result()
                        if res:
                            data_results.append(res)
                fetch_duration = time.time() - fetch_start
                print(
                    f"\n   ✅ Data Fetch Complete in {fetch_duration:.2f}s. "
                    f"(Found {len(data_results)} valid items)"
                )

            print("   Running batch inference...", flush=True)
            scored_opportunities = self._batch_score_symbols(data_results)

            if getattr(self.config, "VERBOSE_SCAN", False):
                for item in scored_opportunities[:50]:
                    act_str = ["HOLD", "BUY", "SELL"][item["action"]]
                    print(f"   {item['symbol']}: {act_str} (Conf: {item['confidence']:.3f})")

            # --- RANKING LOGIC END ---
            
            # Sort by Confidence (Descending)
            # We want high confidence BUYs first.
            # actions: 0=HOLD, 1=BUY, 2=SELL.
            # We separate them.
            
            # DEBUG: Log action distribution BEFORE filtering
            action_counts = {0: 0, 1: 0, 2: 0}
            for x in scored_opportunities:
                action_counts[x['action']] = action_counts.get(x['action'], 0) + 1
            if self.mode == "options":
                print(
                    f"   📊 Raw signals: HOLD={action_counts[0]}, "
                    f"BUY→spread={action_counts[1]}, SELL→close={action_counts[2]}"
                )
            else:
                print(f"   📊 Raw Actions: HOLD={action_counts[0]}, BUY={action_counts[1]}, SELL={action_counts[2]}")

            raw_buys = [x for x in scored_opportunities if x['action'] == 1]
            threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.60)

            passing_buys = [x for x in raw_buys if x['confidence'] > threshold]
            if self.mode == "options":
                label = "spread opens"
            else:
                label = "BUY candidates"
            if passing_buys:
                print(f"   🔍 Top {label} (conf > {threshold}):")
                held_preview = (
                    self._options_underlyings_held()
                    if self.mode == "options"
                    else {p.symbol for p in self.cached_positions}
                )
                for rb in sorted(passing_buys, key=lambda x: x['confidence'], reverse=True)[:5]:
                    tag = " [held]" if rb["symbol"] in held_preview else ""
                    print(f"      {rb['symbol']}: Conf={rb['confidence']:.3f}{tag}")
            elif raw_buys:
                top_raw = max(raw_buys, key=lambda x: x['confidence'])
                print(
                    f"   🔍 No {label} above {threshold} "
                    f"(best raw BUY: {top_raw['symbol']} @ {top_raw['confidence']:.3f})"
                )
                for rb in sorted(raw_buys, key=lambda x: x['confidence'], reverse=True)[:5]:
                    print(f"      (raw) {rb['symbol']}: {rb['confidence']:.3f}")
            
            buys = [x for x in scored_opportunities if x['action'] == 1 and x['confidence'] > threshold]
            
            # CRITICAL FIX: Filter out symbols we already hold BEFORE sniper mode
            if self.mode == 'options':
                held_symbols = self._options_underlyings_held()
            else:
                held_symbols = {p.symbol for p in self.cached_positions}
            buys_before_filter = len(buys)
            
            # Identify which ones are being dropped for logging
            dropped_symbols = [x['symbol'] for x in buys if x['symbol'] in held_symbols]
            buys = [x for x in buys if x['symbol'] not in held_symbols]
            
            if dropped_symbols:
                if self.mode == "options":
                    print(
                        f"   🔇 Skipped {len(dropped_symbols)} spread opens "
                        f"(already have spread): {', '.join(dropped_symbols)}"
                    )
                else:
                    print(
                        f"   🔇 Filtered out {len(dropped_symbols)} BUY signals "
                        f"for already-held positions: {', '.join(dropped_symbols)}"
                    )
            # CRITICAL: Only treat as SELL if confidence is HIGH (model is certain about exit)
            # Avoid selling on weak signals which are just neutral HOLDS
            sell_threshold = getattr(self.config, 'SELL_CONFIDENCE_THRESHOLD', threshold)
            sells = [x for x in scored_opportunities if x['action'] == 2 and x['confidence'] > sell_threshold]
            
            # Debug: Log filtered SELL count
            sells_raw = [x for x in scored_opportunities if x['action'] == 2]
            if sells_raw and len(sells) < len(sells_raw):
                filtered_out = len(sells_raw) - len(sells)
                print(f"   🔇 Filtered out {filtered_out} weak SELL signals (below {sell_threshold}).")
            
            # DEBUG: Check confidence distribution
            if scored_opportunities:
                confs = [x['confidence'] for x in scored_opportunities]
                print(f"   📊 Confidence Stats: Min={min(confs):.3f}, Max={max(confs):.3f}, Mean={np.mean(confs):.3f}")

            # Sort BUYs by confidence (Higher Q-value = Better)
            buys.sort(key=lambda x: x['confidence'], reverse=True)
            
            # --- SNIPER MODE: Hard Filter for Top K ---
            # Determine slots available (Approximation using cached state)
            if self.mode == 'options':
                current_holdings = len(self._options_underlyings_held())
            else:
                current_holdings = len([
                    p for p in self.cached_positions
                    if getattr(p, 'asset_class', 'us_equity') == 'us_equity'
                ])
            slots_left = max(0, self.config.MAX_POSITIONS - current_holdings)
            
            # DEBUG: Position state
            if self.mode == "options":
                print(
                    f"   📦 Open spreads: {current_holdings}/{self.config.MAX_POSITIONS} "
                    f"| Slots for new spreads: {slots_left}"
                )
            else:
                print(
                    f"   📦 Position State: {current_holdings}/{self.config.MAX_POSITIONS} "
                    f"| Slots Available: {slots_left}"
                )
            
            # Skip buys if spendable funds too low (cash for equity bots, buying power for options).
            spendable = self._spendable_cash()
            min_funds = float(getattr(self.config, "MIN_BUYING_POWER", 50.0))
            if spendable < min_funds:
                if self.mode == "options":
                    print(
                        f"   ⚠️ Insufficient buying power (${spendable:.2f}). "
                        "Skipping ALL BUY signals."
                    )
                else:
                    print(
                        f"   ⚠️ Insufficient cash on hand (${spendable:.2f}). "
                        "Skipping ALL BUY signals (cash-only)."
                    )
                buys = []
            
            # --- SPY regime filters (fear + above-VWAP trend) ---
            if getattr(self.config, "ENABLE_SPY_FEAR_FILTER", True):
                market_mood = self._check_market_sentiment()
                if market_mood == "fear":
                    thresh = getattr(self.config, "SPY_FEAR_BLOCK_PCT", -1.0)
                    print(
                        f"   SPY fear filter: blocking new buys "
                        f"(SPY day change < {thresh:.1f}%).",
                        flush=True,
                    )
                    buys = []

            if buys:
                if self.mode == "options":
                    print(f"   ✅ {len(buys)} spread open(s) queued (conf > {threshold:.2f})")
                else:
                    print(f"   BUY signals after filters: {len(buys)}")
            else:
                if self.mode == "options":
                    print(f"   NO spread opens this scan (need conf > {threshold:.2f})")
                else:
                    print(f"   NO BUY signals passed filters (need >{threshold:.2f})")

            # Filter buys list to match available slots (Sniper Mode)
            if buys and slots_left < len(buys):
                # Print info before cutting
                if slots_left > 0:
                     print(f"   ✂️ Sniper Mode: Limiting {len(buys)} candidates to Top {slots_left} available slots...")
                     buys = buys[:slots_left]
                else:
                     print(f"   ⚠️ Max Positions Full ({current_holdings}/{self.config.MAX_POSITIONS}). Ignoring all {len(buys)} buy signals.")
                     buys = []

            if buys:
                if self.mode == "options":
                    print(f"\n📊 BULLISH SPREAD CANDIDATES (ranked by confidence):")
                else:
                    print(f"\n📊 TOP RANKED BUY OPPORTUNITIES (Execution Optimization Active):")
                for i, item in enumerate(buys[:10]):
                     print(f"   {i+1}. {item['symbol']} (Conf: {item['confidence']:.3f})")
                print("-" * 30)
            
            # Execute SELLs first (free up cash), then refresh account before BUYs.
            for item in sells:
                sym = item['symbol']
                live_px = float(item['price'])
                if self.mode == 'options':
                    held_opts = self._options_underlyings_held()
                    if sym not in held_opts:
                        continue
                    allow_sell = self._swing_allow_agent_sell(sym, live_px * 0.99, live_px)
                else:
                    positions_dict = {p.symbol: p for p in self.cached_positions}
                    pos = positions_dict.get(sym)
                    if not pos:
                        continue
                    avg_entry = float(getattr(pos, 'avg_entry_price', 0) or 0)
                    allow_sell = self._swing_allow_agent_sell(sym, avg_entry, live_px)
                if not allow_sell:
                    print(f"   🔇 Deferring agent SELL for {item['symbol']} (min hold / trail)")
                    continue
                if self.mode == "options":
                    print(f"📉 CLOSE spread {item['symbol']} (Conf: {item['confidence']:.2f})")
                else:
                    print(f"📉 Executing SELL for {item['symbol']} (Conf: {item['confidence']:.2f})")
                self.execute_trade(item['symbol'], 2, item['price'], confidence=item['confidence'])

            if buys:
                self.refresh_account_state()
                per_slot = self._calc_buy_alloc()
                if self.mode == "options":
                    print(
                        f"   💵 Spread slot budget: ${per_slot:.2f} "
                        f"(equity ${self.cached_equity:.0f} / {self.config.MAX_POSITIONS}, "
                        f"BP ${self.cached_buying_power:.0f})"
                    )
                else:
                    print(
                        f"   💵 Equal-weight slot size: ${per_slot:.2f} "
                        f"(equity ${self.cached_equity:.0f} / {self.config.MAX_POSITIONS} slots)"
                    )

            min_funds = float(getattr(self.config, "MIN_BUYING_POWER", 50.0))
            for item in buys:
                if self._spendable_cash() < min_funds:
                    if self.mode == "options":
                        print("   🛑 Stopping BUYs — insufficient buying power.")
                    else:
                        print("   🛑 Stopping BUYs — insufficient cash on hand (cash-only).")
                    break
                if self.mode == "options":
                    print(f"📈 OPEN bullish spread on {item['symbol']} (Conf: {item['confidence']:.2f})")
                else:
                    print(f"📈 Executing BUY for {item['symbol']} (Conf: {item['confidence']:.2f})")
                result = self.execute_trade(item['symbol'], 1, item['price'], confidence=item['confidence'])
                if result == 'NO_BP':
                    if self.mode == "options":
                        print("   🛑 Stopping BUYs — insufficient buying power.")
                    else:
                        print("   🛑 Stopping BUYs — insufficient cash on hand (cash-only).")
                    break

            if (
                self.mode == "options"
                and self.bearish_agent is not None
                and getattr(self.config, "ENABLE_BEARISH_OPENS", False)
            ):
                bear_thresh = float(getattr(self.config, "BEARISH_CONFIDENCE_THRESHOLD", 0.65))
                bear_scored = self._batch_score_symbols(data_results, agent=self.bearish_agent)
                held_opts = self._options_underlyings_held()
                bear_buys = [
                    x for x in bear_scored
                    if x["action"] == 1
                    and x["confidence"] > bear_thresh
                    and x["symbol"] not in held_opts
                ]
                bear_buys.sort(key=lambda x: x["confidence"], reverse=True)
                slots_left = max(0, self.config.MAX_POSITIONS - len(held_opts))
                if bear_buys and slots_left > 0:
                    print(f"\n📊 BEARISH SPREAD CANDIDATES (bearish model BUY > {bear_thresh}):")
                    for i, item in enumerate(bear_buys[: min(10, slots_left)]):
                        print(f"   {i+1}. {item['symbol']} (Conf: {item['confidence']:.3f})")
                    for item in bear_buys[:slots_left]:
                        if self._spendable_cash() < min_funds:
                            print("   🛑 Stopping bearish opens — insufficient buying power.")
                            break
                        print(f"📉 OPEN bearish spread on {item['symbol']} (Conf: {item['confidence']:.2f})")
                        result = self._execute_bearish_options_open(
                            item['symbol'], item['price'], confidence=item['confidence'],
                        )
                        if result == 'NO_BP':
                            print("   🛑 Stopping bearish opens — insufficient buying power.")
                            break

            if buys:
                # Sync positions/cash from Alpaca after fills (notional qty unknown until fill).
                try:
                    time.sleep(1.5)
                    self.refresh_account_state()
                except Exception:
                    pass

            # Sleep
            sleep_secs = getattr(self.config, 'SCAN_INTERVAL_SECONDS', None)
            if sleep_secs is None:
                sleep_secs = int(getattr(self.config, 'SCAN_INTERVAL_MINUTES', 5)) * 60
            sleep_time = int(sleep_secs)
            print(f"💤 Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchlist", type=str, help="Path to watchlist file")
    parser.add_argument(
        "--mode",
        type=str,
        default="swing",
        choices=["swing", "paper_swing", "options"],
        help="Trading mode: swing (live), paper_swing, or options spreads",
    )
    parser.add_argument("--skip-data-update", action="store_true",
                        help="Skip yfinance CSV refresh on startup (use existing local data)")
    args = parser.parse_args()
    
    bot = MultiModalTrader(
        watchlist_path=args.watchlist,
        mode=args.mode,
        skip_data_update=args.skip_data_update,
    )
    bot.run_loop()

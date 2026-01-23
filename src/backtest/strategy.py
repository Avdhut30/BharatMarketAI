# src/backtest/strategy.py

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    horizon_days: int = 5
    p_up_entry: float = 0.55
    p_up_exit: float = 0.50
    use_trend_200: bool = True
    atr_stop_mult: float = 1.5
    round_trip_cost: float = 0.001  # 0.10% round trip

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False


@dataclass
class ARIMAConfig:
    name: str = "ARIMA"
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    use_auto_arima: bool = False


def fit_forecast_arima_statsmodels(
    y_hist: np.ndarray,
    horizon: int,
    cfg: ARIMAConfig,
) -> np.ndarray:
    model = SARIMAX(
        y_hist,
        order=cfg.order,
        seasonal_order=cfg.seasonal_order,
        trend="n",
        enforce_stationarity=cfg.enforce_stationarity,
        enforce_invertibility=cfg.enforce_invertibility,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=horizon)
    return np.asarray(fc, dtype=np.float64)


def fit_forecast_arima_auto(
    y_hist: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if not HAS_PMDARIMA:
        raise ImportError("pmdarima não está instalado, mas use_auto_arima=True foi solicitado.")

    model = pm.auto_arima(
        y_hist,
        seasonal=False,
        m=1,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        d=None,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        information_criterion="aic",
    )
    fc = model.predict(n_periods=horizon)
    return np.asarray(fc, dtype=np.float64)
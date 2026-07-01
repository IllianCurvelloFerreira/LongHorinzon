from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import warnings
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    pm = None
    HAS_PMDARIMA = False


@dataclass
class SARIMAConfig:
    name: str = "SARIMA"
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 0, 1, 24)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    use_auto_arima: bool = False

    # Limita o número de iterações do otimizador.
    # Isso evita que o SARIMA fique preso por muito tempo no Weather.
    maxiter: int = 25

    # Usa apenas os pontos mais recentes para ajustar o SARIMA.
    # Para Weather, isso reduz muito o tempo de ajuste.
    # Se quiser deixar ainda mais rápido, reduza para 3000 ou 2000.
    history_limit: int = 5000

    # Se o SARIMA falhar, retorna uma previsão sazonal simples
    # em vez de quebrar o experimento inteiro.
    use_fallback: bool = True


def _clean_series(y: np.ndarray) -> np.ndarray:
    """
    Limpa a série antes do ajuste.
    Remove valores não finitos e garante vetor 1D em float64.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        raise ValueError("Série vazia após limpeza.")

    return y


def _limit_history(
    y: np.ndarray,
    history_limit: Optional[int],
) -> np.ndarray:
    """
    Mantém apenas os últimos history_limit pontos da série.

    Isso é importante para Weather, pois o SARIMA com sazonalidade m=144
    fica muito pesado quando ajustado em mais de 40 mil observações.
    """
    if history_limit is None:
        return y

    if history_limit <= 0:
        return y

    if len(y) > history_limit:
        return y[-history_limit:]

    return y


def _pad_or_trim_forecast(
    fc: np.ndarray,
    horizon: int,
    fallback_value: float,
) -> np.ndarray:
    """
    Garante que a previsão tenha exatamente o tamanho do horizonte.
    """
    fc = np.asarray(fc, dtype=np.float64).reshape(-1)

    if len(fc) == 0:
        fc = np.repeat(fallback_value, horizon)

    if len(fc) < horizon:
        pad_value = fc[-1]
        fc = np.pad(
            fc,
            pad_width=(0, horizon - len(fc)),
            mode="constant",
            constant_values=pad_value,
        )

    if len(fc) > horizon:
        fc = fc[:horizon]

    return fc.astype(np.float64)


def _seasonal_naive_fallback(
    y_hist: np.ndarray,
    horizon: int,
    seasonal_order: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Fallback simples caso o SARIMA falhe.

    Se houver sazonalidade m, repete a última janela sazonal.
    Caso contrário, repete o último valor observado.
    """
    y_hist = _clean_series(y_hist)

    if seasonal_order is not None and len(seasonal_order) == 4:
        m = seasonal_order[3]
    else:
        m = 1

    if m is None or m <= 1 or len(y_hist) < m:
        return np.repeat(y_hist[-1], horizon).astype(np.float64)

    last_season = y_hist[-m:]
    reps = int(np.ceil(horizon / m))
    fc = np.tile(last_season, reps)[:horizon]

    return fc.astype(np.float64)


def fit_forecast_sarima_statsmodels(
    y_hist: np.ndarray,
    horizon: int,
    cfg: SARIMAConfig,
) -> np.ndarray:
    """
    Ajusta SARIMA via statsmodels e retorna a previsão.

    Versão revisada para bases grandes, principalmente Weather:
        - limpa a série;
        - limita o histórico usado no ajuste;
        - limita o número de iterações;
        - usa low_memory quando disponível;
        - usa simple_differencing/concentrate_scale quando disponível;
        - se falhar, retorna fallback seasonal naive.
    """
    y_hist = _clean_series(y_hist)
    y_fit = _limit_history(
        y=y_hist,
        history_limit=cfg.history_limit,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                model = SARIMAX(
                    y_fit,
                    order=cfg.order,
                    seasonal_order=cfg.seasonal_order,
                    trend="n",
                    enforce_stationarity=cfg.enforce_stationarity,
                    enforce_invertibility=cfg.enforce_invertibility,
                    simple_differencing=True,
                    concentrate_scale=True,
                )
            except TypeError:
                model = SARIMAX(
                    y_fit,
                    order=cfg.order,
                    seasonal_order=cfg.seasonal_order,
                    trend="n",
                    enforce_stationarity=cfg.enforce_stationarity,
                    enforce_invertibility=cfg.enforce_invertibility,
                )

            try:
                res = model.fit(
                    disp=False,
                    method="lbfgs",
                    maxiter=cfg.maxiter,
                    low_memory=True,
                )
            except TypeError:
                res = model.fit(
                    disp=False,
                    method="lbfgs",
                    maxiter=cfg.maxiter,
                )

            fc = res.forecast(steps=horizon)

            return _pad_or_trim_forecast(
                fc=fc,
                horizon=horizon,
                fallback_value=y_fit[-1],
            )

    except Exception as e:
        if not cfg.use_fallback:
            raise

        print(
            "[WARN] SARIMA falhou ou não convergiu. "
            "Usando fallback seasonal naive. "
            f"Erro: {type(e).__name__}: {e}"
        )

        return _seasonal_naive_fallback(
            y_hist=y_fit,
            horizon=horizon,
            seasonal_order=cfg.seasonal_order,
        )


def fit_forecast_sarima_auto(
    y_hist: np.ndarray,
    horizon: int,
    m: int,
) -> np.ndarray:
    """
    Auto-ARIMA sazonal.

    Atenção:
        Para Weather com m=144, auto_arima pode continuar muito lento.
        Use apenas se realmente precisar.
    """
    if not HAS_PMDARIMA:
        raise ImportError(
            "pmdarima não está instalado, mas use_auto_arima=True foi solicitado."
        )

    y_hist = _clean_series(y_hist)

    # Limita histórico também no auto_arima para evitar travamento.
    if len(y_hist) > 5000:
        y_fit = y_hist[-5000:]
    else:
        y_fit = y_hist

    model = pm.auto_arima(
        y_fit,
        seasonal=True,
        m=m,
        start_p=0,
        start_q=0,
        max_p=1,
        max_q=1,
        start_P=0,
        start_Q=0,
        max_P=1,
        max_Q=1,
        d=None,
        D=0,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        information_criterion="aic",
    )

    fc = model.predict(n_periods=horizon)

    return _pad_or_trim_forecast(
        fc=fc,
        horizon=horizon,
        fallback_value=y_fit[-1],
    )
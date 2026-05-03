from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def build_tree_model(args):
    """
    Retorna um modelo para prever diretamente múltiplos passos à frente.

    Entrada:
        X: [n_amostras, lookback * n_features]

    Saída:
        y: [n_amostras, horizon]
    """

    if args.model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    if args.model == "xgboost":
        base_model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

        return MultiOutputRegressor(base_model)

    raise ValueError("model deve ser 'random_forest' ou 'xgboost'.")
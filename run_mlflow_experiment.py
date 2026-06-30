"""v3 (MLOps): tracking de experimentos com MLflow.

Loga o experimento de frequência (GLM vs GBM) no MLflow local (./mlruns):
parâmetros, métrica (mean Poisson deviance) e o modelo. Depois abra a UI:

    python -m mlflow ui        # http://localhost:5000

Run:  python run_mlflow_experiment.py
"""
from __future__ import annotations

import pathlib
import sys

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.data import CATEGORICAL, NUMERIC, load_frequency  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402
from src.pricing.gbm import build_frequency_gbm  # noqa: E402


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("frequency_glm_vs_gbm")

    df = load_frequency()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    X = df[CATEGORICAL + NUMERIC].copy()
    X[CATEGORICAL] = OrdinalEncoder().fit_transform(df[CATEGORICAL])
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, df["Frequency"], df["Exposure"], test_size=0.2, random_state=42
    )

    experiments = [
        ("glm", build_frequency_pipeline, {"glm__sample_weight": wtr}),
        ("gbm", build_frequency_gbm, {"sample_weight": wtr}),
    ]
    for name, build, fit_kwargs in experiments:
        with mlflow.start_run(run_name=name):
            model = build().fit(Xtr, ytr, **fit_kwargs)
            dev = mean_poisson_deviance(yte, model.predict(Xte), sample_weight=wte)
            mlflow.log_param("model", name)
            mlflow.log_metric("mean_poisson_deviance", float(dev))
            try:
                mlflow.sklearn.log_model(model, name="model")
            except TypeError:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as exc:  # API varia entre versoes do mlflow
                print(f"  (modelo nao logado como artefato: {exc})")
            print(f"logado [{name}] mean_poisson_deviance={dev:.5f}")

    print("\nUI:  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()

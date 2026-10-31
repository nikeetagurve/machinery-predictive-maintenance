
"""Streaming inference and alert generation using the trained GRU model."""

from collections import defaultdict, deque
from pathlib import Path
import argparse
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.load_data import load_fd001


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_FILE = ARTIFACTS_DIR / "gru_model.pth"
PREPROCESSOR_FILE = ARTIFACTS_DIR / "gru_preprocessing.joblib"

DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "reports" / "streaming_predictions.csv"

MACHINE_COL = "unit_number"
CYCLE_COL = "time_cycles"


class GRUClassifier(nn.Module):
    """Same GRU architecture used during training."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.classifier(last_hidden).squeeze(1)


class StreamingPredictor:
    """Maintain engine-specific sensor history and generate alerts."""

    def __init__(self, model_path=MODEL_FILE, preprocessor_path=PREPROCESSOR_FILE):

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model configuration and weights.
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.feature_cols = list(checkpoint["feature_cols"])
        self.sequence_length = checkpoint["sequence_length"]
        self.threshold = checkpoint["prediction_threshold"]

        self.model = GRUClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load the training-fitted preprocessing objects.
        preprocessing = joblib.load(preprocessor_path)

        self.imputer = preprocessing["imputer"]
        self.scaler = preprocessing["scaler"]

        if list(preprocessing["feature_cols"]) != self.feature_cols:
            raise ValueError(
                "Model and preprocessing feature orders do not match."
            )

        # Each engine gets its own rolling sequence.
        self.engine_buffers = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )

        self.last_cycle = {}

        print(f"Model loaded successfully on {self.device}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Alert threshold: {self.threshold}")

    def predict(self, row: dict):
        """
        Process one incoming sensor row.

        Returns a prediction dictionary when enough history is available.
        Otherwise returns None.
        """

        engine_id = row[MACHINE_COL]
        cycle = row[CYCLE_COL]

        # Validate incoming features.
        missing = [
            col for col in self.feature_cols if col not in row
        ]

        if missing:
            raise ValueError(
                f"Missing sensor features: {missing}"
            )

        # Ensure cycles arrive consecutively for each engine.
        if engine_id in self.last_cycle:
            expected_cycle = self.last_cycle[engine_id] + 1

            if cycle != expected_cycle:
                raise ValueError(
                    f"Engine {engine_id}: expected cycle "
                    f"{expected_cycle}, received {cycle}. "
                    "Input must contain consecutive cycles."
                )

        self.last_cycle[engine_id] = cycle

        raw_features = pd.DataFrame(
            [[row[col] for col in self.feature_cols]],
            columns=self.feature_cols,
        )

        # Apply exactly the same fitted preprocessing as training.
        imputed = self.imputer.transform(raw_features)
        scaled = self.scaler.transform(imputed)

        self.engine_buffers[engine_id].append(
            scaled[0].astype(np.float32)
        )

        # Wait until 30 readings are available.
        if len(self.engine_buffers[engine_id]) < self.sequence_length:
            return None

        sequence = np.asarray(
            self.engine_buffers[engine_id],
            dtype=np.float32,
        )

        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()

        status = (
            "ALERT"
            if probability >= self.threshold
            else "NORMAL"
        )

        return {
            MACHINE_COL: engine_id,
            CYCLE_COL: cycle,
            "failure_probability": round(probability, 4),
            "status": status,
        }


def run_stream(
    data: pd.DataFrame,
    predictor: StreamingPredictor,
    delay: float = 0.0,
):
    """Simulate incoming sensor readings one row at a time."""

    results = []

    data = data.sort_values(
        [MACHINE_COL, CYCLE_COL]
    ).reset_index(drop=True)

    for _, row in data.iterrows():

        prediction = predictor.predict(row.to_dict())

        if prediction is None:
            print(
                f"Engine {row[MACHINE_COL]} | "
                f"Cycle {row[CYCLE_COL]} | "
                f"Collecting history "
                f"({len(predictor.engine_buffers[row[MACHINE_COL]])}"
                f"/{predictor.sequence_length})"
            )
        else:
            results.append(prediction)

            print(
                f"Engine: {prediction[MACHINE_COL]} | "
                f"Cycle: {prediction[CYCLE_COL]} | "
                f"Failure Probability: "
                f"{prediction['failure_probability']:.2%} | "
                f"Status: {prediction['status']}"
            )

        if delay > 0:
            time.sleep(delay)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="GRU streaming inference and alert generation."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV containing incoming sensor rows. "
             "If omitted, the FD001 test dataset is simulated.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save prediction results.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between incoming rows.",
    )

    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay cannot be negative.")

    predictor = StreamingPredictor()

    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {args.input}"
            )

        incoming_data = pd.read_csv(args.input)

        required_cols = (
            [MACHINE_COL, CYCLE_COL] + predictor.feature_cols
        )

        missing_cols = [
            col for col in required_cols
            if col not in incoming_data.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Input CSV is missing columns: {missing_cols}"
            )

    else:
        print("\nUsing NASA C-MAPSS FD001 test data as simulated stream.")

        _, incoming_data, _ = load_fd001()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = run_stream(
        incoming_data,
        predictor,
        delay=args.delay,
    )

    predictions.to_csv(
        args.output,
        index=False,
    )

    print("\nStreaming inference completed.")
    print(f"Total predictions: {len(predictions)}")

    if not predictions.empty:
        print(
            f"Total alerts: "
            f"{(predictions['status'] == 'ALERT').sum()}"
        )

    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()


"""Tests for streaming prediction inputs and engine history."""

from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.modeling.predict import StreamingPredictor


@pytest.fixture
def predictor():
    """Create a lightweight predictor without loading model files."""
    instance = StreamingPredictor.__new__(StreamingPredictor)

    instance.feature_cols = ["sensor_1", "sensor_2"]
    instance.sequence_length = 3
    instance.threshold = 0.5
    instance.device = torch.device("cpu")

    # Identity preprocessing for predictable test inputs.
    instance.imputer = SimpleNamespace(
        transform=lambda data: data.to_numpy(dtype=float)
    )
    instance.scaler = SimpleNamespace(
        transform=lambda data: np.asarray(data)
    )

    instance.engine_buffers = defaultdict(lambda: deque(maxlen=3))
    instance.last_cycle = {}

    return instance


def make_row(engine=1, cycle=1):
    return {
        "unit_number": engine,
        "time_cycles": cycle,
        "sensor_1": 10.0,
        "sensor_2": 20.0,
    }


def test_missing_sensor_feature_raises_error(predictor):
    row = make_row()
    del row["sensor_2"]

    with pytest.raises(ValueError, match="Missing sensor features"):
        predictor.predict(row)


def test_nonconsecutive_cycles_raise_error(predictor):
    assert predictor.predict(make_row(cycle=1)) is None

    with pytest.raises(ValueError, match="expected cycle 2"):
        predictor.predict(make_row(cycle=3))


def test_predict_waits_until_enough_history(predictor):
    assert predictor.predict(make_row(cycle=1)) is None
    assert predictor.predict(make_row(cycle=2)) is None

    assert len(predictor.engine_buffers[1]) == 2


class FakeModel:
    """Return a fixed model logit for predictable alert tests."""

    def __init__(self, logit):
        self.logit = logit

    def __call__(self, tensor):
        return torch.tensor([self.logit], dtype=torch.float32)


def test_high_failure_probability_triggers_alert(predictor):
    predictor.model = FakeModel(2.0)

    predictor.predict(make_row(cycle=1))
    predictor.predict(make_row(cycle=2))
    result = predictor.predict(make_row(cycle=3))

    assert result["status"] == "ALERT"
    assert result["failure_probability"] > predictor.threshold


def test_low_failure_probability_is_normal(predictor):
    predictor.model = FakeModel(-2.0)

    predictor.predict(make_row(cycle=1))
    predictor.predict(make_row(cycle=2))
    result = predictor.predict(make_row(cycle=3))

    assert result["status"] == "NORMAL"
    assert result["failure_probability"] < predictor.threshold


def test_probability_equal_to_threshold_triggers_alert(predictor):
    predictor.model = FakeModel(0.0)

    predictor.predict(make_row(cycle=1))
    predictor.predict(make_row(cycle=2))
    result = predictor.predict(make_row(cycle=3))

    assert result["failure_probability"] == 0.5
    assert result["status"] == "ALERT"

def test_engine_histories_are_independent(predictor):
    predictor.model = FakeModel(2.0)

    # Engine 1 receives two readings.
    assert predictor.predict(make_row(engine=1, cycle=1)) is None
    assert predictor.predict(make_row(engine=1, cycle=2)) is None

    # Engine 2 starts its own history.
    assert predictor.predict(make_row(engine=2, cycle=1)) is None

    assert len(predictor.engine_buffers[1]) == 2
    assert len(predictor.engine_buffers[2]) == 1

    # Engine 1 now has enough readings to predict.
    result = predictor.predict(make_row(engine=1, cycle=3))

    assert result["unit_number"] == 1
    assert result["status"] == "ALERT"

    # Engine 2 still has only one reading.
    assert len(predictor.engine_buffers[2]) == 1


def test_predictions_are_reproducible(predictor):
    predictor.model = FakeModel(2.0)

    rows = [make_row(cycle=i) for i in range(1, 4)]

    for row in rows:
        first_result = predictor.predict(row)

    # Reset streaming history and repeat the same inputs.
    predictor.engine_buffers.clear()
    predictor.last_cycle.clear()

    for row in rows:
        second_result = predictor.predict(row)

    assert first_result == second_result
    

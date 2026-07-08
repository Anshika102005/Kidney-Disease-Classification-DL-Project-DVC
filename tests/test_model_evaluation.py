from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cnnClassifier.components.model_evaluation_mlflow import Evaluation


class DummyModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape


def test_target_size_uses_model_input_shape():
    evaluation = Evaluation(SimpleNamespace(params_image_size=[160, 160, 3]))
    evaluation.model = DummyModel((None, 224, 224, 3))

    assert evaluation._get_target_size() == (224, 224)


def test_target_size_falls_back_to_config():
    evaluation = Evaluation(SimpleNamespace(params_image_size=[160, 160, 3]))

    assert evaluation._get_target_size() == (160, 160)

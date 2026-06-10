import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cnnClassifier.components.model_evaluation_mlflow import ModelEvaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        self.evaluator = ModelEvaluation()

    def main(self):
        self.evaluator.evaluate()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n x==========x")
    except Exception as e:
        logger.exception(e)
        raise e

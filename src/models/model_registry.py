from src.models.isolation_forest_model import IsolationForestDetector
from src.models.svm_model import OneClassSVMDetector
from src.models.statistical_models import ZScoreAnomalyDetector, MovingAverageAnomalyDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector

MODEL_REGISTRY = {
    "isolation_forest": IsolationForestDetector,
    "oneclass_svm": OneClassSVMDetector,
    "zscore": ZScoreAnomalyDetector,
    "mad": MovingAverageAnomalyDetector,
    "lstm_autoencoder": LSTMAutoencoderDetector
}

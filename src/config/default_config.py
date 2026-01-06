"""
Configuration file for S.A.F.E Dashboard
Centralized configuration for paths and settings
"""

from pathlib import Path

# ============= PATHS =============

# Base project directory
PROJECT_ROOT = Path("C:/Users/somas/PycharmProjects/S.A.F.E")

# Models directory (where your trained models are stored)
MODELS_PATH = PROJECT_ROOT / "scripts" / "models"

# Data directories
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Create directories if they don't exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)

# ============= API SETTINGS =============

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True  # Set to False in production

# API Base URL (for frontend to connect)
API_BASE_URL = f"http://localhost:{API_PORT}"

# ============= MODEL SETTINGS =============

# Model file extensions to look for
MODEL_EXTENSIONS = [".pkl", ".joblib", ".pth"]

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "zscore": {
        "threshold": 2.5,
        "window_size": None
    },
    "moving_average": {
        "window_size": 10,
        "threshold": 2.0
    },
    "threshold": {
        "percentile_based": True
    },
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": 0.1
    },
    "svm": {
        "kernel": "rbf",
        "nu": 0.1
    },
    "lstm": {
        "sequence_length": 20,
        "hidden_size": 32,
        "epochs": 30
    }
}

# ============= DASHBOARD SETTINGS =============

# Streamlit Configuration
STREAMLIT_PORT = 8501
STREAMLIT_THEME = "light"  # or "dark"

# Dashboard refresh settings
AUTO_REFRESH_ENABLED = False
AUTO_REFRESH_INTERVAL = 10  # seconds

# Chart settings
CHART_HEIGHT = 400
CHART_COLOR_SCHEME = "RdYlGn_r"  # Red-Yellow-Green reversed

# Alert settings
DEFAULT_ALERT_LIMIT = 20
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# ============= DATA SETTINGS =============

# Dataset configurations
DATASETS = {
    "training": ["eth", "hotel", "univ"],
    "validation": ["zara1", "zara2"],
    "test": ["mall"]
}

# Grid configuration
GRID_SIZE = (5, 5)
TIME_WINDOW = 5

# ============= FEATURE SETTINGS =============

# Feature engineering parameters
FEATURE_PARAMS = {
    "footfall": {
        "baseline_window": 50,
        "anomaly_threshold": 2.5,
        "surge_threshold": 1.5
    }
}

# ============= RISK SCORING =============

# Risk level thresholds
RISK_LEVELS = {
    "Low": (0.0, 0.4),
    "Medium": (0.4, 0.7),
    "High": (0.7, 1.0)
}

# Risk colors for visualization
RISK_COLORS = {
    "High": "#f44336",
    "Medium": "#ff9800",
    "Low": "#4caf50"
}

# ============= LOGGING =============

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = PROJECT_ROOT / "logs" / "safe.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# ============= CORS SETTINGS =============

# CORS allowed origins (for API)
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    f"http://localhost:{STREAMLIT_PORT}",
    "*"  # Allow all (remove in production)
]

# ============= WEBSOCKET SETTINGS =============

# WebSocket configuration
WS_ENABLED = True
WS_UPDATE_INTERVAL = 2  # seconds

# ============= MODEL METADATA =============

# Model information for display
MODEL_INFO = {
    "zscore": {
        "name": "Z-Score Detector",
        "type": "statistical",
        "description": "Statistical anomaly detection using Z-scores",
        "best_for": "Fast detection, interpretable results"
    },
    "moving_average": {
        "name": "Moving Average Detector",
        "type": "statistical",
        "description": "Time-series anomaly detection using moving averages",
        "best_for": "Temporal patterns, trend analysis"
    },
    "threshold": {
        "name": "Threshold Detector",
        "type": "statistical",
        "description": "Simple threshold-based anomaly detection",
        "best_for": "Rule-based detection, quick setup"
    },
    "isolation_forest": {
        "name": "Isolation Forest",
        "type": "machine_learning",
        "description": "Tree-based ensemble anomaly detection",
        "best_for": "Robust detection, handles noise well"
    },
    "svm": {
        "name": "One-Class SVM",
        "type": "machine_learning",
        "description": "Boundary-based anomaly detection using SVM",
        "best_for": "Clear decision boundaries, complex patterns"
    },
    "lstm": {
        "name": "LSTM Autoencoder",
        "type": "deep_learning",
        "description": "Sequential pattern learning for time-series",
        "best_for": "Complex temporal dependencies, trajectory analysis"
    },
    "ensemble": {
        "name": "Ensemble Model",
        "type": "ensemble",
        "description": "Combines multiple models for robust detection",
        "best_for": "Best overall performance, production use"
    }
}

# ============= HELPER FUNCTIONS =============

def get_model_path(model_id: str) -> Path:
    """Get the full path for a model file"""
    return MODELS_PATH / f"{model_id}.pkl"

def list_available_models():
    """List all available model files in the models directory"""
    available = []
    for ext in MODEL_EXTENSIONS:
        available.extend(MODELS_PATH.glob(f"*{ext}"))
    return [f.stem for f in available]

def get_risk_color(risk_score: float) -> str:
    """Get color based on risk score"""
    if risk_score >= HIGH_RISK_THRESHOLD:
        return RISK_COLORS["High"]
    elif risk_score >= MEDIUM_RISK_THRESHOLD:
        return RISK_COLORS["Medium"]
    else:
        return RISK_COLORS["Low"]

def get_risk_level(risk_score: float) -> str:
    """Get risk level based on score"""
    if risk_score >= HIGH_RISK_THRESHOLD:
        return "High"
    elif risk_score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "Low"

# ============= VALIDATION =============

def validate_config():
    """Validate configuration and paths"""
    errors = []

    # Check if models directory exists
    if not MODELS_PATH.exists():
        errors.append(f"Models directory not found: {MODELS_PATH}")

    # Check if data directory exists
    if not DATA_PATH.exists():
        errors.append(f"Data directory not found: {DATA_PATH}")

    # Check for at least one model file
    if not list_available_models():
        errors.append(f"No model files found in {MODELS_PATH}")

    return errors

# Print configuration on import
if __name__ == "__main__":
    print("=" * 60)
    print("S.A.F.E Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Path: {MODELS_PATH}")
    print(f"Data Path: {DATA_PATH}")
    print(f"API URL: {API_BASE_URL}")
    print(f"\nAvailable Models: {list_available_models()}")

    errors = validate_config()
    if errors:
        print("\n⚠️ Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ Configuration validated successfully!")
    print("=" * 60)
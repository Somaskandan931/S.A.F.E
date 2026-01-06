DEFAULT_CONFIG = {
    'data': {
        'raw_data_path': "data/raw/",
        'processed_data_path': "data/processed/",
        'synthetic_data_path': "data/synthetic/",
        'time_window': 60,
        'zone_count': 10
    },
    'features': {
        'temporal_features': ['hour_of_day', 'day_of_week', 'is_peak_hour'],
        'crowd_features': ['density', 'speed_mean', 'speed_variance', 
                         'direction_variance', 'entry_exit_ratio', 
                         'footfall_change_rate'],
        'footfall': {
            'baseline_window': 30,
            'anomaly_threshold': 2.5,
            'surge_threshold': 1.5
        }
    },
    'models': {
        'statistical': {
            'z_score_threshold': 3.0,
            'window_size': 10
        },
        'isolation_forest': {
            'n_estimators': 100,
            'contamination': 0.1,
            'max_samples': 256,
            'random_state': 42
        },
        'lstm_autoencoder': {
            'sequence_length': 30,
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'dropout': 0.2
        }
    },
    'risk_scoring': {
        'weights': {
            'footfall_anomaly': 0.4,
            'flow_disruption': 0.4,
            'temporal_abnormality': 0.2
        },
        'thresholds': {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        },
        'escalation': {
            'persistence_window': 3,
            'concurrent_anomalies_threshold': 2
        }
    },
    'system': {
        'random_seed': 42,
        'n_jobs': -1
    }
}

def get_config():
    """Get default configuration"""
    return DEFAULT_CONFIG.copy()
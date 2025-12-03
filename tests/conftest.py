"""
Test configuration and fixtures for GAM-SSM-LUR.
"""

import numpy as np
import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def synthetic_data():
    """Generate small synthetic dataset for testing."""
    np.random.seed(42)
    
    n_locations = 10
    n_times = 20
    n_features = 5
    
    X = np.random.randn(n_locations * n_times, n_features)
    y = X @ np.random.randn(n_features) + np.random.randn(n_locations * n_times) * 0.5
    time_index = np.repeat(np.arange(n_times), n_locations)
    location_index = np.tile(np.arange(n_locations), n_times)
    
    return {
        'X': X,
        'y': y,
        'time_index': time_index,
        'location_index': location_index,
        'n_locations': n_locations,
        'n_times': n_times,
        'n_features': n_features,
    }


@pytest.fixture
def simple_ssm_data():
    """Generate data from a simple state space model."""
    np.random.seed(42)
    
    T_len = 100
    state_dim = 3
    
    T = np.eye(state_dim) * 0.9
    Q = np.eye(state_dim) * 0.1
    H = np.eye(state_dim) * 0.2
    
    states = np.zeros((T_len, state_dim))
    observations = np.zeros((T_len, state_dim))
    
    for t in range(T_len):
        if t > 0:
            states[t] = T @ states[t-1] + np.random.multivariate_normal(np.zeros(state_dim), Q)
        observations[t] = states[t] + np.random.multivariate_normal(np.zeros(state_dim), H)
        
    return {
        'states': states,
        'observations': observations,
        'T': T,
        'Q': Q,
        'H': H,
        'state_dim': state_dim,
        'T_len': T_len,
    }

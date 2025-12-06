# Code Refactoring Plan - GAM-SSM-LUR

This document outlines the redundancies identified in the codebase and the refactoring strategy to eliminate them.

## ✅ Completed Work

### 1. Created Utility Module (`src/gam_ssm_lur/utils.py`)

Centralized common functions to eliminate redundancy across the codebase:

**Functions implemented:**
- `ensure_array()` - Convert pandas/array-like to numpy arrays (replaces 30+ duplicate conversions)
- `extract_feature_names()` - Extract feature names from DataFrame or use provided names (replaces 3+ duplicates)
- `compute_prediction_intervals()` - Calculate confidence intervals using z-scores (replaces 4+ duplicates)
- `extract_diagonal()` - Handle diagonal extraction from covariance matrices (replaces 8+ duplicates)
- `ensure_diagonal_matrix()` - Ensure covariance is in full matrix form
- `compute_aic()` - Calculate Akaike Information Criterion
- `compute_bic()` - Calculate Bayesian Information Criterion
- `compute_r_squared()` - Calculate R-squared with NaN handling
- `reshape_flat_to_matrix()` - Reshape flat arrays to (n_times, n_locations)
- `reshape_matrix_to_flat()` - Reshape matrices to flat arrays

### 2. Created Base Model Class (`src/gam_ssm_lur/base.py`)

**BaseEstimator class:**
- Provides `_check_fitted()` method (eliminates 3 duplicate implementations)
- Provides abstract `fit()` and `predict()` methods
- Provides generic `save()` and `load()` methods using pickle
- Provides `get_params()` and `set_params()` for sklearn compatibility
- Requires subclasses to implement `_get_state_dict()` and `_set_state_dict()`

**ModelSummary class:**
- Standardized container for model statistics (R², RMSE, MAE, AIC, BIC)
- Replaces 3+ custom summary implementations
- Provides `to_dict()` and `__repr__()` methods

### 3. Partially Refactored SpatialGAM

**Changes made:**
- Inherits from `BaseEstimator`
- Uses `extract_feature_names()` utility in `fit()` method
- Uses `ensure_array()` utility for type conversions
- Implemented `_get_state_dict()` and `_set_state_dict()` for serialization
- Removed duplicate `_check_fitted()` method (now inherited)

**Note:** The old `save()` and `load()` methods still exist and need to be removed once testing confirms the base class methods work correctly.

## 📋 Remaining Work

### Priority 1: Complete Model Refactoring

#### A. Complete SpatialGAM Refactoring
**File:** `src/gam_ssm_lur/spatial_gam.py`

**Tasks:**
1. Remove duplicate `save()` method (line 473-503) - now inherited from BaseEstimator
2. Remove duplicate `load()` classmethod (line 505-532) - now inherited from BaseEstimator
3. Update `predict()` method to use `ensure_array()` utility
4. Update `get_residuals()` method to use `ensure_array()` utility
5. Test that save/load works with new base class implementation

#### B. Refactor StateSpaceModel
**File:** `src/gam_ssm_lur/state_space.py`

**Tasks:**
1. Make class inherit from `BaseEstimator`
2. Remove duplicate `_check_fitted()` method (line 508-511)
3. Replace prediction interval calculations with `compute_prediction_intervals()` utility (lines 314-316, 364-366)
4. Use `extract_diagonal()` utility for covariance handling (lines 322-327)
5. Implement `_get_state_dict()` and `_set_state_dict()` methods
6. Remove duplicate `save()` and `load()` methods (lines 551-609)
7. Update `fit()` to use `ensure_array()` for type conversions (lines 220-227)

#### C. Refactor HybridGAMSSM
**File:** `src/gam_ssm_lur/hybrid_model.py`

**Tasks:**
1. Make class inherit from `BaseEstimator`
2. Remove duplicate `_check_fitted()` method (line 607-610)
3. Replace prediction interval calculations with `compute_prediction_intervals()` utility (lines 418-421, 469-472)
4. Use `extract_feature_names()` in `fit()` method (lines 224-226)
5. Use `ensure_array()` for type conversions (lines 223-235, 398-400, 453-455, 507-508)
6. Use `reshape_flat_to_matrix()` and `reshape_matrix_to_flat()` utilities (lines 283, 406, 415, 459-460)
7. Implement `_get_state_dict()` and `_set_state_dict()` methods
8. Test serialization/deserialization

### Priority 2: Update Other Files

#### D. Refactor EMEstimator
**File:** `src/gam_ssm_lur/em_estimator.py`

**Tasks:**
1. Use `extract_diagonal()` utility (lines 264-267, 284-287, 289-292, 377-379)
2. Use `ensure_diagonal_matrix()` utility where applicable

#### E. Refactor Kalman Filter
**File:** `src/gam_ssm_lur/kalman.py`

**Tasks:**
1. Use `extract_diagonal()` and `ensure_diagonal_matrix()` utilities (lines 503-512)
2. Consider extracting threshold constants to a config file (lines 162-163)

#### F. Refactor Features Module
**File:** `src/gam_ssm_lur/features.py`

**Tasks:**
1. Use `extract_feature_names()` utility in `fit()` method (lines 132-139)
2. Use `ensure_array()` for type conversions (lines 132-143)

### Priority 3: Update Examples and Tests

#### G. Update Example Scripts
**Files:** `examples/01_basic_usage.py`, `examples/reproduce_paper.py`

**Tasks:**
1. Extract common data validation logic to `examples/data_utils.py`
2. Use centralized column detection and parsing
3. Update imports to use new utilities where applicable

#### H. Update Tests
**Files:** `tests/test_hybrid_model.py`, `tests/test_spatial_gam.py`, `tests/test_state_space.py`

**Tasks:**
1. Add tests for new utility functions in `test_utils.py`
2. Add tests for BaseEstimator in `test_base.py`
3. Verify that save/load works with refactored models
4. Ensure all existing tests still pass

### Priority 4: Documentation and Configuration

#### I. Create Constants/Config Module
**File:** `src/gam_ssm_lur/config.py` (new)

**Tasks:**
1. Consolidate threshold constants (DENSE_THRESHOLD, DIAGONAL_THRESHOLD)
2. Consolidate default feature configurations (buffer distances, road types, land use types)
3. Update imports across the codebase

#### J. Update __init__.py
**File:** `src/gam_ssm_lur/__init__.py`

**Tasks:**
1. Export new `BaseEstimator` class
2. Export new `ModelSummary` class
3. Export commonly-used utilities from `utils` module
4. Update documentation strings

## 🎯 Expected Benefits

### Lines of Code Reduction
- **Estimated redundant code:** ~200-300 lines
- **After refactoring:** Reduced by 60-70%

### Maintainability Improvements
1. **Single source of truth** - Bug fixes in one place benefit all models
2. **Consistent behavior** - All models use same validation, conversion, and serialization logic
3. **Easier testing** - Utilities can be tested independently
4. **Better documentation** - Centralized functions are easier to document

### Code Quality Metrics
| Metric | Before | After (Est.) |
|--------|--------|--------------|
| Duplicate methods | 40+ | <5 |
| Copy-paste blocks | 15+ | 0 |
| Files with similar logic | 8 | 2 (utils, base) |

## ⚠️ Testing Strategy

### Phase 1: Unit Tests
1. Test all new utility functions independently
2. Test BaseEstimator save/load with mock subclass
3. Test ModelSummary formatting and conversions

### Phase 2: Integration Tests
1. Test each refactored model (SpatialGAM, StateSpaceModel, HybridGAMSSM)
2. Verify save/load preserves model state correctly
3. Verify predictions match before/after refactoring

### Phase 3: End-to-End Tests
1. Run full example scripts (`01_basic_usage.py`, `reproduce_paper.py`)
2. Compare output metrics with baseline
3. Verify no regressions in model performance

## 📊 Migration Guide

For users of the library, the refactoring should be **backward compatible**:

### No Breaking Changes
- All public APIs remain the same
- Model behavior is unchanged
- Serialization format may change, but old models can still be loaded

### Optional New Features
- Can now use `from gam_ssm_lur.utils import ensure_array` for custom workflows
- Can subclass `BaseEstimator` for custom models
- Can use `ModelSummary` for standardized reporting

## 🔄 Rollback Plan

If refactoring introduces bugs:

1. **Git tags:** Tag current working version before merging refactoring
2. **Feature flags:** Keep old and new implementations temporarily with runtime switch
3. **Gradual rollout:** Refactor one model at a time, test thoroughly before proceeding
4. **Parallel implementations:** Maintain `_legacy` versions during transition period

## 📅 Implementation Timeline

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 1 | Complete SpatialGAM refactoring | 2-3 hours |
| Phase 2 | Refactor StateSpaceModel | 2-3 hours |
| Phase 3 | Refactor HybridGAMSSM | 2-3 hours |
| Phase 4 | Update EMEstimator, Kalman, Features | 2-4 hours |
| Phase 5 | Update examples and documentation | 2-3 hours |
| Phase 6 | Comprehensive testing | 3-4 hours |
| **Total** | | **13-20 hours** |

## ✨ Future Enhancements

After completing basic refactoring, consider:

1. **Type hints:** Add comprehensive type hints using `Protocol` for duck typing
2. **Abstract base classes:** Define formal interfaces for Estimators, Predictors, etc.
3. **Logging utilities:** Centralize logging configuration and formatting
4. **Performance profiling:** Add decorators for timing critical functions
5. **Caching:** Add memoization for expensive computations
6. **Parallel processing:** Utilities for parallelizing predictions across locations

---

**Status:** In Progress
**Last Updated:** 2025-12-06
**Next Steps:** Complete SpatialGAM refactoring and create unit tests for utils module

"""
Tests for Module 1.2: PCA on Yield Curves.

Validates that PCA decomposition produces correct number of components,
high explained variance, and sensible loadings structure.
"""

import numpy as np
import pandas as pd
import pytest


class TestPCAResults:
    """Tests for the PCA decomposition of yield curve changes."""
    
    def test_all_countries_have_results(self, pca_results, synthetic_yield_changes):
        """Every country in the input should produce PCA results."""
        for country in synthetic_yield_changes:
            assert country in pca_results, \
                f"Missing PCA results for {country}"
    
    def test_correct_number_of_components(self, pca_results):
        """PCA should extract the requested number of components."""
        for country, res in pca_results.items():
            n_comp = res["n_components"]
            assert res["scores"].shape[1] == n_comp, \
                f"{country}: expected {n_comp} components in scores"
            assert len(res["explained_var"]) == n_comp, \
                f"{country}: expected {n_comp} explained variance values"
            assert res["loadings"].shape[0] == n_comp, \
                f"{country}: expected {n_comp} loading vectors"
    
    def test_explained_variance_sums_correctly(self, pca_results):
        """
        Total explained variance across all components should be <= 1.0.
        For yield curves, first 3 PCs typically explain >90%.
        """
        for country, res in pca_results.items():
            total_var = res["explained_var"].sum()
            assert 0 < total_var <= 1.0 + 1e-10, \
                f"{country}: total explained variance {total_var:.4f} is invalid"
    
    def test_high_explained_variance(self, pca_results):
        """
        For yield curve data, the first 3 components should explain 
        the vast majority of variance (>85%).
        """
        for country, res in pca_results.items():
            total_var = res["explained_var"].sum()
            assert total_var > 0.85, \
                f"{country}: first {res['n_components']} PCs explain only " \
                f"{total_var:.1%} of variance (expected >85%)"
    
    def test_pc1_dominates(self, pca_results):
        """
        PC1 (level) should explain more variance than any subsequent component.
        This is a fundamental property of yield curve PCA.
        """
        for country, res in pca_results.items():
            pc1_var = res["explained_var"][0]
            for i in range(1, len(res["explained_var"])):
                assert pc1_var > res["explained_var"][i], \
                    f"{country}: PC1 ({pc1_var:.1%}) should dominate " \
                    f"PC{i+1} ({res['explained_var'][i]:.1%})"
    
    def test_scores_have_correct_dates(self, pca_results, synthetic_yield_changes):
        """PCA scores should be indexed by trading dates."""
        for country, res in pca_results.items():
            dy = synthetic_yield_changes[country].dropna()
            assert len(res["scores"]) == len(dy), \
                f"{country}: score length mismatch"
            assert res["scores"].index.equals(dy.index), \
                f"{country}: score dates don't match input dates"
    
    def test_scores_are_standardized(self, pca_results):
        """
        PCA scores (from standardized data) should have approximately 
        zero mean (within numerical precision).
        """
        for country, res in pca_results.items():
            for col in res["scores"].columns:
                mean = res["scores"][col].mean()
                assert abs(mean) < 0.1, \
                    f"{country} {col}: mean score {mean:.4f} is not near zero"
    
    def test_loadings_are_unit_vectors(self, pca_results):
        """Each loading vector should have unit norm."""
        for country, res in pca_results.items():
            for i in range(res["n_components"]):
                loading_norm = np.linalg.norm(res["loadings"][i])
                assert abs(loading_norm - 1.0) < 1e-6, \
                    f"{country} PC{i+1}: loading norm is {loading_norm:.6f}, expected 1.0"

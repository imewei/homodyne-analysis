"""
Analysis Helper Functions
=========================

Utility functions for result interpretation, quality assessment, and
analysis support for the homodyne analysis package.

This module contains standalone helper functions that provide interpretation
and quality assessment for analysis results.
"""

from typing import List


def get_chi2_interpretation(chi2_value: float) -> str:
    """Provide interpretation of reduced chi-squared value with uncertainty context.

    The reduced chi-squared uncertainty quantifies the reliability of the average:
    - Small uncertainty (< 0.1 * χ²_red): Consistent fit quality across angles
    - Moderate uncertainty (0.1-0.5 * χ²_red): Some angle variation, generally acceptable
    - Large uncertainty (> 0.5 * χ²_red): High variability between angles, potential systematic issues

    Parameters
    ----------
    chi2_value : float
        Reduced chi-squared value

    Returns
    -------
    str
        Interpretation string with quality assessment and statistical meaning
    """
    if chi2_value <= 1.0:
        return f"Excellent fit (χ²_red = {
            chi2_value:.2f} ≤ 1.0): Model matches data within expected noise"
    elif chi2_value <= 2.0:
        return f"Very good fit (χ²_red = {
            chi2_value:.2f}): Model captures main features with minor deviations"
    elif chi2_value <= 5.0:
        return f"Acceptable fit (χ²_red = {
            chi2_value:.2f}): Model reasonable but some systematic deviations present"
    elif chi2_value <= 10.0:
        return f"Poor fit (χ²_red = {
            chi2_value:.2f}): Significant deviations suggest model inadequacy or underestimated uncertainties"
    else:
        return f"Very poor fit (χ²_red = {
            chi2_value:.2f}): Major systematic deviations, model likely inappropriate"


def get_quality_explanation(quality: str) -> str:
    """Provide explanation of quality assessment."""
    explanations = {
        "excellent": "Model provides exceptional agreement with experimental data across all angles",
        "acceptable": "Model provides reasonable agreement with experimental data for most angles",
        "warning": "Model shows concerning deviations that may indicate systematic issues",
        "poor": "Model shows significant inadequacies in describing the experimental data",
        "critical": "Model is fundamentally inappropriate for this dataset",
    }
    return explanations.get(quality, "Unknown quality level")


def get_quality_recommendations(quality: str, issues: List[str]) -> List[str]:
    """Provide actionable recommendations based on quality assessment."""
    recommendations = []

    if quality == "excellent":
        recommendations.append("Results are reliable for publication")
        recommendations.append("Consider this model for further analysis")
    elif quality == "acceptable":
        recommendations.append("Results may be suitable with appropriate caveats")
        recommendations.append(
            "Consider checking specific angles with higher chi-squared"
        )
    elif quality == "warning":
        recommendations.append("Investigate systematic deviations before publication")
        recommendations.append("Consider alternative models or parameter ranges")
        recommendations.append("Check experimental uncertainties and data quality")
    elif quality in ["poor", "critical"]:
        recommendations.append("Do not use results for quantitative conclusions")
        recommendations.append("Consider fundamental model revision")
        recommendations.append("Check experimental setup and data processing")
        recommendations.append("Investigate alternative theoretical approaches")

    # Add issue-specific recommendations
    for issue in issues:
        if "outliers" in issue.lower():
            recommendations.append(
                "Investigate outlier angles for experimental artifacts"
            )
        if "good angles" in issue.lower():
            recommendations.append(
                "Consider focusing analysis on subset of reliable angles"
            )

    return recommendations

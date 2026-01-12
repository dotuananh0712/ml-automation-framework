"""Explainability configuration models.

Defines Pydantic models for SHAP-based model explainability.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ExplainerType(str, Enum):
    """Supported SHAP explainer types."""

    TREE = "tree"  # TreeExplainer for tree-based models
    LINEAR = "linear"  # LinearExplainer for linear models
    KERNEL = "kernel"  # KernelExplainer (model-agnostic, slow)
    AUTO = "auto"  # Auto-detect based on model type


class ExplainabilityConfig(BaseModel):
    """Configuration for SHAP model explainability.

    Example:
        ```yaml
        explainability:
          enabled: true
          explainer_type: auto
          max_samples: 1000
          generate_summary_plot: true
          generate_bar_plot: true
        ```
    """

    enabled: bool = Field(default=False, description="Enable SHAP explanations")
    explainer_type: ExplainerType = Field(
        default=ExplainerType.AUTO,
        description="SHAP explainer type (auto-detected if not specified)",
    )
    max_samples: int = Field(
        default=1000,
        ge=10,
        description="Maximum samples for SHAP calculation (for performance)",
    )
    background_samples: int = Field(
        default=100,
        ge=10,
        description="Background samples for KernelExplainer",
    )
    generate_summary_plot: bool = Field(
        default=True,
        description="Generate SHAP summary plot",
    )
    generate_bar_plot: bool = Field(
        default=True,
        description="Generate SHAP bar plot (feature importance)",
    )
    generate_beeswarm_plot: bool = Field(
        default=False,
        description="Generate SHAP beeswarm plot",
    )
    generate_dependence_plots: bool = Field(
        default=False,
        description="Generate SHAP dependence plots for top features",
    )
    dependence_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top features for dependence plots",
    )
    compute_interactions: bool = Field(
        default=False,
        description="Compute SHAP interaction values (computationally expensive)",
    )

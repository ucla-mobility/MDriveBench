"""
Scenario Generator Package

Automated generation of diverse, valid driving scenarios for multi-agent coordination testing.
"""

from .capabilities import (
    CATEGORY_FEASIBILITY,
    CategoryFeasibility,
    TopologyType,
    ActorKind,
    MotionType,
    ConstraintType,
    EgoManeuver,
    LateralPosition,
    TimingPhase,
    get_feasible_categories,
    get_infeasible_categories,
)
from .constraints import (
    ScenarioSpec,
    EgoVehicleSpec,
    InterVehicleConstraint,
    NonEgoActorSpec,
    validate_spec,
    spec_to_dict,
    spec_from_dict,
)
from .validator import ScenarioValidator, ValidationResult
from .generator import ScenarioGenerator, GenerationConfig, TemplateGenerator
from .scene_validator import SceneValidator, SceneValidationResult, ValidationIssue, ValidationIssueDetail
from .generation_loop import (
    GenerationLoop,
    GenerationLoopConfig,
    GenerationResult,
    PipelineRunner,
    GenerationLogger,
)
from .scenario_ir import (
    ScenarioIR,
    VehicleIR,
    ConstraintIR,
    ActorIR,
    Cardinal,
    extract_ir_with_llm,
    extract_ir_with_regex,
    merge_ir_extractions,
)
from .geometric_checker import (
    GeometricError,
    GeometricValidationResult,
    validate_geometric_consistency,
    build_geometric_feedback,
)

__all__ = [
    # Capabilities
    "CATEGORY_FEASIBILITY",
    "CategoryFeasibility",
    "TopologyType",
    "ActorKind",
    "MotionType", 
    "ConstraintType",
    "EgoManeuver",
    "LateralPosition",
    "TimingPhase",
    "get_feasible_categories",
    "get_infeasible_categories",
    # Constraints
    "ScenarioSpec",
    "EgoVehicleSpec",
    "InterVehicleConstraint",
    "NonEgoActorSpec",
    "validate_spec",
    "spec_to_dict",
    "spec_from_dict",
    # Validator
    "ScenarioValidator",
    "ValidationResult",
    # Generator
    "ScenarioGenerator",
    "GenerationConfig",
    "TemplateGenerator",
    # Scene Validator
    "SceneValidator",
    "SceneValidationResult",
    "ValidationIssue",
    "ValidationIssueDetail",
    # Generation Loop
    "GenerationLoop",
    "GenerationLoopConfig",
    "GenerationResult",
    "PipelineRunner",
    "GenerationLogger",
    # Scenario IR
    "ScenarioIR",
    "VehicleIR",
    "ConstraintIR",
    "ActorIR",
    "Cardinal",
    "extract_ir_with_llm",
    "extract_ir_with_regex",
    "merge_ir_extractions",
    # Geometric Checker
    "GeometricError",
    "GeometricValidationResult",
    "validate_geometric_consistency",
    "build_geometric_feedback",
]

"""Config-string → class resolution, and the model-type derivation built on it.

:func:`derive_model_type` is shared by the eval session and infer recipe
resolution, so it lives here with the class-resolution helpers it stands on
rather than in ``leaderboard/session.py`` — importing the session to reach it
would point the dependency sideways across the two CLI subpackages.
``scripts/check_layer_imports.py`` rejects that edge.

The model-kind resolver consumes the provider-neutral requirements projection
from ``sieval.core``. It deliberately imports neither CLI subpackage: both the
eval composition root and infer recipe resolution depend on this module, never
on each other.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import copy
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from loguru import logger

from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    InlineModelBinding,
    InputKind,
    NamedModelBinding,
    RequirementContext,
    TaskModelRequirement,
    aggregate_task_requirements,
)
from sieval.core.types import JSONValue

# Registry for simple name lookups
DATASET_MODULE = "sieval.datasets"
TASK_MODULE = "sieval.tasks"


def load_class_from_path(class_path: str) -> type:
    """
    Load a class from a full module path like 'sieval.core.datasets.AIME2024Dataset'.
    """
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path: {class_path}. Expected format: 'module.ClassName'"
        )

    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as exc:
        if _is_missing_module_error(exc, module_name):
            raise ImportError(f"Could not import module '{module_name}'") from exc
        # Internal dependency missing — propagate the original error
        raise
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no class '{class_name}'"
        ) from e


def load_class_from_name(name: str, search_modules: list[str]) -> type:
    """Load a class by searching in multiple modules."""
    for module_name in search_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        except ImportError as exc:
            if not _is_missing_module_error(exc, module_name):
                raise
            continue

    raise ImportError(
        f"Could not find class '{name}' in any of: {search_modules}. "
        f"Use full path like 'my_module.{name}' for custom classes."
    )


def resolve_class(class_spec: str, search_modules: list[str]) -> type:
    """
    Resolve a class from either:
    - A full path: "sieval.core.datasets.AIME2024Dataset"
    - A simple name: "AIME2024Dataset"
    """
    if class_spec.startswith("."):
        raise ValueError(
            f"Relative import syntax is not supported: '{class_spec}'. "
            f"Use a simple name ('MyClass') or full path ('pkg.module.MyClass')"
        )
    if "." in class_spec:
        # Looks like a full path
        return load_class_from_path(class_spec)
    else:
        # Simple name, search in modules
        return load_class_from_name(class_spec, search_modules)


def resolve_dataset_class(class_spec: str) -> type:
    """Resolve a dataset class."""
    return resolve_class(class_spec, [DATASET_MODULE])


def _is_missing_module_error(exc: ImportError, target_module: str) -> bool:
    """Check if an ImportError is due to the target module itself not existing.

    Returns True only when the missing module IS the one we tried to import
    (i.e. the module simply doesn't exist).  Returns False when the target
    module exists but one of its internal dependencies is missing — in that
    case the error should be propagated so the user sees the real cause.
    """
    missing = getattr(exc, "name", None)
    if missing is None:
        return False
    # The target module itself is missing, or one of its parent packages
    return missing == target_module or target_module.startswith(f"{missing}.")


def resolve_task_class(class_spec: str) -> type:
    """Resolve a task class by searching in sieval.tasks submodules."""
    if class_spec.startswith("."):
        raise ValueError(
            f"Relative import syntax is not supported: '{class_spec}'. "
            f"Use a simple name ('MyClass') or full path ('pkg.module.MyClass')"
        )
    # For tasks, we need to search in submodules of sieval.tasks
    if "." in class_spec:
        return load_class_from_path(class_spec)

    # Try to find in sieval.tasks submodules
    try:
        tasks_module = importlib.import_module(TASK_MODULE)
        # Check if directly available (re-exported in __init__)
        if hasattr(tasks_module, class_spec):
            return getattr(tasks_module, class_spec)
    except ImportError as exc:
        if not _is_missing_module_error(exc, TASK_MODULE):
            raise
        # sieval.tasks itself doesn't exist — fall through to heuristic search

    # Search in submodules based on naming convention
    # e.g., "AIME2024ZeroShotGenTask" -> "aime_2024_0shot_gen"
    submodule_candidates = _guess_submodule_names(class_spec)
    for submodule in submodule_candidates:
        full_module = f"{TASK_MODULE}.{submodule}"
        try:
            module = importlib.import_module(full_module)
            if hasattr(module, class_spec):
                return getattr(module, class_spec)
        except ImportError as exc:
            if not _is_missing_module_error(exc, full_module):
                raise
            continue

    raise ImportError(
        f"Could not find task class '{class_spec}'. "
        f"Use full path like 'sieval.tasks.my_task.{class_spec}' for custom tasks."
    )


def validate_named_config_map(
    section_name: str,
    section_cfg: Any,
) -> dict[str, dict[str, Any]]:
    """Validate a config section is a ``name -> dict`` mapping, and return it.

    Shared so every entry point that reads a section reports the same error for
    the same malformed config. ``derive_model_type`` is now reached from the
    infer layer *before* an ``EvalSession`` exists, and full config validation
    only runs under ``--dry-run``, so without this a list-shaped ``tasks:``
    surfaced as an ``AttributeError`` from inside recipe resolution.
    """
    if not isinstance(section_cfg, dict):
        raise ValueError(
            f"'{section_name}' configuration must be a dictionary "
            "mapping names to config"
        )

    for item_name, item_cfg in section_cfg.items():
        if not isinstance(item_cfg, dict):
            raise ValueError(
                f"'{section_name}.{item_name}' configuration must be a dictionary"
            )

    return section_cfg


def derive_model_type(
    model_name: str,
    explicit_type: str | None,
    requirements: AggregatedTaskRequirements,
) -> Literal["chat", "gen"]:
    """Derive one deployment root's legacy kind from normalized evidence.

    ``TaskModelRequirement.requires.input`` is the sole task-side authority.
    When that evidence exists, legacy YAML ``type:`` is a checked assertion.
    With no input evidence, an explicit value remains a compatibility fallback;
    otherwise the historical ``chat`` default applies.

    The resolver never inspects task ``model_type`` metadata, model wrapper
    classes, dialects, engines, URLs, or provider response shapes.
    """

    if not isinstance(model_name, str) or not model_name:
        raise TypeError("model_name must be a non-empty string")
    if explicit_type not in (None, "chat", "gen"):
        raise ValueError(
            f"Model deployment root '{model_name}' has invalid type "
            f"{explicit_type!r}; expected 'chat' or 'gen'"
        )
    if not isinstance(requirements, AggregatedTaskRequirements):
        raise TypeError("requirements must be AggregatedTaskRequirements")

    kinds = requirements.input
    if len(kinds) > 1:
        evidence = "\n".join(
            "  - "
            + kind.value
            + ": "
            + ", ".join(sorted(requirements.input_sources.get(kind, ())))
            for kind in sorted(kinds, key=lambda item: item.value)
        )
        raise ValueError(
            f"Model deployment root '{model_name}' has conflicting normalized "
            f"input requirements:\n{evidence}\n"
            "Chat and completion consumers must use separate root model configs."
        )

    if kinds:
        input_kind = next(iter(kinds))
        derived_type: Literal["chat", "gen"] = (
            "chat" if input_kind is InputKind.CHAT else "gen"
        )
        if explicit_type is not None and explicit_type != derived_type:
            sources = ", ".join(sorted(requirements.input_sources.get(input_kind, ())))
            source_detail = f" from {sources}" if sources else ""
            raise ValueError(
                f"Model deployment root '{model_name}' declares type: "
                f"{explicit_type}, but normalized TaskRequirements require "
                f"'{derived_type}' ({input_kind.value}){source_detail}. Legacy "
                "type is a checked assertion when task evidence exists."
            )
        logger.info(
            "Derived model deployment root '{}' type as '{}' from normalized "
            "task requirements",
            model_name,
            derived_type,
        )
        return derived_type

    if explicit_type is not None:
        return cast(Literal["chat", "gen"], explicit_type)
    logger.info(
        "Using default type 'chat' for model deployment root '{}' with no input "
        "requirement evidence",
        model_name,
    )
    return "chat"


@dataclass(frozen=True)
class ConfigModelTypeResolution:
    """Legacy model kinds derived from normalized task requirement hooks."""

    model_types_by_root: Mapping[str, Literal["chat", "gen"]]
    model_types_by_config: Mapping[str, Literal["chat", "gen"]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_types_by_root",
            MappingProxyType(dict(self.model_types_by_root)),
        )
        object.__setattr__(
            self,
            "model_types_by_config",
            MappingProxyType(dict(self.model_types_by_config)),
        )


def resolve_config_model_types(
    config: Mapping[str, Any],
) -> ConfigModelTypeResolution:
    """Adapt YAML config into the normalized evidence accepted by the resolver.

    This is the recipe-layer adapter for callers that run before an
    :class:`EvalSession` exists. It invokes each task's
    ``model_requirements_for()`` hook with dialect-free bindings and then calls
    :func:`derive_model_type`; it never reads legacy ``Task.model_type``.

    Import/class failures remain fail-soft here because normal configuration
    validation reports them with richer context later. A successfully resolved
    hook is validated strictly so it cannot replace the supplied binding.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    models = validate_named_config_map("models", config.get("models", {}))
    tasks = validate_named_config_map("tasks", config.get("tasks", {}))
    raw_datasets = config.get("datasets", {})
    datasets = validate_named_config_map("datasets", raw_datasets)

    chains = {name: _config_model_chain(name, models) for name in models}
    bindings = {name: _config_named_binding(name, chains[name]) for name in models}
    records: list[TaskModelRequirement] = []

    for task_name, task_config in tasks.items():
        class_spec = task_config.get("class")
        if not isinstance(class_spec, str) or not class_spec:
            continue
        try:
            task_class = resolve_task_class(class_spec)
        except (ImportError, AttributeError):
            continue

        model_name = _config_task_model_name(task_name, task_config, models)
        candidate = bindings[model_name]
        context = _config_requirement_context(
            task_name,
            task_config,
            candidate,
            datasets,
        )
        requirement_hook = getattr(task_class, "model_requirements_for", None)
        if not callable(requirement_hook):
            continue
        task_records = requirement_hook(context)
        if not isinstance(task_records, tuple):
            raise TypeError(
                f"{task_class.__name__}.model_requirements_for() must return a tuple"
            )
        for record in task_records:
            if not isinstance(record, TaskModelRequirement):
                raise TypeError(
                    f"{task_class.__name__}.model_requirements_for() returned a "
                    "non-TaskModelRequirement value"
                )
            expected = context.model_bindings.get(record.role)
            if expected is None:
                raise ValueError(
                    f"{task_class.__name__}.model_requirements_for() declared "
                    f"unknown model role {record.role!r}"
                )
            if record.binding != expected:
                raise ValueError(
                    f"{task_class.__name__}.model_requirements_for() changed the "
                    f"normalized binding for role {record.role!r}"
                )
            records.append(record)

    by_root_bindings: dict[str, list[NamedModelBinding]] = {}
    for binding in bindings.values():
        by_root_bindings.setdefault(binding.root_deployment_key, []).append(binding)

    by_root: dict[str, Literal["chat", "gen"]] = {}
    by_config: dict[str, Literal["chat", "gen"]] = {}
    for root_key, root_bindings in by_root_bindings.items():
        root_name = chains[root_bindings[0].config_name][0][0]
        root_records = (
            record
            for record in records
            if isinstance(record.binding, NamedModelBinding)
            and record.binding.root_deployment_key == root_key
        )
        requirements = aggregate_task_requirements(root_records)
        explicit_type = _config_explicit_type_for_root(
            root_name,
            tuple(root_bindings),
            models,
        )
        model_type = derive_model_type(root_name, explicit_type, requirements)
        by_root[root_key] = model_type
        for binding in root_bindings:
            by_config[binding.config_name] = model_type

    return ConfigModelTypeResolution(by_root, by_config)


def _config_model_chain(
    model_name: str,
    models: Mapping[str, dict[str, Any]],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return one root-to-leaf model inheritance chain."""

    chain: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def visit(current: str) -> None:
        if current in seen:
            raise ValueError(
                "Circular model inheritance detected: "
                + " -> ".join([name for name, _ in chain] + [current])
            )
        seen.add(current)
        try:
            model_config = models[current]
        except KeyError as exc:
            raise ValueError(
                f"Model '{model_name}' references unknown base model '{current}'"
            ) from exc
        chain.append((current, model_config))
        base = model_config.get("base")
        if base is None:
            return
        if not isinstance(base, str) or not base:
            raise ValueError(f"Model '{current}' has invalid 'base' value: {base!r}")
        visit(base)

    visit(model_name)
    chain.reverse()
    return tuple(chain)


def _config_named_binding(
    model_name: str,
    chain: tuple[tuple[str, dict[str, Any]], ...],
) -> NamedModelBinding:
    root_name, root_config = chain[0]
    requested = root_config.get("name")
    if not requested:
        infer_config = root_config.get("infer")
        checkpoint = (
            infer_config.get("checkpoint")
            if isinstance(infer_config, Mapping)
            else None
        )
        if not checkpoint:
            checkpoint = root_config.get("path")
        requested = Path(checkpoint).name if checkpoint else root_name
    if not isinstance(requested, str) or not requested:
        raise ValueError(f"Model '{root_name}' has no usable requested model id")
    return NamedModelBinding(
        binding_id=f"model:{model_name}",
        root_deployment_key=f"model:{root_name}",
        requested_model_id=requested,
        config_name=model_name,
    )


def _config_task_model_name(
    task_name: str,
    task_config: Mapping[str, Any],
    models: Mapping[str, dict[str, Any]],
) -> str:
    model_ref = task_config.get("model")
    if model_ref is not None and not isinstance(model_ref, str):
        raise ValueError(f"Task '{task_name}': 'model' must be a string reference")
    if model_ref:
        if model_ref not in models:
            raise ValueError(
                f"Task '{task_name}' references unknown model '{model_ref}'"
            )
        return model_ref
    if len(models) == 1:
        return next(iter(models))
    if not models:
        raise ValueError(f"Task '{task_name}': no models defined in config")
    raise ValueError(
        f"Task '{task_name}': 'model' required when multiple models are defined"
    )


def _config_requirement_context(
    task_name: str,
    task_config: Mapping[str, Any],
    candidate: NamedModelBinding,
    datasets: Mapping[str, dict[str, Any]],
) -> RequirementContext:
    raw_args = task_config.get("args") or {}
    if not isinstance(raw_args, Mapping):
        raise ValueError(f"Task '{task_name}' args must be a dictionary")
    task_args = copy.deepcopy(dict(raw_args))
    model_bindings: dict[str, NamedModelBinding | InlineModelBinding] = {
        "candidate": candidate
    }

    grader = task_args.pop("grader", None)
    if grader is not None:
        if not isinstance(grader, Mapping):
            raise ValueError(
                f"Task '{task_name}' grader must be an inline mapping before launch"
            )
        model_bindings["grader"] = _config_inline_binding(task_name, "grader", grader)

    dataset_ref = task_config.get("dataset")
    if isinstance(dataset_ref, str):
        dataset_config: Mapping[str, Any] = datasets.get(dataset_ref, {})
    elif isinstance(dataset_ref, Mapping):
        dataset_config = dataset_ref
    else:
        # Model-kind selection remains fail-soft for an incomplete task; the
        # normal config validator reports the missing dataset later.
        dataset_config = {}

    infer_args = task_config.get("infer_args") or {}
    if not isinstance(infer_args, Mapping):
        raise ValueError(f"Task '{task_name}' infer_args must be a dictionary")
    return RequirementContext(
        model_bindings=model_bindings,
        task_args=cast(Mapping[str, JSONValue], task_args),
        dataset_config=cast(Mapping[str, JSONValue], copy.deepcopy(dataset_config)),
        infer_args=cast(Mapping[str, JSONValue], copy.deepcopy(dict(infer_args))),
    )


def _config_inline_binding(
    task_name: str,
    role: str,
    raw_config: Mapping[str, Any],
) -> InlineModelBinding:
    requested = raw_config.get("model")
    if not isinstance(requested, str) or not requested:
        raise ValueError(
            f"Task '{task_name}' inline {role} model requires a non-empty 'model'"
        )
    safe_config = {
        str(key): copy.deepcopy(value)
        for key, value in raw_config.items()
        if key not in {"api_key", "authorization"}
    }
    encoded = json.dumps(
        safe_config,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    dialect = raw_config.get("dialect", "openai_chat")
    if not isinstance(dialect, str) or not dialect:
        raise ValueError(
            f"Task '{task_name}' inline {role} dialect must be a non-empty string"
        )
    binding_id = f"inline:{task_name}:{role}:{digest}"
    return InlineModelBinding(
        binding_id=binding_id,
        root_deployment_key=f"deployment:{binding_id}",
        requested_model_id=requested,
        config=cast(Mapping[str, JSONValue], safe_config),
        dialect_id=dialect,
    )


def _config_explicit_type_for_root(
    root_name: str,
    bindings: tuple[NamedModelBinding, ...],
    models: Mapping[str, dict[str, Any]],
) -> Literal["chat", "gen"] | None:
    declarations = {
        binding.config_name: models[binding.config_name]["type"]
        for binding in bindings
        if "type" in models[binding.config_name]
    }
    invalid = {
        name: value
        for name, value in declarations.items()
        if value not in ("chat", "gen")
    }
    if invalid:
        details = ", ".join(
            f"{name}={value!r}" for name, value in sorted(invalid.items())
        )
        raise ValueError(
            f"Model deployment root '{root_name}' has invalid type assertion(s): "
            f"{details}; expected 'chat' or 'gen'"
        )
    values = set(declarations.values())
    if len(values) > 1:
        details = ", ".join(
            f"{name}={value!r}" for name, value in sorted(declarations.items())
        )
        raise ValueError(
            f"Models sharing deployment root '{root_name}' declare conflicting "
            f"type assertions: {details}"
        )
    if not values:
        return None
    return cast(Literal["chat", "gen"], next(iter(values)))


def _guess_submodule_names(class_name: str) -> list[str]:
    """
    Guess possible submodule names from a class name.
    e.g. "AIME2024ZeroShotGenTask" -> ["aime_2024_0shot_gen", "aime_2024_zero_shot_gen"]
    """
    import re

    # Remove 'Task' suffix if present
    name = class_name
    if name.endswith("Task"):
        name = name[:-4]

    # Convert CamelCase to snake_case with proper handling:
    # 1. Handle consecutive capitals followed by lowercase (e.g., "AIME" -> "AIME_")
    # 2. Handle lowercase/digit followed by uppercase (e.g., "tGen" -> "t_Gen")
    # 3. Handle letters followed by digits (e.g., "AIME2024" -> "AIME_2024")
    # 4. Handle digits followed by letters (e.g., "2024Zero" -> "2024_Zero")

    # Step 1: Insert underscore between consecutive capitals and following lowercase
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Step 2: Insert underscore between lowercase/digit and uppercase
    s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)
    # Step 3: Insert underscore between letters and digits
    s3 = re.sub(r"([A-Za-z])(\d)", r"\1_\2", s2)
    # Step 4: Insert underscore between digits and letters
    s4 = re.sub(r"(\d)([A-Za-z])", r"\1_\2", s3)

    snake = s4.lower()

    candidates = []

    # Primary candidate: with "0shot" style
    if "zero_shot" in snake:
        candidates.append(snake.replace("zero_shot", "0shot"))
    if "few_shot" in snake:
        candidates.append(snake.replace("few_shot", "kshot"))

    # Also include the original snake_case version
    candidates.append(snake)

    return candidates

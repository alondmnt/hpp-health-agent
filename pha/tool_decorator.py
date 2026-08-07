"""Auto-generate tool wrappers from typed function signatures.

Decorator that automatically:
1. Extracts input schema from function signature
2. Creates Pydantic validation models
3. Generates framework adapters (DSPy)
4. Wraps exceptions in structured error envelopes
5. Tracks provenance for reproducibility
6. Supports artifact tracking convention (_aid suffix)

Usage:
    @tool(version="1.0.0", categories=["CGM"])
    def my_tool(data_id: str, *, units: str = "mg/dL") -> ResultModel:
        '''Compute metrics from data.'''
        return ResultModel(...)

    # Direct call (returns metadata)
    result = my_tool(data_id="abc")

    # Framework call (returns envelope with provenance)
    envelope = my_tool.invoke({"data_id": "abc"})

    # Clean API access (recommended for new code)
    print(my_tool.tool.name)              # "my_tool"
    print(my_tool.tool.version)           # "1.0.0"
    envelope = my_tool.tool.invoke(data_id="abc")  # Pythonic kwargs

Design: Simple interface for developers, comprehensive adapter for frameworks.

Artifact Convention:
    Fields ending in '_aid' are automatically extracted as artifact references.
    This maintains a flat structure (zen: "flat is better than nested").

    class MyMetadata(BaseModel):
        result: float                      # Fact
        count: int                         # Fact
        output_data_aid: str               # Artifact reference

    return MyMetadata(
        result=42.0,
        count=100,
        output_data_aid=output_aid         # Auto-extracted to envelope.artifacts
    )

Error Codes:
    INPUT_VALIDATION, ARTIFACT_NOT_FOUND, INVALID_INPUT, KEY_ERROR,
    TYPE_ERROR, FILE_NOT_FOUND, PERMISSION_DENIED, TIMEOUT, UNEXPECTED

Public API:
    Recommended (clean):
        func.tool.name, func.tool.version, func.tool.categories,
        func.tool.description, func.tool.input_model, func.tool.output_model,
        func.tool.invoke()

    Legacy (still supported):
        func.__tool_name__, func.__version__, func.__categories__,
        func.invoke(), func.ainvoke()
"""
from __future__ import annotations

import inspect
import re
from typing import (
    Annotated, Any, Callable, Dict, Generic, Literal, Optional, Type, TypeVar, Union,
    get_origin, get_type_hints
)
from functools import wraps

from pydantic import BaseModel, Field, create_model, ConfigDict
from pydantic.fields import FieldInfo

T = TypeVar("T", bound=Callable)
M = TypeVar("M", bound=BaseModel)

# ============================================================================
# Tool Output Models (used by @tool decorator)
# ============================================================================

class Provenance(BaseModel):
    """Provenance information for tool execution.

    Tracks tool version and configuration used for reproducibility.

    Attributes:
        tool_version: Semantic version of the tool (e.g., "1.0.0")
        options_used: Configuration options used during execution
        input_hash: Optional hash of input data for reproducibility
    """
    model_config = ConfigDict(extra="forbid")

    tool_version: str
    options_used: Dict[str, Any] = Field(default_factory=dict)
    input_hash: Optional[str] = None


class ToolError(BaseModel):
    """Structured error information for tool failures.

    Used in ToolOutputBase.error field when status is "error".

    Attributes:
        code: Error code (e.g., "INPUT_VALIDATION", "TIMEOUT", "UNEXPECTED")
        message: Human-readable error message
        retriable: Whether the error is transient and can be retried
        details: Optional dictionary with additional error context
    """
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retriable: bool = False
    details: Optional[Dict[str, Any]] = None


class ToolOutputBase(BaseModel, Generic[M]):
    """Generic tool output envelope with typed metadata.

    This is used by the @tool decorator to wrap tool outputs with
    standard fields for status, errors, warnings, and provenance.

    Type Parameters:
        M: Pydantic model defining the tool-specific metadata structure

    Attributes:
        status: Execution status ("ok", "error", or "partial")
        summary: Human-readable execution summary
        metadata: Tool-specific results (typed via generic M), None on error
        artifacts: Dictionary mapping artifact names to storage IDs
        warnings: List of non-fatal warnings from execution
        error: Optional error details if status is "error"
        provenance: Optional provenance tracking
    """
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error", "partial"] = "ok"
    summary: str = ""
    metadata: Optional[M] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: Optional[ToolError] = None
    provenance: Optional[Provenance] = None

# ============================================================================
# Decorator Implementation
# ============================================================================

# Optional dependency for async support
try:
    import anyio
    ANYIO_AVAILABLE = True
except ImportError:
    ANYIO_AVAILABLE = False


__all__ = [
    # Decorator
    "tool",
    # Helpers
    "get_tool_wrapper",
    "get_tool_schemas",
    "to_dspy_tool",
    "convert_tools_for_dspy",
    # Models (for tool authors)
    "ToolOutputBase",
    "ToolError",
    "Provenance",
]


def _map_exception(e: Exception) -> ToolError:
    """Map an exception to a structured ToolError.

    Provides LLM-friendly error codes and retry hints for common error types.

    Args:
        e: The exception to map

    Returns:
        ToolError with appropriate code, message, retriable flag, and details
    """
    import traceback
    import sys
    from pydantic import ValidationError

    from .artifact_store import ArtifactNotFoundError, ArtifactTypeMismatchError

    if isinstance(e, ValidationError):
        return ToolError(
            code="INPUT_VALIDATION",
            message=str(e),
            retriable=False,
            details={"errors": e.errors()}
        )

    # Check custom artifact exceptions first (more specific than base types)
    if isinstance(e, ArtifactNotFoundError):
        return ToolError(
            code="ARTIFACT_NOT_FOUND",
            message=str(e),
            retriable=False,
        )

    if isinstance(e, ArtifactTypeMismatchError):
        return ToolError(
            code="ARTIFACT_TYPE_MISMATCH",
            message=str(e),
            retriable=False,
        )

    if isinstance(e, TimeoutError):
        return ToolError(
            code="TIMEOUT",
            message=str(e),
            retriable=True,
        )

    if isinstance(e, FileNotFoundError):
        return ToolError(
            code="FILE_NOT_FOUND",
            message=str(e),
            retriable=False,
        )

    if isinstance(e, PermissionError):
        return ToolError(
            code="PERMISSION_DENIED",
            message=str(e),
            retriable=False,
        )

    if isinstance(e, ValueError):
        return ToolError(
            code="INVALID_INPUT",
            message=str(e),
            retriable=False,
        )

    # Generic KeyError/TypeError - less specific than custom artifact exceptions
    if isinstance(e, KeyError):
        return ToolError(
            code="KEY_ERROR",
            message=str(e),
            retriable=False,
        )

    if isinstance(e, TypeError):
        return ToolError(
            code="TYPE_ERROR",
            message=str(e),
            retriable=False,
        )

    # Catch-all for unexpected errors
    return ToolError(
        code="UNEXPECTED",
        message=str(e),
        retriable=False,
        details={
            "exc_type": type(e).__name__,
            "trace": traceback.format_exception(*sys.exc_info())[-1].strip()
        }
    )


def _extract_field_info(
    param: inspect.Parameter,
    description: Optional[str] = None,
) -> tuple[Any, FieldInfo]:
    """Extract Pydantic Field info for non-Annotated parameters.

    This is only used for parameters without Annotated metadata,
    using docstring descriptions as fallback.

    Args:
        param: Function parameter from inspect.signature()
        description: Optional description from docstring

    Returns:
        (type_annotation, FieldInfo) suitable for create_model()
    """
    # Get type annotation (non-Annotated)
    annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any

    # Build FieldInfo with docstring description
    if param.default == inspect.Parameter.empty:
        # Required parameter
        field_info = Field(..., description=description)
    else:
        # Optional with default from function signature
        field_info = Field(default=param.default, description=description)

    return (annotation, field_info)


def _extract_docstring_descriptions(func: Callable) -> Dict[str, str]:
    """Extract parameter descriptions from docstring Args section.

    Simple parser: expects one line per arg in format "param_name: description".
    Multi-line or indented continuations are intentionally UNSUPPORTED - use
    Annotated[..., Field(description="...")] for complex descriptions.

    Args:
        func: Function with docstring

    Returns:
        Dict mapping parameter names to descriptions (single-line only)
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return {}

    descriptions = {}
    in_args_section = False

    for line in docstring.split('\n'):
        stripped = line.strip()

        # Detect Args: section
        if stripped in ('Args:', 'Arguments:', 'Parameters:'):
            in_args_section = True
            continue

        # End of Args section (blank line or new section)
        if in_args_section and (not stripped or (stripped[0].isupper() and stripped.endswith(':'))):
            in_args_section = False
            continue

        if in_args_section and ':' in stripped:
            # Parse "param_name: description" (single line only, no continuation)
            parts = stripped.split(':', 1)
            param_name = parts[0].strip().split('(')[0].strip()  # Remove type hint if present
            description = parts[1].strip() if len(parts) > 1 else ""
            if param_name and description:
                descriptions[param_name] = description

    return descriptions


def _create_input_model_from_signature(
    func: Callable,
    func_name: str,
) -> Type[BaseModel]:
    """Dynamically create Pydantic input model from function signature.

    Only supports standard parameters (positional-or-keyword, keyword-only).
    Positional-only (/) parameters are rejected because the wrapper invokes
    the function with keyword arguments only.

    Args:
        func: Function to analyze
        func_name: Name for the model (e.g., 'cgm_metrics')

    Returns:
        Dynamically created Pydantic model class

    Raises:
        TypeError: If function uses *args, **kwargs, or positional-only (/) params
    """
    sig = inspect.signature(func)

    # Resolve annotations with get_type_hints to handle string annotations
    # (from __future__ import annotations) and preserve Annotated metadata
    try:
        hints = get_type_hints(func, include_extras=True)
    except (NameError, AttributeError, TypeError):
        # Can fail if forward references can't be resolved or invalid annotations
        hints = {}

    # Reject unsupported parameter kinds early
    # Positional-only params not supported because wrapper invokes with **kwargs
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise TypeError(
                f"{func.__name__} cannot use *args/**kwargs or positional-only (/) parameters. "
                f"Found {param.kind.name} parameter '{param_name}'."
            )

    # Extract descriptions from docstring (fallback only)
    descriptions = _extract_docstring_descriptions(func)

    # Build field definitions
    field_definitions = {}
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue

        # Get resolved annotation from type hints, fall back to signature
        ann = hints.get(param_name, param.annotation if param.annotation != inspect.Parameter.empty else Any)

        # Keep Annotated types intact - let create_model handle them
        if get_origin(ann) is Annotated:
            # Preserve Annotated metadata and constraints, use function default
            default = param.default if param.default != inspect.Parameter.empty else ...
            field_definitions[param_name] = (ann, default)
        else:
            # Non-Annotated: use Field(..., description=docstring) fallback
            annotation, field_info = _extract_field_info(param, descriptions.get(param_name))
            field_definitions[param_name] = (annotation, field_info)

    # Create model dynamically
    model_name = f"{func_name.title().replace('_', '')}Input"
    input_model = create_model(model_name, **field_definitions)

    # Add docstring
    input_model.__doc__ = f"Auto-generated input schema for {func_name}."

    return input_model


def _extract_artifacts_from_metadata(metadata: BaseModel) -> Dict[str, Any]:
    """Extract artifact references from metadata (convention helper).

    Convention:
    - Fields ending in '_aid' (singular) with string values are artifact references
    - Fields ending in '_aids' (plural) with dict values are multi-artifact references

    Args:
        metadata: Tool metadata model instance

    Returns:
        Dict mapping artifact names to artifact IDs

    Example:
        >>> class MyMetadata(BaseModel):
        ...     daily_metrics_aid: str
        ...     modality_aids: Dict[str, str]
        ...     result: float
        >>> meta = MyMetadata(
        ...     daily_metrics_aid="df_123",
        ...     modality_aids={"cgm": "df_abc", "diet": "df_def"},
        ...     result=42.0
        ... )
        >>> _extract_artifacts_from_metadata(meta)
        {'daily_metrics': 'df_123', 'cgm': 'df_abc', 'diet': 'df_def'}
    """
    artifacts = {}

    for field_name, field_value in metadata.model_dump().items():
        # Single artifact: field_aid -> artifact_name: artifact_id
        if field_name.endswith('_aid') and isinstance(field_value, str):
            artifact_name = field_name[:-4]  # Remove '_aid' suffix
            artifacts[artifact_name] = field_value

        # Multiple artifacts: field_aids -> {key: artifact_id, ...}
        elif field_name.endswith('_aids') and isinstance(field_value, dict):
            # Flatten dict into artifacts (each key becomes artifact name)
            for key, aid in field_value.items():
                if isinstance(aid, str):
                    artifacts[key] = aid

    return artifacts


def _build_envelope(
    tool_name: str,
    version: str,
    req: BaseModel,
    meta: BaseModel,
    output_model: Type[BaseModel],
) -> BaseModel:
    """Build standard output envelope (DRY helper).

    Automatically extracts artifact references (fields ending in '_aid')
    from metadata and populates envelope.artifacts dict.

    Args:
        tool_name: Name of the tool
        version: Tool version
        req: Validated input model instance
        meta: Validated metadata model instance
        output_model: Output envelope model class

    Returns:
        Output envelope instance with artifacts extracted from metadata
    """
    return output_model(
        status="ok",
        summary=f"Executed {tool_name}",
        metadata=meta,
        artifacts=_extract_artifacts_from_metadata(meta),
        provenance=Provenance(
            tool_version=version,
            options_used=req.model_dump(),
        ),
    )


def _normalize_invoke(
    pyfunc: Callable,
    input_model: Type[BaseModel],
    metadata_model: Type[BaseModel],
    output_model: Type[BaseModel],
    *,
    tool_name: str,
    version: str,
) -> tuple[Callable[[dict], BaseModel], Optional[Callable]]:
    """Create async-safe invoke() and ainvoke() methods.

    Returns both sync and async versions to support different execution contexts.
    The sync version detects running event loops and adapts accordingly.

    Args:
        pyfunc: The original tool function (returns metadata model)
        input_model: Pydantic model for input validation
        metadata_model: Pydantic model returned by pyfunc
        output_model: Pydantic model for full output envelope
        tool_name: Name of the tool
        version: Tool version

    Returns:
        (invoke, ainvoke) tuple:
        - invoke: Sync function (with smart event loop detection)
        - ainvoke: Native async function (or None if anyio unavailable)
    """

    def _execute_sync(payload: dict) -> BaseModel:
        """Execute tool synchronously and build envelope.

        DRY helper to avoid duplication between sync/async paths.
        """
        parsed = input_model.model_validate(payload)
        res = pyfunc(**parsed.model_dump())
        if inspect.isawaitable(res):
            raise RuntimeError(
                f"Cannot call async function {pyfunc.__name__} synchronously. "
                f"Use `.ainvoke(...)` instead."
            )
        # Validate metadata and build envelope
        meta = metadata_model.model_validate(res)
        return _build_envelope(tool_name, version, parsed, meta, output_model)

    def _build_error_envelope(e: Exception, payload: dict, parsed: Optional[BaseModel] = None) -> BaseModel:
        """Build error envelope (DRY helper)."""
        error = _map_exception(e)
        try:
            options_used = parsed.model_dump() if parsed else payload
        except Exception:
            options_used = payload

        return output_model(
            status="error",
            summary=f"{tool_name} failed: {error.code}",
            metadata=None,
            error=error,
            provenance=Provenance(
                tool_version=version,
                options_used=options_used,
            )
        )

    if ANYIO_AVAILABLE:
        async def _ainvoke(payload: dict) -> BaseModel:
            """Async invoke: validates input, calls function, builds envelope."""
            try:
                parsed = input_model.model_validate(payload)
                res = pyfunc(**parsed.model_dump())
                if inspect.isawaitable(res):
                    res = await res
                # Validate metadata and build envelope
                meta = metadata_model.model_validate(res)
                return _build_envelope(tool_name, version, parsed, meta, output_model)
            except Exception as e:
                return _build_error_envelope(e, payload, parsed if 'parsed' in locals() else None)

        def _sync_invoke(payload: dict) -> BaseModel:
            """Sync invoke with smart event loop detection.

            Detects if there's a running event loop (Jupyter, FastAPI, etc.)
            and falls back to direct synchronous execution if so.
            """
            import asyncio

            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # Event loop is running - execute synchronously
                try:
                    return _execute_sync(payload)
                except Exception as e:
                    parsed = None
                    try:
                        parsed = input_model.model_validate(payload)
                    except Exception:
                        pass
                    return _build_error_envelope(e, payload, parsed)

            except RuntimeError:
                # No event loop running - use anyio.run as designed
                return anyio.run(_ainvoke, payload)

        return _sync_invoke, _ainvoke
    else:
        # No anyio available - sync only
        def _sync_invoke(payload: dict) -> BaseModel:
            """Sync invoke (no async support)."""
            try:
                return _execute_sync(payload)
            except Exception as e:
                parsed = None
                try:
                    parsed = input_model.model_validate(payload)
                except Exception:
                    pass
                return _build_error_envelope(e, payload, parsed)

        return _sync_invoke, None


def _pick_description(func: Callable, explicit: Optional[str]) -> str:
    """Pick tool description with fallback precedence.

    Precedence:
    1. Explicit description from @tool(description="...")
    2. First paragraph of function docstring
    3. Humanized function name

    Args:
        func: Function to get description for
        explicit: Explicit description from decorator arg

    Returns:
        Tool description string
    """
    if explicit and explicit.strip():
        return explicit.strip()

    if func.__doc__:
        # Get first paragraph (until first blank line)
        para = func.__doc__.strip().split("\n\n", 1)[0].strip()
        if para:
            return para

    # Fallback: humanize function name
    return func.__name__.replace("_", " ").capitalize()


class ToolMetadata:
    """Namespaced metadata accessor for decorated tools.

    Provides a clean API for accessing tool metadata without polluting
    the function namespace with dunder attributes.

    Attributes:
        name: Tool name
        version: Semantic version
        categories: List of categories
        description: Tool description
        mandatory: Whether tool is mandatory
        depends_on: List of tool dependencies
        input_model: Pydantic input schema
        output_model: Pydantic output envelope schema

    Example:
        >>> @tool(version="1.0.0", categories="CGM")
        ... def my_tool(data_id: str) -> ResultModel:
        ...     return ResultModel(...)
        >>>
        >>> # New API (clean)
        >>> print(my_tool.tool.name)
        >>> print(my_tool.tool.version)
        >>> result = my_tool.tool.invoke({"data_id": "abc"})
        >>>
        >>> # Old API (still works)
        >>> print(my_tool.__tool_name__)
        >>> result = my_tool.invoke({"data_id": "abc"})
    """

    def __init__(self, func_ref: Callable) -> None:
        """Initialize with reference to decorated function.

        Args:
            func_ref: The decorated function with metadata attributes
        """
        self._func = func_ref

    @property
    def name(self) -> str:
        """Tool name."""
        return self._func.__tool_name__

    @property
    def version(self) -> str:
        """Tool version."""
        return self._func.__version__

    @property
    def categories(self) -> list[str]:
        """Tool categories."""
        return self._func.__categories__

    @property
    def description(self) -> str:
        """Tool description."""
        return self._func.__description__

    @property
    def mandatory(self) -> bool:
        """Whether tool is mandatory."""
        return self._func.__mandatory__

    @property
    def depends_on(self) -> list[str]:
        """Tool dependencies."""
        return self._func.__depends_on__

    @property
    def input_model(self) -> Type[BaseModel]:
        """Pydantic input schema."""
        return self._func.__input_model__

    @property
    def output_model(self) -> Type[BaseModel]:
        """Pydantic output envelope schema."""
        return self._func.__output_model__

    def invoke(self, payload: Optional[dict] = None, **kwargs) -> BaseModel:
        """Universal invoke accepting dict or kwargs.

        Provides a unified interface that accepts both dictionary-style
        (registry/framework) and keyword argument-style (Pythonic) calls.

        Args:
            payload: Dictionary of parameters (registry style)
            **kwargs: Keyword arguments (Pythonic style)

        Returns:
            Tool output envelope

        Example:
            >>> # Dict-style (registry/framework)
            >>> result = tool.tool.invoke({"data_id": "abc"})
            >>>
            >>> # Kwargs-style (Pythonic)
            >>> result = tool.tool.invoke(data_id="abc")
        """
        if payload is not None:
            # Dict-style call from registry/framework
            return self._func.invoke(payload)
        else:
            # Kwargs-style call (pythonic)
            return self._func.invoke(kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ToolMetadata(name={self.name!r}, version={self.version!r}, "
            f"categories={self.categories!r})"
        )


def tool(
    *,
    version: str = "1.0.0",
    categories: Union[list[str], str] = "Health",
    name: Optional[str] = None,
    description: Optional[str] = None,
    mandatory: bool = False,
    depends_on: Optional[list[str]] = None,
) -> Callable[[T], T]:
    """Decorator to auto-generate a validated tool wrapper.

    This decorator:
    1. Extracts input schema from signature (prefers Annotated[..., Field(...)])
    2. Creates Pydantic input model automatically
    3. Generates wrapper: dict → validate → call typed function → validate output → wrap
    4. Wraps ALL exceptions in structured error envelopes (never raises)
    5. Adds invoke() and ainvoke() methods for registry-agnostic calling

    Error Handling:
        All exceptions are caught and returned as structured error envelopes with
        status="error", ToolError (code, message, retriable), and preserved provenance.
        This ensures LLM agents always get parseable responses.

    Args:
        version: Tool version (semantic versioning)
        categories: Tool categories (string or list). Can belong to multiple categories.
        name: Override tool name (defaults to function name)
        description: Override description (uses fallback precedence)
        mandatory: If True, this tool MUST always be included in workflows for its categories
        depends_on: List of tool names this tool depends on (for orchestration planning)

    Returns:
        Decorated function (unchanged, with metadata attached)

    Example:
        from typing import Annotated
        from pydantic import Field

        @tool(
            version="1.0.0",
            categories=["CGM", "DataProcessing"],
            mandatory=True,
            depends_on=["cgm_parser"]
        )
        def compute_cgm_metrics(
            timeseries_aid: Annotated[str, Field(description="Artifact ID of CGM data")],
            *,
            units: Annotated[Literal["mg/dL", "mmol/L"], Field(description="Units")] = "mg/dL",
        ) -> CGMMetricsMetadata:
            '''Compute CGM metrics.'''
            # ... work ...
            return CGMMetricsMetadata(...)
    """
    def decorator(func: T) -> T:
        # Normalize categories to list
        if isinstance(categories, str):
            _categories = [categories]
        elif isinstance(categories, list):
            _categories = categories
        else:
            _categories = ["Health"]

        # Get function name and sanitize for valid Python identifier
        raw_name = name or func.__name__
        tool_name = re.sub(r'[^0-9a-zA-Z_]', '_', raw_name)

        # Get description with fallback precedence
        tool_description = _pick_description(func, description)

        # Get return type for output model
        # Use get_type_hints to resolve string annotations (from __future__ import annotations)
        try:
            type_hints = get_type_hints(func)
            return_annotation = type_hints.get('return', inspect.Signature.empty)
        except (NameError, AttributeError, TypeError):
            # Fallback to signature if get_type_hints fails (forward refs, etc.)
            sig = inspect.signature(func)
            return_annotation = sig.return_annotation

        if return_annotation == inspect.Signature.empty:
            raise TypeError(f"{func.__name__} must have return type annotation")

        # Check if return type is a Pydantic model (more robust check)
        try:
            is_pydantic = inspect.isclass(return_annotation) and issubclass(return_annotation, BaseModel)
        except TypeError:
            # issubclass can raise if return_annotation isn't a class
            is_pydantic = False

        if not is_pydantic:
            raise TypeError(
                f"{func.__name__} must return a Pydantic BaseModel, "
                f"got {return_annotation} (type: {type(return_annotation)})"
            )

        metadata_model = return_annotation

        # Create input model from signature (includes varargs check)
        input_model = _create_input_model_from_signature(func, tool_name)

        # Create output model (use the sanitized name for class names)
        safe_name = tool_name.title().replace('_', '')
        output_model_name = f"{safe_name}Output"
        output_model = type(
            output_model_name,
            (ToolOutputBase[metadata_model],),
            {'__doc__': f"Auto-generated output schema for {tool_name}."}
        )

        # Create wrapper function
        @wraps(func)
        def wrapper(inp: dict) -> dict:
            """Auto-generated wrapper for registry compatibility."""
            try:
                # 1. Validate input dict
                req = input_model.model_validate(inp)

                # 2. Call typed function
                result = func(**req.model_dump())

                # 3. Validate metadata
                validated_metadata = metadata_model.model_validate(result)

                # 4. Build envelope using DRY helper
                output = _build_envelope(tool_name, version, req, validated_metadata, output_model)

                # 5. Return as dict
                return output.model_dump(by_alias=True)

            except Exception as e:
                # Map exception to structured error
                error = _map_exception(e)

                # Build error envelope (req may not exist if validation failed)
                try:
                    options_used = req.model_dump() if 'req' in locals() else inp
                except Exception:
                    options_used = inp

                output = output_model(
                    status="error",
                    summary=f"{tool_name} failed: {error.code}",
                    metadata=None,
                    error=error,
                    provenance=Provenance(
                        tool_version=version,
                        options_used=options_used,
                    )
                )
                return output.model_dump(by_alias=True)

        # Create async-safe invoke() and ainvoke() methods
        # invoke returns full envelope for consistency with wrapper
        invoke, ainvoke = _normalize_invoke(
            func, input_model, metadata_model, output_model,
            tool_name=tool_name, version=version
        )

        # Attach metadata to original function (public stable API via dunders)
        # Single-underscore attrs are internal/legacy - prefer dunders
        setattr(func, "__tool_name__", tool_name)
        setattr(func, "__version__", version)
        setattr(func, "__categories__", _categories)
        setattr(func, "__description__", tool_description)
        setattr(func, "__mandatory__", mandatory)
        setattr(func, "__depends_on__", depends_on or [])
        setattr(func, "__input_model__", input_model)
        setattr(func, "__output_model__", output_model)
        setattr(func, "invoke", invoke)
        if ainvoke:
            setattr(func, "ainvoke", ainvoke)

        # Internal attrs for helper functions (get_tool_schemas, get_tool_wrapper)
        setattr(func, "_tool_wrapper", wrapper)
        setattr(func, "_input_model", input_model)
        setattr(func, "_output_model", output_model)

        # Add namespaced metadata accessor (new clean API)
        setattr(func, "tool", ToolMetadata(func))

        # Return original function (unchanged, but with metadata attached)
        return func

    return decorator


def get_tool_wrapper(func: Callable) -> Callable[[dict], dict]:
    """Get the auto-generated wrapper for a decorated tool function.

    The wrapper provides a dict → dict interface compatible with tool registries
    that expect plain JSON inputs and outputs (like LangChain).

    Args:
        func: Function decorated with @tool

    Returns:
        The wrapper function (dict -> dict).
        Wrapper handles: validation, calling typed function, wrapping output.

    Raises:
        AttributeError: If function not decorated with @tool
    """
    try:
        return getattr(func, '_tool_wrapper')
    except AttributeError as e:
        raise AttributeError(
            f"Function '{func.__name__}' is not decorated with @tool. "
            f"Use @tool(version='...') to decorate it first."
        ) from e


def get_tool_schemas(func: Callable) -> tuple[Type[BaseModel], Type[BaseModel]]:
    """Get input and output models for a decorated tool.

    Args:
        func: Function decorated with @tool

    Returns:
        (InputModel, OutputModel) tuple

    Raises:
        AttributeError: If function not decorated with @tool
    """
    try:
        input_model = getattr(func, '_input_model')
        output_model = getattr(func, '_output_model')
        return (input_model, output_model)
    except AttributeError as e:
        raise AttributeError(
            f"Function '{func.__name__}' is not decorated with @tool. "
            f"Use @tool(version='...') to decorate it first."
        ) from e


def _tool_trace_enabled() -> bool:
    """Return whether ReAct tool call tracing should be emitted."""
    import os

    return os.environ.get("PHA_TOOL_TRACE", "").lower() in {"1", "true", "yes", "on"}


def _redact_tool_args(value: Any) -> Any:
    """Redact sensitive values while preserving enough shape for debugging."""
    sensitive_terms = ("api_key", "key", "token", "secret", "password")

    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(term in key_text for term in sensitive_terms):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_tool_args(item)
        return redacted

    if isinstance(value, list):
        return [_redact_tool_args(item) for item in value[:20]]

    if isinstance(value, tuple):
        return tuple(_redact_tool_args(item) for item in value[:20])

    if isinstance(value, str) and len(value) > 500:
        return f"{value[:500]}...<truncated {len(value)} chars>"

    return value


def _emit_tool_trace(event: str, tool_name: str, **fields: Any) -> None:
    """Emit a machine-readable trace line for one ReAct tool event.

    Enabled by setting PHA_TOOL_TRACE=1. Lines go to stderr prefixed with
    PHA_TOOL_TRACE so they can be grepped out of a run log.
    """
    import json
    import sys
    from datetime import datetime, timezone

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "tool": tool_name,
        **fields,
    }
    print(f"PHA_TOOL_TRACE {json.dumps(record, default=str)}", file=sys.stderr, flush=True)


def to_dspy_tool(func: Callable) -> Callable:
    """Convert @tool decorated function to DSPy-compatible format.

    DSPy ReAct and other agents expect tools to:
    - Accept **kwargs (handled by preserving signature)
    - Return str or simple dict for LLM observation
    - Have __name__, __doc__, __annotations__ for introspection

    This adapter:
    1. Wraps the tool's invoke() method
    2. Converts ToolOutputBase envelope → concise observation string
    3. Preserves function signature for DSPy parameter extraction
    4. Returns JSON-formatted observations for LLM reasoning

    Args:
        func: Function decorated with @tool

    Returns:
        DSPy-compatible wrapper function (kwargs → observation string)

    Raises:
        AttributeError: If function not decorated with @tool

    Example:
        @tool(version="1.0.0", categories=["CGM"])
        def cgm_metrics(timeseries_aid: str, units: str = "mg/dL") -> MetricsMetadata:
            '''Compute CGM metrics.'''
            ...

        # Convert for DSPy
        cgm_metrics_dspy = to_dspy_tool(cgm_metrics)

        # Use in ReAct
        from dspy.agents import ReAct
        agent = ReAct(tools=[cgm_metrics_dspy], max_iters=5)
        result = agent("Analyze CGM data from artifact abc123")
    """
    import json
    from functools import wraps

    # Verify it's a decorated tool
    if not hasattr(func, '__tool_name__'):
        raise AttributeError(
            f"Function '{func.__name__}' must be decorated with @tool first. "
            f"Use @tool(version='...') before calling to_dspy_tool()."
        )

    if not hasattr(func, 'invoke'):
        raise AttributeError(
            f"Function '{func.__name__}' is missing 'invoke' method. "
            f"This might be an older version of @tool decorator."
        )

    @wraps(func)
    def dspy_wrapper(**kwargs) -> str:
        """DSPy-compatible wrapper returning observation string.

        This function is called by DSPy agents with tool arguments.
        It returns a formatted observation string for the LLM to reason about.
        """
        trace_enabled = _tool_trace_enabled()
        if trace_enabled:
            import time

            start_time = time.monotonic()
            _emit_tool_trace(
                "start",
                func.__tool_name__,
                args=_redact_tool_args(kwargs),
            )
        else:
            start_time = 0.0

        # Call the tool via invoke() to get full envelope with error handling
        try:
            result = func.invoke(kwargs)
        except Exception as exc:
            if trace_enabled:
                import time

                _emit_tool_trace(
                    "exception",
                    func.__tool_name__,
                    elapsed_seconds=round(time.monotonic() - start_time, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise

        if trace_enabled:
            import time

            _emit_tool_trace(
                "end",
                func.__tool_name__,
                elapsed_seconds=round(time.monotonic() - start_time, 3),
                status=result.status,
                summary=result.summary,
            )

        # Convert envelope to concise observation for LLM
        if result.status == "ok":
            # Success: format key information for LLM
            obs_parts = [f"✓ {result.summary}"]

            # Add metadata (no truncation - metadata is typically compact and all fields may be needed)
            if result.metadata:
                metadata_dict = result.metadata.model_dump()
                obs_parts.append(f"Metadata: {json.dumps(metadata_dict, indent=2)}")

            # Add artifact references
            if result.artifacts:
                obs_parts.append(f"Artifacts: {json.dumps(result.artifacts)}")

            # Add warnings if any
            if result.warnings:
                obs_parts.append(f"Warnings: {', '.join(result.warnings)}")

            return "\n".join(obs_parts)

        else:
            # Error: return structured error for LLM to understand
            error_info = {
                "status": "error",
                "error_code": result.error.code if result.error else "UNKNOWN",
                "message": result.error.message if result.error else "Unknown error",
                "retriable": result.error.retriable if result.error else False,
            }

            # Add error details if available
            if result.error and result.error.details:
                error_info["details"] = result.error.details

            return f"✗ Tool failed\n{json.dumps(error_info, indent=2)}"

    # Preserve metadata for DSPy introspection
    # DSPy uses inspect.signature() and __doc__ to understand tool parameters
    dspy_wrapper.__name__ = func.__tool_name__
    dspy_wrapper.__doc__ = func.__description__

    # Copy function signature for DSPy parameter extraction
    # This allows DSPy to discover parameters and their types
    sig = inspect.signature(func)
    dspy_wrapper.__signature__ = sig

    # Copy annotations (DSPy may use these)
    if hasattr(func, '__annotations__'):
        dspy_wrapper.__annotations__ = func.__annotations__.copy()

    # Store reference to original decorated function
    dspy_wrapper.__wrapped__ = func
    dspy_wrapper.__tool_name__ = func.__tool_name__
    dspy_wrapper.__is_dspy_tool__ = True  # Marker for debugging

    return dspy_wrapper


def convert_tools_for_dspy(
    tools: Union[list[Callable], Dict[str, Callable]]
) -> list[Callable]:
    """Convert @tool-decorated functions to DSPy format.

    There is no implicit tool registry: whatever is passed here is exactly what
    the agent can call.

    Args:
        tools: Either a list of @tool-decorated functions, or a
            {name: function} mapping.

    Returns:
        List of DSPy-compatible tool functions

    Raises:
        ValueError: If tools format is invalid

    Example:
        from tools.cgm_parser.main import cgm_parser
        from tools.cgm_metrics.main import cgm_metrics

        dspy_tools = convert_tools_for_dspy([cgm_parser, cgm_metrics])
        agent = ReAct(signature="question -> answer", tools=dspy_tools, max_iters=10)
    """
    import warnings

    # Case 1: List of decorated functions (direct, NO TU)
    if isinstance(tools, list) and tools and callable(tools[0]):
        dspy_tools = []
        for func in tools:
            if not hasattr(func, '__tool_name__'):
                warnings.warn(
                    f"Function '{func.__name__}' is not decorated with @tool, skipping"
                )
                continue
            try:
                dspy_tool = to_dspy_tool(func)
                dspy_tools.append(dspy_tool)
            except Exception as e:
                warnings.warn(f"Failed to convert tool '{func.__name__}': {e}")
        return dspy_tools

    # Case 2: Dict registry (custom, NO TU)
    if isinstance(tools, dict):
        dspy_tools = []
        for name, func in tools.items():
            if not callable(func):
                warnings.warn(f"Tool '{name}' is not callable, skipping")
                continue
            if not hasattr(func, '__tool_name__'):
                warnings.warn(
                    f"Function '{name}' is not decorated with @tool, skipping"
                )
                continue
            try:
                dspy_tool = to_dspy_tool(func)
                dspy_tools.append(dspy_tool)
            except Exception as e:
                warnings.warn(f"Failed to convert tool '{name}': {e}")
        return dspy_tools

    # Invalid format
    raise ValueError(
        f"Invalid tools format: {type(tools)}. Expected None, List[str], "
        f"List[Callable], or Dict[str, Callable]"
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from typing import Literal

    # Example metadata model
    class ExampleMetadata(BaseModel):
        """Results from example tool."""
        count: int = Field(..., description="Number of items processed")
        score: float = Field(..., ge=0, le=1, description="Quality score")

    # Example 1: Using Annotated (preferred)
    @tool(version="1.0.0", categories=["Example"])
    def example_tool(
        input_data_id: Annotated[str, Field(description="Artifact ID of input data")],
        *,
        mode: Annotated[Literal["fast", "accurate"], Field(description="Processing mode")] = "fast",
        threshold: Annotated[float, Field(ge=0, le=1, description="Detection threshold")] = 0.5,
        include_details: bool = True,
    ) -> ExampleMetadata:
        """Process data using example algorithm."""
        # Fake processing
        count = 42 if mode == "accurate" else 10
        score = threshold * 1.5 if threshold < 0.7 else 0.95

        return ExampleMetadata(count=count, score=score)

    # Example 2: Using docstring fallback (also works)
    @tool(version="1.0.0", categories=["Example"])
    def example_tool_docstring(
        input_data_id: str,
        *,
        mode: Literal["fast", "accurate"] = "fast",
        threshold: float = 0.5,
    ) -> ExampleMetadata:
        """Process data using example algorithm.

        Args:
            input_data_id: Artifact ID of input data
            mode: Processing mode (fast or accurate)
            threshold: Detection threshold (0-1)
        """
        count = 42 if mode == "accurate" else 10
        score = threshold * 1.5 if threshold < 0.7 else 0.95
        return ExampleMetadata(count=count, score=score)

    # Test direct call
    result = example_tool("data123", mode="accurate", threshold=0.8)
    print(f"Direct call: {result}")
    print(f"  count={result.count}, score={result.score}")

    # Test invoke() method (registry-agnostic, returns full envelope)
    result_via_invoke = example_tool.invoke({"input_data_id": "data123", "mode": "fast"})
    print(f"\nVia invoke(): {result_via_invoke.status}, metadata: {result_via_invoke.metadata}")

    # Test ainvoke() if available (for async contexts)
    if hasattr(example_tool, 'ainvoke'):
        print(f"ainvoke available: Yes (use in async contexts to avoid anyio.run issues)")

    # Get schemas
    input_model, output_model = get_tool_schemas(example_tool)
    print(f"\nInput model: {input_model.__name__}")
    print(f"Output model: {output_model.__name__}")

    # Test wrapper (dict → dict)
    wrapper = get_tool_wrapper(example_tool)
    if wrapper:
        result_dict = wrapper({"input_data_id": "data123", "threshold": 0.9})
        print(f"\nWrapper result: {result_dict['status']}, summary: {result_dict['summary']}")

    # Test name sanitization
    @tool(version="1.0.0", name="test-tool-with-hyphens")
    def test_tool(x: str) -> ExampleMetadata:
        """Test sanitization."""
        return ExampleMetadata(count=1, score=0.5)

    print(f"\nSanitized name: {test_tool.__tool_name__}")  # Should be 'test_tool_with_hyphens'

    

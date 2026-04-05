"""Project error taxonomy."""


class EngLLMError(Exception):
    """Base error type for EngLLM."""


class ConfigError(EngLLMError):
    """Raised for configuration loading/normalization issues."""


class RepositoryError(EngLLMError):
    """Raised for repository scanning problems."""


class GitError(EngLLMError):
    """Raised for git command and diff issues."""


class AnalysisError(EngLLMError):
    """Raised for deterministic analysis failures."""


class LLMError(EngLLMError):
    """Raised for provider and generation failures."""


class ValidationError(EngLLMError):
    """Raised for schema validation failures."""


class RenderingError(EngLLMError):
    """Raised for document/artifact rendering failures."""

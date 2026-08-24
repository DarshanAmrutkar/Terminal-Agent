"""Model Profiles & Registry for Terminal Agent.

Provides a dedicated, decoupled architecture for managing LLM profiles,
with simple functions to register, list, look up, and remove models.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ModelProfile(BaseModel):
    """Configuration profile for a specific LLM model and its provider."""
    name: str = Field(description="Short alias for the profile (e.g. 'sonnet', 'nemotron', 'deepseek')")
    provider: str = Field(description="LLM provider: 'anthropic', 'nvidia', 'openai', 'openrouter'")
    model_name: str = Field(description="Full model identifier passed to the API")
    base_url: Optional[str] = Field(default=None, description="Optional API base URL override")
    max_tokens: int = Field(default=8192, description="Max output tokens per response")
    context_window: int = Field(default=100000, description="Context window capacity in tokens")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    description: Optional[str] = Field(default=None, description="Human-friendly description")


# Default built-in presets
DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "sonnet": ModelProfile(
        name="sonnet",
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        max_tokens=8192,
        context_window=200000,
        description="Anthropic Claude Sonnet 4 — powerful reasoning & coding",
    ),
    "haiku": ModelProfile(
        name="haiku",
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        max_tokens=8192,
        context_window=200000,
        description="Anthropic Claude 3.5 Haiku — fast & cost-effective",
    ),
    "nemotron": ModelProfile(
        name="nemotron",
        provider="nvidia",
        model_name="nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://integrate.api.nvidia.com/v1",
        max_tokens=15000,
        context_window=500000,
        description="NVIDIA NIM Nemotron-3 Ultra 550B",
    ),
    "gpt4o": ModelProfile(
        name="gpt4o",
        provider="openai",
        model_name="gpt-4o",
        base_url="https://api.openai.com/v1",
        max_tokens=8192,
        context_window=128000,
        description="OpenAI GPT-4o flagship model",
    ),
    "gpt4o-mini": ModelProfile(
        name="gpt4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        max_tokens=8192,
        context_window=128000,
        description="OpenAI GPT-4o Mini lightweight model",
    ),
    "deepseek": ModelProfile(
        name="deepseek",
        provider="openrouter",
        model_name="deepseek/deepseek-chat",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=8192,
        context_window=64000,
        description="OpenRouter DeepSeek V3",
    ),
    "deepseek-r1": ModelProfile(
        name="deepseek-r1",
        provider="openrouter",
        model_name="deepseek/deepseek-r1",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=8192,
        context_window=64000,
        description="OpenRouter DeepSeek R1 reasoning model",
    ),
    "openrouter-claude": ModelProfile(
        name="openrouter-claude",
        provider="openrouter",
        model_name="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=8192,
        context_window=200000,
        description="OpenRouter Anthropic Claude 3.5 Sonnet",
    ),
    "llama": ModelProfile(
        name="llama",
        provider="openrouter",
        model_name="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=8192,
        context_window=128000,
        description="OpenRouter Meta Llama 3.3 70B Instruct",
    ),
}


class ProfileRegistry:
    """Registry that manages available ModelProfile presets."""

    def __init__(self, populate_defaults: bool = True) -> None:
        self._profiles: dict[str, ModelProfile] = {}
        if populate_defaults:
            self.reset_defaults()

    def register(self, profile: ModelProfile) -> None:
        """Register or update a ModelProfile."""
        self._profiles[profile.name.lower().strip()] = profile

    def unregister(self, name: str) -> None:
        """Remove a profile from the registry."""
        key = name.lower().strip()
        if key in self._profiles:
            del self._profiles[key]

    def get(self, name: str) -> ModelProfile:
        """Retrieve a profile by its name alias.
        
        Raises:
            KeyError: If profile is not found in registry.
        """
        key = name.lower().strip()
        if key not in self._profiles:
            available = ", ".join(sorted(self._profiles.keys()))
            raise KeyError(f"Unknown profile '{name}'. Available profiles: {available}")
        return self._profiles[key]

    def has(self, name: str) -> bool:
        """Check if a profile name exists in the registry."""
        return name.lower().strip() in self._profiles

    def list_all(self) -> dict[str, ModelProfile]:
        """Return a copy of all registered profiles."""
        return dict(self._profiles)

    def reset_defaults(self) -> None:
        """Reset the registry to default built-in presets."""
        self._profiles.clear()
        for name, profile in DEFAULT_PROFILES.items():
            self._profiles[name] = profile.model_copy()


# Global module-level registry instance
profile_registry = ProfileRegistry(populate_defaults=True)


def register_profile(
    name_or_profile: str | ModelProfile | None = None,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 8192,
    context_window: int = 100000,
    temperature: float = 0.0,
    description: Optional[str] = None,
) -> ModelProfile:
    """Easily register a new model profile.
    
    Can be called with a ModelProfile instance:
        register_profile(ModelProfile(name="mistral", provider="openrouter", model_name="mistralai/mistral-large"))
        
    Or with keyword/positional arguments:
        register_profile(name="mistral", provider="openrouter", model_name="mistralai/mistral-large")
    """
    if isinstance(name_or_profile, ModelProfile):
        profile = name_or_profile
    else:
        profile_name = name or (name_or_profile if isinstance(name_or_profile, str) else None)
        if not profile_name:
            raise ValueError("Profile name is required when registering a profile")
        if not provider or not model_name:
            raise ValueError("provider and model_name are required when registering a profile by name")
        profile = ModelProfile(
            name=profile_name,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            context_window=context_window,
            temperature=temperature,
            description=description,
        )
    profile_registry.register(profile)
    return profile


def remove_profile(name: str) -> None:
    """Remove a profile from the registry."""
    profile_registry.unregister(name)


def get_profile(name: str) -> ModelProfile:
    """Retrieve a profile from the registry by name."""
    return profile_registry.get(name)


def list_profiles() -> dict[str, ModelProfile]:
    """List all registered model profiles."""
    return profile_registry.list_all()

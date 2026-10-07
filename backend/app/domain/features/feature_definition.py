from dataclasses import dataclass, field
from typing import Any

from app.domain.features.feature_category import FeatureCategory


@dataclass
class FeatureDefinition:
    name: str
    category: FeatureCategory
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name.strip().lower()
        if not self.name:
            raise ValueError("Feature name cannot be empty")

        self.description = self.description.strip()
        if not self.description:
            raise ValueError("Feature description cannot be empty")

        if isinstance(self.category, str):
            try:
                self.category = FeatureCategory(self.category.strip().lower())
            except ValueError:
                allowed = [c.value for c in FeatureCategory]
                raise ValueError(
                    f"Invalid feature category: {self.category!r}. "
                    f"Must be one of: {allowed}"
                ) from None
        elif not isinstance(self.category, FeatureCategory):
            raise ValueError(f"Invalid feature category: {self.category!r}")

        if not isinstance(self.parameters, dict):
            raise TypeError("Feature parameters must be a dictionary")

        self.parameters = dict(self.parameters)

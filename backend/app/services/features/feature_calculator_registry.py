from app.domain.features.feature_calculator import FeatureCalculator


class FeatureCalculatorRegistry:
    def __init__(self) -> None:
        self._calculators: dict[str, FeatureCalculator] = {}

    def register(self, calculator: FeatureCalculator) -> None:
        if not isinstance(calculator, FeatureCalculator):
            raise TypeError(
                f"calculator must be a FeatureCalculator instance, got {type(calculator).__name__}"
            )

        name = calculator.feature_name.strip().lower()
        if not name:
            raise ValueError("Calculator feature_name cannot be empty")

        if name in self._calculators:
            raise ValueError(
                f"Calculator already registered for feature family: '{name}'"
            )

        self._calculators[name] = calculator

    def get(self, feature_name: str) -> FeatureCalculator:
        if not isinstance(feature_name, str):
            raise TypeError(
                f"feature_name must be a string, got {type(feature_name).__name__}"
            )

        key = feature_name.strip().lower()
        if not key:
            raise ValueError("Feature name cannot be empty")

        if key not in self._calculators:
            raise KeyError(
                f"No calculator registered for feature family: '{key}'"
            )

        return self._calculators[key]

    def __contains__(self, feature_name: str) -> bool:
        if not isinstance(feature_name, str):
            return False
        return feature_name.strip().lower() in self._calculators

    def __len__(self) -> int:
        return len(self._calculators)

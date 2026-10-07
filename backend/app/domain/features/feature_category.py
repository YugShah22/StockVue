from enum import StrEnum


class FeatureCategory(StrEnum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MARKET = "market"
    SENTIMENT = "sentiment"
    MACRO = "macro"

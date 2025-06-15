"""Conftest module."""

import platform

import pytest

from default_detection import PROJECT_DIR

MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)  # noqa

# To make the TRACKING_URI  path compatible for both macOS and Windows
if platform.system() == "Windows":
    TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"
else:
    TRACKING_URI = f"file://{MLRUNS_DIR.as_posix()}"


@pytest.fixture(scope="session")
def expected_feature_names() -> list[str]:
    """Return a list of expected feature names for the default detection model."""
    return [
        # Categorical features (X2, X3, X4)
        "cat__X2_1",
        "cat__X2_2",
        "cat__X3_1",
        "cat__X3_2",
        "cat__X3_3",
        "cat__X3_5",  # Based on unique values in train_set.csv
        "cat__X4_1",
        "cat__X4_2",
        "cat__X4_3",
        # Numerical features (passthrough)
        "remainder__X1",
        "remainder__X5",
        "remainder__X6",
        "remainder__X7",
        "remainder__X8",
        "remainder__X9",
        "remainder__X10",
        "remainder__X11",
        "remainder__X12",
        "remainder__X13",
        "remainder__X14",
        "remainder__X15",
        "remainder__X16",
        "remainder__X17",
        "remainder__X18",
        "remainder__X19",
        "remainder__X20",
        "remainder__X21",
        "remainder__X22",
        "remainder__X23",
    ]


pytest_plugins = ["tests.fixtures.datapreprocessor_fixture", "tests.fixtures.model_fixture"]

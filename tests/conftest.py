"""Configuration pytest commune — AeroPredict.

Rend les modules du projet importables depuis le dossier tests/ et injecte
des modules factices (aerosandbox, tensorflow) lorsque ces dépendances
lourdes ne sont pas installées. Les tests ne portent que sur la logique
pure (génération de grilles, filtrage physique, prétraitement), jamais
sur les simulations XFoil ni l'entraînement du réseau.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Dossier racine du projet (parent de tests/) sur le sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _module_factice(name: str) -> types.ModuleType:
    """Crée un module Python vide enregistrable dans sys.modules."""
    return types.ModuleType(name)


# ── aerosandbox factice (si absent) ──────────────────────────────
try:
    import aerosandbox  # noqa: F401
except ImportError:
    fake_asb = _module_factice("aerosandbox")
    fake_asb.Airfoil = MagicMock(name="Airfoil")
    fake_asb.XFoil = MagicMock(name="XFoil")
    # data_profils._uiuc_profiles utilise pathlib.Path(asb.__file__)
    fake_asb.__file__ = str(ROOT / "tests" / "conftest.py")
    sys.modules["aerosandbox"] = fake_asb

# ── tensorflow / keras factices (si absents) ─────────────────────
try:
    import tensorflow  # noqa: F401
except ImportError:
    keras_mock = MagicMock(name="keras")
    fake_tf = _module_factice("tensorflow")
    fake_tf.keras = keras_mock
    fake_tf.random = MagicMock(name="tf.random")
    sys.modules["tensorflow"] = fake_tf
    sys.modules["tensorflow.keras"] = keras_mock

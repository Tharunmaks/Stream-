"""Flash engine: NumPy-only (runs in Termux). Small dense core in RAM + hashed experts on disk."""
from .engine import FlashModel, create_model, train, sample

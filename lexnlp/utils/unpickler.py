import pickle
import warnings
import joblib


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "sklearn.tree.tree":
            renamed_module = "sklearn.tree"
        if module == "sklearn.ensemble.forest":
            renamed_module = "sklearn.ensemble._forest"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def safe_joblib_load(file_path, suppress_warnings=False):
    """
    Load a pickle file using joblib with compatibility fixes for scikit-learn version mismatches.
    
    Args:
        file_path: Path to the pickle file to load
        suppress_warnings: If True, suppress scikit-learn version mismatch warnings.
                          If False, show warnings (default) to encourage migration.
        
    Returns:
        The loaded object, or None if loading fails
    """
    try:
        # Try using compatibility layer first for tree-based models
        from lexnlp.utils.sklearn_compat import load_compatible
        try:
            return load_compatible(file_path, suppress_warnings=suppress_warnings)
        except (ValueError, TypeError, AttributeError):
            # Fall back to normal loading
            pass
        
        # Normal loading
        if suppress_warnings:
            # Suppress all scikit-learn version mismatch warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, 
                                       message='.*Trying to unpickle.*version.*')
                warnings.filterwarnings('ignore', category=UserWarning,
                                       module='sklearn.base')
                warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
                warnings.filterwarnings('ignore', message='.*node array from the pickle.*')
                warnings.filterwarnings('ignore', message='.*incompatible dtype.*')
                return joblib.load(file_path)
        else:
            # Show warnings to encourage migration
            return joblib.load(file_path)
    except (ValueError, TypeError, AttributeError) as e:
        # Return None for version mismatch errors - models will work without ML classifier
        if not suppress_warnings:
            warnings.warn(
                f"Could not load model from {file_path} due to scikit-learn version mismatch: {e}. "
                "Run 'python scripts/migrate_sklearn_models.py' to update the model files.",
                UserWarning
            )
        return None

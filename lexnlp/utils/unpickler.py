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


def safe_joblib_load(file_path):
    """
    Load a pickle file using joblib with scikit-learn version mismatch warnings suppressed.
    
    Args:
        file_path: Path to the pickle file to load
        
    Returns:
        The loaded object, or None if loading fails
    """
    try:
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
    except (ValueError, TypeError, AttributeError) as e:
        # Silently return None for version mismatch errors - models will work without ML classifier
        return None

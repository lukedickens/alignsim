import pathlib

def get_results_dir(folder_name: str = "plots"):
    """
    Returns the path to the results directory, creating it if it doesn't exist.
    """
    # Locates the directory where utils.py is, then goes up one level to the root
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    
    results_path = project_root / "results" / folder_name
    
    # Create the directory (and any parent folders) if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)
    
    return results_path

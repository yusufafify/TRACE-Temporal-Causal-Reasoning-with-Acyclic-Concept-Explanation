import pickle

def read_pkl(file_path):
    """
    Read and return the contents of a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        The unpickled object
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    file_path = "outputs/2026-03-03/11-31-38/results/confusion_matrix.pkl"
    data = read_pkl(file_path)
    if data is not None:
        print(data)
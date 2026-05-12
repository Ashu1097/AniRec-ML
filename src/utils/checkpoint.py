import pickle


def save_checkpoint(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)
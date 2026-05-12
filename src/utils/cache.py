class DiskCache:
    def __init__(self, *args, **kwargs):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        if value is None:
            raise ValueError("None values are not allowed")
    
        self.store[key] = value

    def clear(self):
        self.store.clear()

    def exists(self, key):
        return key in self.store

    def delete(self, key):
        self.store.pop(key, None)
class DiskCache:
    def __init__(self, *args, **kwargs):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value

    def clear(self):
        self.store.clear()
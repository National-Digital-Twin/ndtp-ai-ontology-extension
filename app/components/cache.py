import os
import pickle
from typing import Optional


class Cache:
    def __init__(self, use_cache: bool = False, cache_file: Optional[str] = None):
        self.use_cache = use_cache
        self.cache_file = cache_file

    def get_from_cache(self, key):
        if self.use_cache and os.path.exists(self.cache_file):
            cache = pickle.load(open(self.cache_file, "rb"))
            return cache.get(key, None)
        return None

    def save_to_cache(self, key, value):
        if not self.use_cache:
            return
        cache = {}
        if os.path.exists(self.cache_file):
            cache = pickle.load(open(self.cache_file, "rb"))

        cache[key] = value
        pickle.dump(cache, open(self.cache_file, "wb"))

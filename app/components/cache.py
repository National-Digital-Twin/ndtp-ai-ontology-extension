import os
import pickle

USE_CACHE = True
CACHE_FILE = "data/cache.pkl"


def get_from_cache(key):
    if USE_CACHE and os.path.exists(CACHE_FILE):
        cache = pickle.load(open(CACHE_FILE, "rb"))
        return cache.get(key, None)
    return None


def save_to_cache(key, value):
    if not USE_CACHE:
        return
    cache = {}
    if os.path.exists(CACHE_FILE):
        cache = pickle.load(open(CACHE_FILE, "rb"))
    cache[key] = value
    pickle.dump(cache, open(CACHE_FILE, "wb"))

# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

import os
import pickle
from typing import Optional
import streamlit as st


class Cache:
    """Cache class for caching results of long-running operations."""

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


def initialise_cache(
    use_cache: bool = False, cache_path: Optional[str] = None
) -> Cache:
    """Initialise the cache in Streamlit session state.

    Args:
        use_cache (bool, optional): Whether to use cache. Defaults to False.
        cache_path (Optional[str], optional): Path to the cache file. Defaults to None.
    """
    st.session_state.cache = Cache(use_cache=use_cache, cache_file=cache_path)


def get_cache() -> Cache:
    """Get the cache from Streamlit session state.

    Returns:
        Cache: The cache object.
    """
    if "cache" not in st.session_state:
        initialise_cache()
    return st.session_state.cache

"""Radix-Tree Global Prefix Caching for Apple Silicon Unified Memory.

Implements a hierarchical prefix trie storing KV cache states across independent
requests and client sessions (vLLM / SGLang architecture).

Features:
- Longest Common Prefix matching across global prompts and system instructions.
- Zero-copy tensor re-use for identical prefixes.
- Automatic Least-Recently-Used (LRU) node eviction under memory budgets.
- Thread-safe operations with telemetry counters.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence


class RadixNode:
    """A node in the Radix Tree representing a contiguous sub-sequence of tokens."""

    def __init__(
        self,
        tokens: tuple[int, ...],
        parent: RadixNode | None = None,
        cache_state: Any | None = None,
    ) -> None:
        self.tokens = tokens
        self.parent = parent
        self.cache_state = cache_state
        self.children: dict[int, RadixNode] = {}  # Keyed by the first token of the child branch
        self.last_accessed: float = time.time()
        self.total_tokens: int = (parent.total_tokens + len(tokens)) if parent else len(tokens)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixCache:
    """Global Radix Tree KV-Cache with LRU eviction."""

    def __init__(self, max_tokens: int = 8192) -> None:
        self.max_tokens = max_tokens
        self.root = RadixNode(tokens=(), parent=None, cache_state=None)
        self.lock = threading.Lock()
        self.total_cached_tokens = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    def match_prefix(
        self, token_ids: Sequence[int]
    ) -> tuple[int, Any | None, list[RadixNode]]:
        """Find the longest matching prefix for the given token sequence.

        Returns:
            (matched_token_count, deepest_cache_state, path_nodes)
        """
        with self.lock:
            if not token_ids:
                return 0, None, []

            curr = self.root
            idx = 0
            matched_nodes: list[RadixNode] = []
            last_valid_state = None
            last_valid_len = 0

            while idx < len(token_ids):
                first_tok = token_ids[idx]
                child = curr.children.get(first_tok)
                if child is None:
                    break

                # Compare child tokens with remaining input tokens
                child_toks = child.tokens
                match_len = 0
                for i in range(min(len(child_toks), len(token_ids) - idx)):
                    if child_toks[i] == token_ids[idx + i]:
                        match_len += 1
                    else:
                        break

                if match_len == len(child_toks):
                    # Full child node matched, descend further
                    idx += match_len
                    curr = child
                    curr.last_accessed = time.time()
                    matched_nodes.append(curr)
                    if curr.cache_state is not None:
                        last_valid_state = curr.cache_state
                        last_valid_len = curr.total_tokens
                else:
                    # Partial match within child node; cannot use incomplete child state safely
                    break

            if last_valid_len > 0:
                self.hits += 1
                self.tokens_saved += last_valid_len
            else:
                self.misses += 1

            return last_valid_len, last_valid_state, matched_nodes

    def insert(
        self, token_ids: Sequence[int], cache_state: Any
    ) -> None:
        """Insert a token sequence and its terminal KV cache state into the Radix Tree."""
        if not token_ids or cache_state is None:
            return

        with self.lock:
            curr = self.root
            idx = 0
            tokens_tuple = tuple(int(t) for t in token_ids)

            while idx < len(tokens_tuple):
                first_tok = tokens_tuple[idx]
                child = curr.children.get(first_tok)

                if child is None:
                    # Create a new branch with all remaining tokens
                    new_node = RadixNode(
                        tokens=tokens_tuple[idx:],
                        parent=curr,
                        cache_state=cache_state,
                    )
                    curr.children[first_tok] = new_node
                    self.total_cached_tokens += len(new_node.tokens)
                    self._check_eviction()
                    return

                # Existing child: find common prefix length
                child_toks = child.tokens
                match_len = 0
                for i in range(min(len(child_toks), len(tokens_tuple) - idx)):
                    if child_toks[i] == tokens_tuple[idx + i]:
                        match_len += 1
                    else:
                        break

                if match_len == len(child_toks):
                    # Entire child matches, advance down
                    idx += match_len
                    curr = child
                    curr.last_accessed = time.time()
                    if idx == len(tokens_tuple):
                        # Exact match on existing node, update state
                        curr.cache_state = cache_state
                        return
                else:
                    # Split child node at match_len
                    split_prefix = child_toks[:match_len]
                    split_suffix = child_toks[match_len:]

                    # 1. Create intermediate node
                    intermediate = RadixNode(
                        tokens=split_prefix,
                        parent=curr,
                        cache_state=None,
                    )
                    curr.children[first_tok] = intermediate

                    # 2. Re-attach the old child with its suffix
                    child.tokens = split_suffix
                    child.parent = intermediate
                    child.total_tokens = intermediate.total_tokens + len(split_suffix)
                    intermediate.children[split_suffix[0]] = child

                    # 3. If new tokens remain, attach new sibling
                    remaining_tokens = tokens_tuple[idx + match_len:]
                    if remaining_tokens:
                        new_branch = RadixNode(
                            tokens=remaining_tokens,
                            parent=intermediate,
                            cache_state=cache_state,
                        )
                        intermediate.children[remaining_tokens[0]] = new_branch
                        self.total_cached_tokens += len(remaining_tokens)
                    else:
                        intermediate.cache_state = cache_state

                    self._check_eviction()
                    return

    def _check_eviction(self) -> None:
        """Evict oldest leaf nodes if cached token limit is exceeded."""
        if self.total_cached_tokens <= self.max_tokens:
            return

        while self.total_cached_tokens > self.max_tokens:
            # Find oldest leaf node
            oldest_leaf = self._find_oldest_leaf(self.root)
            if oldest_leaf is None or oldest_leaf is self.root:
                break

            parent = oldest_leaf.parent
            if parent:
                first_tok = oldest_leaf.tokens[0]
                if first_tok in parent.children:
                    del parent.children[first_tok]
                self.total_cached_tokens -= len(oldest_leaf.tokens)

    def _find_oldest_leaf(self, node: RadixNode) -> RadixNode | None:
        if node.is_leaf() and node is not self.root:
            return node

        oldest = None
        oldest_time = float("inf")
        for child in node.children.values():
            cand = self._find_oldest_leaf(child)
            if cand and cand.last_accessed < oldest_time:
                oldest_time = cand.last_accessed
                oldest = cand
        return oldest

    def clear(self) -> None:
        with self.lock:
            self.root = RadixNode(tokens=(), parent=None, cache_state=None)
            self.total_cached_tokens = 0
            self.hits = 0
            self.misses = 0
            self.tokens_saved = 0


__all__ = ["RadixCache", "RadixNode"]

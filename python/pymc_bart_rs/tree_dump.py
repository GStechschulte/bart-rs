from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


class TreeDump:
    """Lightweight, serializable tree dump with a predict method."""

    def __init__(
        self,
        split_feature: Iterable[int],
        split_value: Iterable[float],
        left_child: Iterable[int],
        right_child: Iterable[int],
        leaf_value: Iterable[float],
        n_left: Optional[Iterable[int]] = None,
        n_right: Optional[Iterable[int]] = None,
        root_index: int = 0,
    ) -> None:
        self.split_feature = [int(v) for v in split_feature]
        self.split_value = [float(v) for v in split_value]
        self.left_child = [int(v) for v in left_child]
        self.right_child = [int(v) for v in right_child]
        self.leaf_value = [float(v) for v in leaf_value]
        self.n_left = [int(v) for v in n_left] if n_left is not None else None
        self.n_right = [int(v) for v in n_right] if n_right is not None else None
        self.root_index = int(root_index)

    @classmethod
    def from_rust(cls, dump: object) -> "TreeDump":
        if isinstance(dump, dict):
            return cls(
                split_feature=dump["split_feature"],
                split_value=dump["split_value"],
                left_child=dump["left_child"],
                right_child=dump["right_child"],
                leaf_value=dump["leaf_value"],
                n_left=dump.get("n_left"),
                n_right=dump.get("n_right"),
                root_index=dump.get("root_index", 0),
            )

        return cls(
            split_feature=getattr(dump, "split_feature"),
            split_value=getattr(dump, "split_value"),
            left_child=getattr(dump, "left_child"),
            right_child=getattr(dump, "right_child"),
            leaf_value=getattr(dump, "leaf_value"),
            n_left=getattr(dump, "n_left", None),
            n_right=getattr(dump, "n_right", None),
            root_index=getattr(dump, "root_index", 0),
        )

    def predict(
        self,
        x: np.ndarray,
        excluded: Optional[Sequence[int]] = None,
        shape: int = 1,
    ) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features).")

        excluded_set = set(excluded) if excluded is not None else None
        preds = np.zeros(x.shape[0], dtype=float)
        for idx, sample in enumerate(x):
            preds[idx] = self._predict_node(self.root_index, sample, excluded_set)

        if shape == 1:
            return preds[None, :]

        return np.tile(preds, (shape, 1))

    def _predict_node(
        self,
        node: int,
        sample: np.ndarray,
        excluded: Optional[set[int]],
    ) -> float:
        if node < 0:
            return 0.0

        split_feature = self.split_feature[node]
        if split_feature < 0:
            return self.leaf_value[node]

        if excluded is not None and split_feature in excluded:
            left_idx = self.left_child[node]
            right_idx = self.right_child[node]
            left_val = self._predict_node(left_idx, sample, excluded)
            right_val = self._predict_node(right_idx, sample, excluded)
            if self.n_left is not None and self.n_right is not None:
                n_left = self.n_left[node]
                n_right = self.n_right[node]
                total = n_left + n_right
                if total > 0:
                    weight = n_left / total
                    return weight * left_val + (1.0 - weight) * right_val
            return 0.5 * (left_val + right_val)

        threshold = self.split_value[node]
        if sample[split_feature] < threshold:
            return self._predict_node(self.left_child[node], sample, excluded)
        return self._predict_node(self.right_child[node], sample, excluded)

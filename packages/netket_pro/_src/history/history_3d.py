from typing import Any, Callable
from netket.utils.history import History
import numpy as np
from netket.utils.types import Array, DType

from numbers import Number

from netket.utils.numbers import is_scalar
from netket.utils.history.accum import (
    AccumulatorFunTypeRegistry,
    init_history_from_data,
)

try:
    from netket.utils.history.history_dict import (
        register_historydict_deserialization_fun,
    )
    from netket.utils.history.history import replace_none_with_nan
except ImportError:

    def register_historydict_deserialization_fun(*args, **kwargs):
        pass

    def replace_none_with_nan(item):
        return item


from netket_pro._src.history.history_2d import History2D, compute_inner_length


def _safe_list(val):
    if not hasattr(val, "__len__"):
        return [val]
    return list(val)


class History3D:
    def __init__(
        self,
        values: History2D | list[History2D],
        iters: list | None = None,
        iter_dtype: DType | None = None,
    ):
        if iters is None:
            iters = [0]

        if is_scalar(iters):
            iters = np.array([iters], dtype=iter_dtype)
        elif isinstance(iters, list):
            iters = np.array(iters, dtype=iter_dtype)

        other_values = ((), ())
        if not isinstance(values, History2D):
            if isinstance(values, (list, tuple)):
                if not len(values) == len(iters):
                    raise ValueError("Length Mismatch between values and iters")
                other_values = (iters[1:], values[1:])
                values = values[0]
                iters = iters[:1]

            else:
                raise TypeError(
                    "values should be a History object or a list of History objects"
                )

        value_dict = values.to_dict().copy()
        for k, v in value_dict.items():
            value_dict[k] = np.expand_dims(np.array(v), 0)

        self._iters_2d = value_dict.pop("axis")

        self._value_dict = value_dict
        self._value_name = values._value_name
        self._single_value = values._single_value
        self._iters = iters
        self._lengths_inner = np.array([values._lengths_inner])
        # self._lengths_inner = [len(values)]

        for i, v in zip(*other_values):
            self.append(v, it=i)

    @classmethod
    def from_serialized_data(
        cls,
        value_dict: dict,
        iters: np.ndarray = None,
        iters_2d: np.ndarray = None,
        *,
        value_name: str | None = None,
        lengths_inner: list[int] | None = None,
    ):

        self = cls.__new__(cls)
        value_dict = value_dict.copy()

        if "axis0" in value_dict:
            _val = value_dict.pop("axis0")
            if iters is None:
                iters = _val
        elif iters is None:
            raise IOError(
                "You are loading some older version of a checkpoint. We can fix it but let me know"
            )
        if "axis1" in value_dict:
            _val = value_dict.pop("axis1")
            if iters_2d is None:
                iters_2d = _val
        elif iters_2d is None:
            raise IOError(
                "You are loading some older version of a checkpoint. We can fix it but let me know"
            )

        # nans are stored as None in json, so we need to replace them with np.nan when loading
        for k, v in value_dict.items():
            if isinstance(v, list):
                value_dict[k] = np.array(replace_none_with_nan(v))

        self._value_dict = value_dict
        self._iters = np.array(iters)
        self._iters_2d = np.array(iters_2d)

        if lengths_inner is None:
            lengths_inner = compute_inner_length(value_dict["iters"], axis=2)
        self._lengths_inner = np.array(lengths_inner, dtype=np.int64)

        # If not set, we can guess if there is a single value
        # and in that case load the single value name.
        if len(set(self.keys()) - {"axis0", "axis1", "iters"}) == 1:
            # TODO: maybe we should serialize this?
            self._single_value = True
            if value_name is None:
                value_name = list(set(self.keys()) - {"axis0", "axis1", "iters"})[0]
        else:
            self._single_value = False
        self._value_name = value_name

        return self

    @property
    def iters(self) -> Array:
        return self._iters

    @property
    def iters_inner(self) -> Array:
        return self._iters_2d

    @property
    def iters_inner_inner(self) -> Array:
        return self._value_dict["iters"]

    @property
    def values(self) -> Array:
        return self._value_dict[self._value_name]

    @property
    def main_value_name(self):
        return self._value_name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.iters_inner_inner.shape[:3]

    def to_dict(self) -> dict:
        result = self._value_dict.copy()
        result["axis0"] = self._iters
        result["axis1"] = self._iters_2d
        return result

    def __len__(self) -> int:
        return len(self.iters)

    def __getattr__(self, attr):
        # Allow users to access fields with . accessor patterns
        if attr in self._value_dict:
            return self._value_dict[attr]

        raise AttributeError

    def __iter__(self):
        return ((it, self[i]) for i, it in enumerate(self.iters))

    def __getitem__(self, key) -> Array:
        # if its an int corresponding to an element not inside the dict,
        # treat it as accessing a slice of a single element
        if isinstance(key, str):
            if key == "iters":
                return self.iters
            else:
                return self._value_dict[key]

        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1 and isinstance(key[0], int):
            return self._get_single(key[0])
        elif len(key) <= 0:
            raise ValueError("Must have at least 1 index.")
        elif len(key) <= 3:
            if isinstance(key[0], int):
                return self._get_single(key[0])[key[1:]]
            else:
                return self._get_slice(key)
        else:
            raise ValueError("Can index at most 3 dimensions in a History3D object.")

    def _get_slice(self, slce: slice) -> History:
        """
        get a slice of iterations from this history object
        """
        hist = History3D.__new__(History3D)
        if len(slce) == 1:
            hist._lengths_inner = self._lengths_inner[slce[0]]
        else:
            hist._lengths_inner = self._lengths_inner[slce[0], slce[1]]
        max_len = self.shape[1]
        max_inner_len = np.max(hist._lengths_inner)
        if len(slce) == 1:
            slce = (slce[0], slice(max_len), slice(max_inner_len))
        elif len(slce) == 2:
            slce1, slce2 = slce
            if isinstance(slce2, slice):
                if slce2.stop is None or slce2.stop > max_len:
                    slce2 = slice(slce2.start, max_len, slce2.step)
            else:
                # convert to slice the inner index, otherwise we break hist3d
                slce2 = slice(slce2, slce2 + 1)
            slce = (slce1, slce2)
        elif len(slce) == 3:
            slce1, slce2, slce3 = slce
            if isinstance(slce2, slice):
                if slce2.stop is None or slce2.stop > max_len:
                    slce2 = slice(slce2.start, max_len, slce2.step)
            else:
                # convert to slice the inner index, otherwise we break hist3d
                slce2 = slice(slce2, slce2 + 1)
            if isinstance(slce3, slice):
                if slce3.stop is None or slce3.stop > max_inner_len:
                    slce3 = slice(slce3.start, max_inner_len, slce3.step)
            else:
                # convert to slice the inner index, otherwise we break hist3d
                slce3 = slice(slce3, slce3 + 1)
            slce = (slce1, slce2, slce3)
        else:
            raise ValueError(f"wrong length {len(slce)}")

        values_sliced = {}
        for key in self._value_dict.keys():
            values_sliced[key] = self._value_dict[key][*slce]

        hist._value_dict = values_sliced
        hist._value_name = self._value_name
        hist._single_value = self._single_value
        hist._iters = self.iters[slce[0]]
        hist._iters_2d = self._iters_2d[slce[0], slce[1]]
        return hist

    def _get_single(self, i: slice) -> History2D:
        values_sliced = {}
        lengths_hist2d = self._lengths_inner[i]
        real_len = np.max(lengths_hist2d)

        # todo this should be adaptive
        real_len_2d = len(lengths_hist2d)
        for key in self.keys():
            values_sliced[key] = self._value_dict[key][i, :real_len_2d, :real_len]

        hist = History2D.__new__(History2D)
        # hist = History2D(values_sliced)
        hist._value_dict = values_sliced
        hist._value_name = self._value_name
        hist._single_value = self._single_value
        hist._iters = self._iters_2d[i]
        hist._lengths_inner = lengths_hist2d.tolist()
        return hist

    def __contains__(self, key: str) -> bool:
        return key in self._value_dict

    def keys(self) -> list:
        _keys = list(self._value_dict.keys())
        return _keys

    def append(
        self, val: History | dict, it: Number | None = None, it1: Number | None = None
    ):
        """
        Append another value to this history object.

        Args:
            val: the value in the next timestep
            it: the time corresponding to this new value. If
                not defined, increment by 1.
        """
        if it is None:
            it = self.iters[-1] + 1
        it0 = it
        if not isinstance(val, History2D):
            raise TypeError()

        old_inner_inner_length = self.shape[2]
        self._lengths_inner = np.concatenate(
            [self._lengths_inner, np.expand_dims(val._lengths_inner, 0)], axis=0
        )
        max_inner_inner_len = np.max(self._lengths_inner)

        self._iters_2d = np.concatenate(
            [self._iters_2d, np.expand_dims(np.array(val.iters), 0)], axis=0
        )

        # print("keys:", list(self.keys()))
        for key in self.keys():
            _vals = self._value_dict[key]

            new_shape = (
                len(_vals) + 1,
                _vals.shape[1],
                max_inner_inner_len,
            ) + _vals.shape[3:]
            if old_inner_inner_length < max_inner_inner_len:
                pad_shape = [(0, 0) for _ in range(_vals.ndim)]
                pad_shape[0] = (0, 1)  # add a new element
                pad_shape[1] = (0, 0)  # assume no change in second dimension
                pad_shape[2] = (0, max_inner_inner_len - old_inner_inner_length)
                if np.issubdtype(_vals.dtype, np.floating):
                    _vals = np.pad(
                        _vals, pad_shape, mode="constant", constant_values=np.nan
                    )
                else:
                    _vals = np.pad(_vals, pad_shape, mode="edge")
                self._value_dict[key] = _vals
            else:
                # try to resize in place the buffer so that we don't reallocate
                # and if we fail, resize tby reallocating to a new buffer.
                try:
                    _vals.resize(new_shape)
                except ValueError:
                    _vals = np.resize(_vals, new_shape)
                    self._value_dict[key] = _vals

            new_val = val._value_dict[key]
            if old_inner_inner_length > max_inner_inner_len:
                _vals[-1][: len(val)] = new_val
                if np.issubdtype(_vals.dtype, np.floating):
                    _vals[-1][len(val) :] = np.nan
                else:
                    _vals[-1][len(val) :] = new_val[-1]
            else:
                vshp = _vals.shape[1:]
                shp = new_val.shape
                # _vals[-1, : shp[0], : shp[1]] = new_val
                _vals[-1, :, : shp[1]] = new_val
                if shp[1] < vshp[1]:
                    if np.issubdtype(_vals.dtype, np.floating):
                        _vals[-1, :, shp[1] :] = np.nan
                    else:
                        _vals[-1, :, shp[1] :] = new_val[:, -1].reshape(-1, 1)

        try:
            self.iters.resize(len(self.iters) + 1)
        except ValueError:
            self._iters = np.resize(self.iters, (len(self.iters) + 1))
        self.iters[-1] = it0

    def __repr__(self):
        if len(self.iters) < 5:
            iters_repr = repr(self.iters)
        else:
            iters_repr = (
                f"[{self.iters[0]}, {self.iters[1]}, ..."
                f" {self.iters[-2]}, {self.iters[-1]}] "
                f"({len(self.iters)} steps)"
            )
        keys = list(set(self.keys()) - {"iters", "axis0", "axis1"})
        return (
            "History3D("
            + f"\n   keys  = {keys}, "
            + f"\n   shape  = {self.shape}, "
            + f"\n   iters = {iters_repr},"
            + "\n)"
        )

    def __str__(self):
        keys = list(set(self.keys()) - {"axis0", "axis1", "iters"})
        return f"History3d(keys={keys}, shape  = {self.shape},)"


def _accum_histories2d_history3d(
    fun: Callable[[Any, Any], Any],
    tree_accum: History,
    tree: History,
    **kwargs,
):
    return fun(tree_accum, tree, **kwargs)


AccumulatorFunTypeRegistry[History2D] = _accum_histories2d_history3d


@init_history_from_data.register
def init_history_from_data_history(val: History2D, step: Any):
    return History3D(val, step)


def is_history3d(hist_dict):
    return "axis0" in hist_dict and "axis1" in hist_dict


def reconstruct_history3d(hist_dict):
    axis0 = hist_dict.pop("axis0")
    axis1 = hist_dict.pop("axis1")
    value_name = "Mean" if "Mean" in hist_dict else None

    return History3D.from_serialized_data(
        hist_dict,
        axis0,
        axis1,
        value_name=value_name,
    )


register_historydict_deserialization_fun(
    is_history3d,
    reconstruct_history3d,
    precedence=10,
)

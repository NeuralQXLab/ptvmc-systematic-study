from netket.vqs import MCState, FullSumState

from netket_pro._src.monkeypatch.util import add_method

from nqxpack import save as _save, load as _load


@add_method(MCState)
def save(self, fileobj):
    """
    Save the Monte Carlo state to a file.

    The format is "netket format v1" which does not require the state to be
    created to load it. You just must have the packages installed.

    The format is a zip-compressed file with the following structure:
        - ``metadata.json``: The metadata of the state.
        - ``config.json``: The configuration of the state.
        - ``data``: a msgpack serialized file with the data of the state.

    You can manually unzip the file and inspect the contents.

    .. warning ::

        If you are using some libraries, such as flax, or your own packages,
        those must still be importable when you load the file.

        A good rule of thumb is to remember to register the ``pyproject.toml``
        file with all that you have installed when you save files!

    Args:
        fileobj: The file to save the state to. Will change the file extension
            to `.nk` if undeclared or different from that one.
    """

    _save({"state": self}, fileobj)


@add_method(FullSumState)
def save(self, fileobj):  # noqa: F811
    """
    Save the Monte Carlo state to a file.

    The format is "netket format v1" which does not require the state to be
    created to load it. You just must have the packages installed.

    The format is a zip-compressed file with the following structure:
        - ``metadata.json``: The metadata of the state.
        - ``config.json``: The configuration of the state.
        - ``data``: a msgpack serialized file with the data of the state.

    You can manually unzip the file and inspect the contents.

    .. warning ::

        If you are using some libraries, such as flax, or your own packages,
        those must still be importable when you load the file.

        A good rule of thumb is to remember to register the ``pyproject.toml``
        file with all that you have installed when you save files!

    Args:
        fileobj: The file to save the state to. Will change the file extension
            to `.nk` if undeclared or different from that one.
    """

    _save({"state": self}, fileobj)


@add_method(MCState)
@classmethod
def load(cls, path, new_seed: bool | int = True):
    """
    Load a Monte Carlo state from a file.

    .. warning ::

        If you used some libraries, such as flax, or your own packages,
        those must still be importable when you load the file.

        A good rule of thumb is to remember to register the ``pyproject.toml``
        file with all that you have installed when you save files!

    Args:
        path: The path to the file to load the state from.
        new_seed: If True, the seed of the state will be replaced by a new one.
            If an integer, the seed will be set to this value.
            If False, the seed will not be changed. (Default: ``True`)
    """
    vstate = _load(path)["state"]
    if new_seed is not False:
        if new_seed is True:
            new_seed = None
        vstate.replace_sampler_seed(new_seed)
    return vstate


@add_method(FullSumState)
@classmethod
def load(cls, path):  # noqa: F811
    """
    Load a Monte Carlo state from a file.

    .. warning ::

        If you used some libraries, such as flax, or your own packages,
        those must still be importable when you load the file.

        A good rule of thumb is to remember to register the ``pyproject.toml``
        file with all that you have installed when you save files!

    Args:
        path: The path to the file to load the state from.
    """

    vstate = _load(path)["state"]
    return vstate

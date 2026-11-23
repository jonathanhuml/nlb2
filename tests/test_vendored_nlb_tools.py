import pandas as pd
import pytest

from nlb2.datasets.nlb import _require_pandas_below_3


def test_bundled_nlb_tools_imports_without_external_package():
    from nlb2._vendor.nlb_tools import evaluation, make_tensors, nwb_interface

    assert callable(evaluation.evaluate)
    assert callable(make_tensors.make_train_input_tensors)
    assert nwb_interface.NWBDataset.__module__ == "nlb2._vendor.nlb_tools.nwb_interface"


def test_nlb_tensor_building_refuses_pandas_3(monkeypatch):
    monkeypatch.setattr(pd, "__version__", "3.0.0")
    with pytest.raises(RuntimeError, match=r"pandas<3.*\.\[nlb\]"):
        _require_pandas_below_3()


def test_nlb_tensor_building_allows_pandas_2(monkeypatch):
    monkeypatch.setattr(pd, "__version__", "2.3.3")
    _require_pandas_below_3()

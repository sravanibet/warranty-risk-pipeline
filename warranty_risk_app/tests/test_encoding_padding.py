import numpy as np
import pandas as pd
import pytest

from warranty_risk.preprocessing.encoding import fit_category_map, encode_values, UNSEEN_CODE
from warranty_risk.prediction.padding import pad_and_mask, pad_single_vehicle, compute_padding_meta
from warranty_risk.labels.features import build_feature_matrix


def _readings(vid, defect, n=3):
    rows = []
    for t in range(n):
        rows.append((vid, "ST1", "TORQUE", "scalar", t, 40.0 + t))
    rows.append((vid, "ST2", "DEFECT", "categorical", 0, defect))
    rows.append((vid, "ST2", "OK", "binary", 0, True))
    return pd.DataFrame(rows, columns=["serial_number", "station_uid", "sensor_uid",
                                       "sensor_type", "timestamp", "value"])


@pytest.fixture
def train_df():
    return pd.concat([_readings("A", "none"), _readings("B", "scratch"), _readings("C", "dent")])


def test_single_vehicle_uses_training_codes_not_its_own(train_df):
    cmap = fit_category_map(train_df)            # dent=0, none=1, scratch=2
    one = _readings("Z", "scratch")
    uids, L = compute_padding_meta(train_df)
    with_map = pad_single_vehicle(one, uids, L, cmap)
    without = pad_single_vehicle(one, uids, L)   # old behaviour: fits on this vehicle only
    assert with_map["DEFECT_t0"].iloc[0] == cmap["scratch"] == 2
    assert without["DEFECT_t0"].iloc[0] == 0     # the bug the map fixes


def test_unseen_category_gets_sentinel_code(train_df):
    cmap = fit_category_map(train_df)
    enc = encode_values(_readings("Z", "crack"), cmap)
    assert enc.loc[enc["sensor_uid"] == "DEFECT", "value"].iloc[0] == UNSEEN_CODE


def test_live_row_matches_training_row(train_df):
    cmap = fit_category_map(train_df)
    uids, L = compute_padding_meta(train_df)
    batch = pad_and_mask(train_df, cmap)
    single = pad_single_vehicle(train_df[train_df.serial_number == "B"], uids, L, cmap)
    assert list(single.columns) == list(batch.columns)
    np.testing.assert_array_equal(single.loc["B"].values, batch.loc["B"].values)


def test_short_vehicle_is_padded_and_masked(train_df):
    cmap = fit_category_map(train_df)
    uids, L = compute_padding_meta(train_df)
    short = _readings("S", "none", n=1)
    row = pad_single_vehicle(short, uids, L, cmap).loc["S"]
    assert row["TORQUE_mask_t0"] == 1 and row["TORQUE_mask_t1"] == 0 and row["TORQUE_t2"] == 0


def test_feature_matrix_uses_shared_map(train_df):
    cmap = fit_category_map(train_df)
    fm = build_feature_matrix(_readings("Z", "scratch"), cmap)
    assert fm.loc["Z", "DEFECT"] == 2

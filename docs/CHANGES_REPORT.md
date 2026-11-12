# Detailed change report: Noor's prototype vs. the current branch

Branch: `warranty-risk-updates` (pushed to `origin`). Baseline: Noor's last commit `f0439ee`.
This report covers the work done with Claude in this session, file by file. Work that was
already in the tree when the session began is listed separately so credit stays accurate.

Line counts are `git --numstat` (added / removed) for the commit named.

---

## 1. Commit map

| Commit | Date | Author | Scope | Written in this session? |
|---|---|---|---|---|
| `b31e38e`, `c488757`, `54a00bd`, `f0439ee` | 22-25 May 2026 | Noor E Jannat | Original prototype | No |
| `fe50c13` | 25 Sep 2026 | Sravani B | Quality Monitoring Agent, 4 phases | No (already present) |
| `1cd3b7b` | this session | Sravani B | Simulator moved to `plant_simulator_pkg`, deps, `.gitignore` | Committed only |
| `ad3416e` | this session | Sravani B | Live scoring, registry, validation, tests, **shared encoding** | Mixed: see section 2 |
| `29331c4` | this session | Sravani B | `Preprocessor` fit/transform | Yes |
| `d75afc8` | this session | Sravani B | README rewrite | Yes |
| `d6aa63e` | this session | Sravani B | Leakage-free evaluation | Yes |

---

## 2. Commit `ad3416e`: shared encoding, padding, registry preprocessing

Files in this commit that were already in the working tree (not written in this session, only
committed): `mqtt_listener.py`, `alerting.py`, `mock_publisher.py`, `stream_quality_agent.py`,
`validation.py` (first version), `app.py` additions, and the tests `test_config_validator`,
`test_data_validator`, `test_feedback`, `test_live_scoring`, `test_mqtt_listener`,
`test_orchestrator_gating`, `test_validation`, `conftest`.

Files below are the parts changed in this session.

### 2.1 `warranty_risk/preprocessing/encoding.py` (new, 41 lines)

| Item | Detail |
|---|---|
| Problem | `LabelEncoder` was fitted on whatever batch it was given. In live scoring that is one vehicle, so a category such as "scratch" got a different integer than at training. The model scored it without any error. |
| `fit_category_map(df)` | Returns `{category_string: code}` over all categorical readings, sorted so codes are deterministic. Fitted once at training. |
| `encode_values(df, category_map=None)` | Returns a copy with a numeric `value` column. Categorical readings go through the map (categories not in the map become `UNSEEN_CODE = -1`); binary readings are cast to int; everything else is coerced to numeric (unparseable becomes NaN, callers decide how to fill). If no map is passed it fits one on the data given, which reproduces the old behaviour for training. |
| Tests | `tests/test_encoding_padding.py` |

### 2.2 `warranty_risk/prediction/padding.py` (55 added / 79 removed)

| Change | Before | After |
|---|---|---|
| Encoding | `LabelEncoder` inlined in both `pad_and_mask` and `pad_single_vehicle` (copy-pasted) | One call to `encode_values` in both |
| Padding loop | Nested Python loops that filter the DataFrame per vehicle and per sensor | One vectorised core `_pad(...)`: sort, `groupby().cumcount()` for the time step, then a single NumPy assignment |
| `_column_names` | Column list built twice | One helper |
| `pad_and_mask(df, category_map=None)` | No map | Optional map |
| `pad_single_vehicle(..., category_map=None)` | Refit an encoder on the one vehicle | Uses the training map; sensors not seen at training are ignored |
| `build_padded_matrix(..., category_map=None)` | No map | Passes the map through |
| Verification | n/a | Output compared with the old implementation on 60 simulated vehicles: same shape, columns, index and values (`allclose`) |

### 2.3 `warranty_risk/labels/features.py` (4 added / 14 removed)

`build_feature_matrix(manufacturing_df, category_map=None)` now calls `encode_values` instead of
its own `LabelEncoder`, so the per-vehicle feature matrix uses the same codes as padding.

### 2.4 `warranty_risk/pipeline.py` (7 added / 3 removed in this commit)

| Line area | Change |
|---|---|
| Imports | `fit_category_map` |
| `__init__` | New attribute `trained_category_map` |
| `_stage_predict` | Fits the map from `manufacturing_df` and passes it to `build_padded_matrix` |
| `score_new_vehicle` | Passes the map to `build_feature_matrix` and `pad_single_vehicle` |

(The `score_new_vehicle` method and `trained_sensor_uids` / `trained_max_seq_len` existed already.)

### 2.4b `warranty_risk/agent/model_registry.py` (file was pre-existing; this session's edits)

| Change | Detail |
|---|---|
| `save(train_result, f1, preprocessing=None)` | New optional dict stored in `metadata.json` under `"preprocessing"` (category map, sensor list, max sequence length) |
| `ModelRecord.preprocessing` | New field, `None` for models saved before this change |
| `load()` | Fills `preprocessing` from metadata with `.get`, so old saves still load |
| Docstring | Explains why the map is stored |
| Tests | Two added to `tests/test_model_registry.py`: preprocessing round-trip, old save still loads |

### 2.4c `warranty_risk/agent/orchestrator.py` (this session's edit: about 9 lines in `learn`)

`model_registry.save(...)` now receives `preprocessing={"category_map", "sensor_uids", "max_seq_len"}`
taken from the pipeline.

### 2.5 `tests/test_encoding_padding.py` (new, 61 lines, 5 tests)

Live row equals training row; training codes beat self-fitted codes; unseen category gets `-1`;
a short vehicle is padded and masked correctly; feature matrix uses the shared map.

---

## 3. Commit `29331c4`: `Preprocessor`

| File | Lines | Detail |
|---|---|---|
| `preprocessing/preprocessor.py` | 171 / 162 (whole-file rewrite; the diff also reflects line-ending normalisation) | See below |
| `preprocessing/__init__.py` | 2 / 2 | Exports `Preprocessor` and `auto_preprocess` |
| `tests/test_preprocessor.py` (new) | 75 | 8 tests |

| Item | Before | After |
|---|---|---|
| Structure | One function `auto_preprocess(df, label_col)` that learned everything from the data it was given | Class `Preprocessor` with `fit`, `transform`, `fit_transform`; `auto_preprocess` is a wrapper |
| Imputation | Medians computed on the data being cleaned | Medians learned in `fit` (train data), reapplied in `transform` to any other rows |
| Dropped columns | Decided on the data being cleaned | Decided in `fit` (missing over 50%, non-numeric, constant); `transform` applies the same drop list and raises if a fitted column is missing |
| Label column over 50% missing | **Dropped as if it were a feature** (bug) | Never dropped; rows with a missing label are dropped and the report says so |
| Label coercion | `astype(int)` only | Also accepts yes/no/true/false/1/0; one-shot path warns on unreadable values, strict path raises |
| Duplicates | Removed everywhere | Removed only on request (`drop_duplicates=True`), so live rows are never dropped |
| Report | List of dicts with `step`, `detail`, `status` | Same steps and wording; compared with the old function on a messy 200-row frame: identical output and identical report |

---

## 4. Commit `d6aa63e`: leakage-free evaluation

### 4.1 `warranty_risk/prediction/models.py` (87 / 30)

| Function | Change |
|---|---|
| `_optimal_threshold` | Docstring: validation data only; returns 0.5 if the labels contain one class |
| `_safe_auc` (new) | AUC returns NaN instead of raising on a single-class split |
| `split_vehicles(y, val_size=0.2, test_size=0.2, random_state=42)` (new) | Disjoint, stratified train / validation / test by vehicle, built on integer positions (the first version failed on Arrow-backed pandas indexes) |
| `train_and_select(ml_data, test_size=0.2, random_state=42, split=None, val_size=0.2)` | Fits on train, picks model and threshold on validation, reports metrics on test. `results_df` gains `Val_F1` and is ranked by it; `F1`, `Precision`, `Recall`, `AUC` are test-set values. Each candidate is `clone`d so global model objects are no longer mutated. Returns extra keys `X_val`, `y_val`, `X_val_scaled`, `split`. |
| `compare_baseline_enhanced` | Threshold from validation, metrics on test |

### 4.2 `warranty_risk/optimization/ga_optimizer.py` (11 / 2)

New parameter `internal_holdout=0.2`. With `0` the GA uses every row it is handed (the caller has
already split). Docstring states the GA uses labels, so it must receive train vehicles only.

### 4.3 `warranty_risk/pipeline.py` (43 / 24)

| Change | Detail |
|---|---|
| `__init__` | New `seed=42`, new `split` attribute |
| `_stage_labels` | Passes `random_state=self.seed` to the label generator |
| `fit_tolerance(feature_df, train_idx)` (new) | Fits the GA on the train vehicles only, then computes tolerance features for all vehicles from those bounds. Returns `(ga_result, feature_df)` without touching state, so validation can repeat it. |
| `_stage_optimize` | Drops tolerance columns left by an earlier run (previously they could be treated as sensors on a second `learn()`); builds the split before the GA; calls `fit_tolerance` |
| `_stage_predict` | `train_and_select(ml_data, split=self.split)` |

### 4.4 `warranty_risk/agent/validation.py` (42 / 31)

Each seed now repeats the whole procedure: new vehicle split, GA re-fitted on that train set
(`refit_ga=True` default; `False` reuses the pipeline's bounds for a quick check), selection on
validation, metrics on test. The baseline (`out_of_tolerance_count > 0`) uses that run's own
bounds on that run's test vehicles. AUC mean/std ignore NaN runs.

### 4.5 `warranty_risk/labels/sigmoid.py` (2 / 1)

`generate_warranty_labels(..., random_state=None)` uses `np.random.default_rng(random_state)`,
so labels are reproducible when a seed is given.

### 4.6 `app.py` (8 / 2)

Prediction tab text now says the model is "selected by validation F1" with the threshold "chosen
on validation", and a caption shows the train / validation / test counts and states that the
metrics are on held-out test vehicles.

### 4.7 `tests/test_leakage_free.py` (new, 88 lines, 6 tests)

| Test | What it guards |
|---|---|
| Split disjoint, complete, stratified | No vehicle in two sets |
| GA only ever sees train vehicles | A spy on `run_ga_optimization` checks its input equals the train ids and `internal_holdout == 0` (my first version, which flipped test labels and compared bounds, was vacuous: the GA optimum did not move even when it saw all labels, so I replaced it) |
| Test labels cannot move model or threshold | Flipping every test label leaves `best_name`, `best_threshold` and `Val_F1` unchanged |
| Shuffled labels give chance-level AUC | AUC between 0.25 and 0.75 |
| No overlap in the training result | Train, validation and test indexes are disjoint |
| Seeded labels reproducible | Same seed gives the same labels; a different seed does not |

---

## 5. Commit `d75afc8`: README (73 / 17)

| Section | Change |
|---|---|
| Status banner | Notes the agent extension and links to Known limitations |
| 5. Repository structure | Adds `agent/`, `preprocessing/`, `tests/`, live-scoring scripts, `plant_simulator_pkg/`; removes the old `simulator/` folder from the tree |
| 6. Package table | Adds `Preprocessor`, `encoding`, `pad_single_vehicle`, `score_new_vehicle`, `QualityMonitoringAgent`, `model_registry` |
| 7. UI | Replaces the "Five tabs" table with the four phases |
| 8. How to run | Test command, live scoring instructions |
| 9. Design decisions | Two new rows (encoding, preprocessing) |
| 10. Completed work | Seven new checked items, later one more for the leakage-free evaluation |
| Known limitations (new) | Small test set, threshold not cost-based, synthetic labels, self-reinforcing feedback, partial vehicles, upload path |
| Credit | Noor's "Developed by" lines kept (the working tree had deleted them; restored); "Extended by: Sravani B" added |

`d6aa63e` also updated the README (7 / 5): Stage 4 text, limitations and next steps.

---

## 6. Measured effect of the evaluation change

Same simulated data, 3 seeds, 500 vehicles, small GA (15 generations, population 40).

| Labels | Old procedure F1 / AUC | New procedure F1 / AUC |
|---|---|---|
| Real | 0.473 / 0.776 | 0.376 / 0.743 |
| Shuffled (no signal) | 0.403 / 0.685 | 0.160 / 0.513 |

The old procedure scores well above chance on labels that carry no information, which is
leakage. The new one is at chance. The old gap came from several sources together: GA fit on all
labels, threshold chosen on the test set, model chosen on the test set.

---

## 7. Test inventory

| Suite | Tests | Written in this session |
|---|---|---|
| `test_encoding_padding.py` | 5 | Yes |
| `test_preprocessor.py` | 8 | Yes |
| `test_leakage_free.py` | 6 | Yes |
| `test_model_registry.py` | 6 | 2 added (preprocessing round-trip, old-save compatibility) |
| Others (config, data, feedback, live scoring, MQTT, orchestrator gating, validation) | 48 | No (already in tree) |
| **Total** | **73** | |

Run: `cd warranty_risk_app` then `python -m pytest tests/`.

---

## 8. Not changed, still open

| Item | File | Note |
|---|---|---|
| Feedback loop is self-reinforcing | `agent/feedback.py` | Uses all vehicles and does not distinguish real from simulated labels; mutates the config in place |
| Upload path stops before Learning | `agent/orchestrator.py` | Validated and cleaned, but no feature matrix is built |
| No database or historian connector | none | Only file upload and MQTT |
| Cost-aware threshold | `prediction/models.py` | Threshold maximises validation F1, ignoring `c_fa` and `c_fn` |
| Partial-vehicle scoring | `prediction/padding.py` | Unreached stations are zero-filled |
| `Preprocessor` not used on a train/test split | `preprocessing/` | Built and tested, not yet wired into a pipeline path |
| Synthetic, circular labels | `labels/sigmoid.py` | Needs real or independently generated claims |
| Not on `main` | git | Work lives on `warranty-risk-updates` |

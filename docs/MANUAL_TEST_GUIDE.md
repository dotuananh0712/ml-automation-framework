# ML Automation Framework - Manual Test Guide

This guide provides step-by-step instructions to manually test the ML Automation Framework with a real propensity model (customer churn prediction).

## Prerequisites

1. Install the framework and dependencies:
```bash
cd /path/to/ml-automation-framework
pip3 install -e ".[validation,explainability,tuning]"
```

2. Ensure MLflow is running locally (or set MLFLOW_TRACKING_URI):
```bash
# Optional: Start MLflow locally
mlflow server --host 127.0.0.1 --port 5000
# Then set: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## Step 1: Generate Test Data

**Purpose**: Create a realistic customer churn dataset for testing

```bash
# Navigate to the framework directory
cd /path/to/ml-automation-framework

# Generate test datasets
python tests/manual/generate_data.py
```

**Expected Output**:
```
Generating test datasets...
Created: tests/manual/data/churn_train.parquet (500 rows)
Created: tests/manual/data/churn_test.parquet (100 rows)
Created: tests/manual/data/churn_bad_data.parquet (500 rows)
Created: tests/manual/data/churn_train.csv

Test datasets generated successfully!

Data statistics:
  Total samples: 500
  Churn rate: ~30-40% (varies due to randomness)
  Features: ['customer_id', 'age', 'tenure_months', 'monthly_charges', 'total_charges', 'contract_type', 'payment_method', 'churn']
```

**Files Created**:
- `tests/manual/data/churn_train.parquet` - Training dataset (500 rows)
- `tests/manual/data/churn_test.parquet` - Test dataset (100 rows)
- `tests/manual/data/churn_bad_data.parquet` - Dataset with validation issues
- `tests/manual/data/churn_train.csv` - CSV version of training data

**Data Schema**:
```
customer_id: Customer identifier
age: Customer age (18-75)
tenure_months: Months with company (1-72)
monthly_charges: Monthly bill amount ($)
total_charges: Total lifetime charges ($)
contract_type: Month-to-month, One year, or Two year
payment_method: Credit card, Bank transfer, Electronic check, or Mailed check
churn: Target variable (0 = retained, 1 = churned)
```

---

## Test Case 1: Basic Training Pipeline

**Purpose**: Test existing functionality (no new features)

### Step 1.1: Validate Config

```bash
mlf validate tests/manual/configs/test_train_basic.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_train_basic.yaml
Config validated: manual_test_basic_train
  Name: manual_test_basic_train
  Type: classification
  Model: xgboost
  Experiment: /manual_tests/basic_train
```

### Step 1.2: Train Model

```bash
mlf train tests/manual/configs/test_train_basic.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_train_basic.yaml
Config validated: manual_test_basic_train
  Pipeline type: classification
  Model: xgboost
  Data source: tests/manual/data/churn_train.parquet

Training complete!
MLflow Run ID: abc123def456... (actual run ID shown)

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric             ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_accuracy      │ 0.7500   │
│ test_f1            │ 0.7200   │
│ test_precision     │ 0.7800   │
│ test_recall        │ 0.6800   │
│ test_roc_auc       │ 0.8200   │
│ train_accuracy     │ 0.8100   │
│ train_f1           │ 0.8000   │
│ train_precision    │ 0.8300   │
│ train_recall       │ 0.7900   │
│ train_roc_auc      │ 0.8700   │
│ val_accuracy       │ 0.7600   │
│ val_f1             │ 0.7400   │
│ val_precision      │ 0.7900   │
│ val_recall         │ 0.7100   │
│ val_roc_auc        │ 0.8300   │
└────────────────────┴──────────┘
```

### Step 1.3: Verify in MLflow

Open MLflow UI:
```bash
open http://127.0.0.1:5000
# or
curl http://127.0.0.1:5000
```

**Check**:
- Experiment `/manual_tests/basic_train` exists
- One run with metrics displayed
- Model artifact present under "Artifacts"
- Parameters logged

---

## Test Case 2: Evaluate Command

**Purpose**: Test model evaluation on new data

### Step 2.1: Get Run ID from Test Case 1

From the training output, note the MLflow Run ID (e.g., `abc123def456`)

```bash
# Option 1: Use the Run ID from previous output
RUN_ID="abc123def456"  # Replace with actual ID

# Option 2: Query MLflow to find the latest run
mlflow runs list --experiment-name /manual_tests/basic_train | grep "run_id"
```

### Step 2.2: Evaluate Model

```bash
mlf evaluate 78195e62b78c489cb0b70d2cd3955c38 tests/manual/data/churn_test.parquet
```

**Expected Output**:
```
Loading model from MLflow run: abc123def456
Model loaded successfully
Loading evaluation data: tests/manual/data/churn_test.parquet
Data loaded: 100 rows, 8 columns
Generating predictions...

Evaluation complete!

┏━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric     ┃ Value    ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━┩
│ accuracy   │ 0.7400   │
│ f1         │ 0.7100   │
│ precision  │ 0.7900   │
│ recall     │ 0.6500   │
│ roc_auc    │ 0.8100   │
└────────────┴──────────┘

Logging metrics to MLflow...
Metrics logged to run: eval_run_789
```

### Step 2.3: Verify Evaluation Run

Check MLflow UI:
- New evaluation run created in `/manual_tests/basic_train` experiment
- Parameters include: `source_run_id`, `eval_data_path`, `eval_data_rows`
- Metrics include: `eval_accuracy`, `eval_f1`, `eval_precision`, `eval_recall`, `eval_roc_auc`

---

## Test Case 3: Error Handling

**Purpose**: Test user-friendly error messages

### Step 3.1: Missing File Error

```bash
mlf train tests/manual/configs/nonexistent.yaml
```

**Expected Output**:
```
File not found: tests/manual/configs/nonexistent.yaml
Suggestion: Check path and ensure file exists
```

### Step 3.2: Missing Column Error

Create a config with wrong target column:
```bash
cat > tests/manual/configs/test_wrong_column.yaml << 'EOF'
name: test_wrong_column
pipeline_type: classification
data:
  source: tests/manual/data/churn_train.parquet
  format: parquet
  target_column: nonexistent_column
  train_ratio: 0.8
features:
  numeric_impute_strategy: median
model:
  model_type: xgboost
mlflow:
  experiment_name: /manual_tests/error_test
EOF

mlf train tests/manual/configs/test_wrong_column.yaml
```

**Expected Output**:
```
Column error: Column 'nonexistent_column' not found in dataset
Available columns: ['customer_id', 'age', 'tenure_months', ... ]
```

---

## Test Case 4: Data Validation (Great Expectations)

**Purpose**: Test Great Expectations integration

### Step 4.1: Valid Data Passes Validation

```bash
mlf train tests/manual/configs/test_validation.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_validation.yaml
Config validated: manual_test_validation
  Pipeline type: classification
  Model: random_forest
  Data source: tests/manual/data/churn_train.parquet

Data validation passed
Training complete!
...
MLflow Run ID: validation_run_123
```

### Step 4.2: Invalid Data Fails Validation

```bash
mlf train tests/manual/configs/test_validation_bad.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_validation_bad.yaml

Data validation failed:
- Column 'age': expect_column_values_to_not_be_null failed
  Violations: 11 rows have null values
- Column 'age': expect_column_values_to_be_between failed
  Violations: 5 rows have negative values
- Column 'contract_type': expect_column_values_to_be_in_set failed
  Violations: 5 rows have invalid categories

(Exit code: 1 - Pipeline halted)
```

### Step 4.3: Verify in MLflow

- Successful validation run shows validation completion message
- Failed validation prevents model training
- Check MLflow run logs for validation details

---

## Test Case 5: SHAP Explainability

**Purpose**: Test SHAP feature importance generation

### Step 5.1: Train with SHAP

```bash
mlf train tests/manual/configs/test_shap.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_shap.yaml
Config validated: manual_test_shap
Training complete!

Generating SHAP explanations
Creating SHAP explainer (type=tree)
Computing SHAP values (n_samples=200)
Generated SHAP summary plot
Generated SHAP bar plot
SHAP explanations complete (n_plots=2)

MLflow Run ID: shap_run_456
```

### Step 5.2: Check SHAP Artifacts

Open MLflow UI and navigate to the run. Under "Artifacts":
```
shap_plots/
├── shap_summary.png          # Beeswarm plot of SHAP values
├── shap_bar.png              # Mean absolute SHAP values
└── shap_dependence_*.png     # Optional dependence plots (if enabled)

shap_feature_importance.json  # Dictionary of feature importance scores
```

**Example SHAP Output**:
```
Feature Importance (from JSON):
{
  "monthly_charges": 0.245,
  "tenure_months": 0.201,
  "age": 0.156,
  "total_charges": 0.189,
  "contract_type": 0.087,
  "payment_method": 0.058
}
```

### Step 5.3: Interpret Results

- **monthly_charges**: Most important predictor of churn
- **tenure_months**: Inversely correlated (higher tenure = less churn)
- **age**: Moderate importance
- Low importance features suggest feature engineering opportunity

---

## Test Case 6: Hyperparameter Tuning (Optuna)

**Purpose**: Test automated hyperparameter optimization

### Step 6.1: Run Tuning (10 trials)

```bash
mlf tune tests/manual/configs/test_tuning.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_tuning.yaml
Config validated: manual_test_tuning
  Pipeline type: classification
  Model: xgboost
  Trials: 10
  Metric: val_f1
  Direction: maximize

Loading data: tests/manual/data/churn_train.parquet
Data loaded: 500 rows, 8 columns
  Train: 350 rows
  Validation: 150 rows

Starting hyperparameter tuning...

Trial 0: F1=0.6234  (n_estimators=45, max_depth=4, learning_rate=0.0523)
Trial 1: F1=0.6512  (n_estimators=87, max_depth=6, learning_rate=0.1245)
Trial 2: F1=0.6789  (n_estimators=56, max_depth=5, learning_rate=0.0876)
...
Trial 9: F1=0.7012  (n_estimators=92, max_depth=7, learning_rate=0.1523)

Tuning complete!

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Parameter       ┃ Value    ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ learning_rate   │ 0.152316 │
│ max_depth       │ 7        │
│ n_estimators    │ 92       │
└━━━━━━━━━━━━━━━━┴──────────┘

Best val_f1: 0.7012
Best trial: #9
Total trials: 10

Search space:
  n_estimators: [20, 100] (int)
  max_depth: [2, 8] (int)
  learning_rate: [0.01, 0.3] (log_float)
```

### Step 6.2: Override Trial Count

```bash
mlf tune tests/manual/configs/test_tuning.yaml --trials 5 --metric val_accuracy
```

**Expected Output**: Runs only 5 trials optimizing accuracy instead of F1

### Step 6.3: Check MLflow Trials

Open MLflow UI and navigate to `/manual_tests/tuning` experiment:
- Parent run: `manual_test_tuning_tuning`
- Child runs: 10 nested runs (one per trial)
- Each trial shows:
  - Parameters: `n_estimators`, `max_depth`, `learning_rate`
  - Metrics: `val_f1` (or specified metric)
  - Status: COMPLETED or PRUNED

**MLflow Nested Runs Structure**:
```
manual_test_tuning_tuning (Parent)
├── trial_0 (Child)
│   ├── n_estimators: 45
│   ├── max_depth: 4
│   ├── learning_rate: 0.0523
│   └── val_f1: 0.6234
├── trial_1 (Child)
│   └── ...
...
└── trial_9 (Child)
    └── val_f1: 0.7012 (Best)
```

---

## Test Case 7: Full Pipeline (All Features)

**Purpose**: Test data validation + SHAP + training together

```bash
mlf train tests/manual/configs/test_full_pipeline.yaml
```

**Expected Output**:
```
Loading config: tests/manual/configs/test_full_pipeline.yaml
Config validated: manual_test_full_pipeline

Data validation passed (3 expectations checked)
Training complete!
Generating SHAP explanations
SHAP explanations complete (n_plots=2)

MLflow Run ID: full_pipeline_789
```

**Verification in MLflow**:
- Run has model artifact
- Run has metrics for train/val/test
- Run has SHAP artifacts
- Run logs show validation completion

---

## Test Case 8: Databricks Deployment (Optional)

**Purpose**: Test deployment to Databricks Model Serving (requires Databricks workspace)

### Step 8.1: Prerequisites

```bash
# Set Databricks credentials (if deploying to Databricks)
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Step 8.2: Deploy Model

```bash
# Use Run ID from a previous training
mlf deploy $RUN_ID --endpoint churn-predictor --size Small
```

**Expected Output**:
```
Deploying model
  Run ID: abc123def456
  Target: databricks-model-serving
  Endpoint: churn-predictor

Deployment initiated!
  Endpoint: churn-predictor
  URL: https://your-workspace.databricks.com/serving-endpoints/churn-predictor
  Status: PENDING

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Property      ┃ Value          ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ model_name    │ churn_predictor│
│ model_version │ 1              │
│ workload_size │ Small          │
│ scale_to_zero │ True           │
└━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━┘
```

### Step 8.3: Check Endpoint Status

```bash
mlf endpoint-status churn-predictor
```

**Expected Output**:
```
Endpoint churn-predictor: READY
```

---

## Summary Checklist

Use this checklist to track test completion:

- [ ] Test Case 1: Basic Training
  - [ ] 1.1 Config validation passes
  - [ ] 1.2 Training completes with metrics
  - [ ] 1.3 MLflow artifacts visible

- [ ] Test Case 2: Evaluate
  - [ ] 2.1 Evaluation on test data succeeds
  - [ ] 2.2 Metrics displayed correctly
  - [ ] 2.3 MLflow evaluation run created

- [ ] Test Case 3: Error Handling
  - [ ] 3.1 Friendly missing file error
  - [ ] 3.2 Friendly column not found error

- [ ] Test Case 4: Data Validation
  - [ ] 4.1 Valid data passes validation
  - [ ] 4.2 Invalid data fails with detailed errors
  - [ ] 4.3 MLflow shows validation results

- [ ] Test Case 5: SHAP
  - [ ] 5.1 Training with SHAP completes
  - [ ] 5.2 SHAP plots generated and logged
  - [ ] 5.3 Feature importance JSON created

- [ ] Test Case 6: Optuna Tuning
  - [ ] 6.1 Tuning completes with 10 trials
  - [ ] 6.2 Best hyperparameters shown
  - [ ] 6.3 MLflow shows nested trial runs

- [ ] Test Case 7: Full Pipeline
  - [ ] 7 All features work together

- [ ] Test Case 8: Deployment (Optional)
  - [ ] 8.1 Model deploys to Databricks
  - [ ] 8.2 Endpoint becomes READY
  - [ ] 8.3 Status command works

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Install missing optional dependencies:
```bash
pip install -e ".[validation,explainability,tuning]"
```

### Issue: MLflow runs not visible

**Solution**: Check MLFLOW_TRACKING_URI:
```bash
echo $MLFLOW_TRACKING_URI
# If empty, set it:
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

### Issue: SHAP plots not generated

**Solution**: Install matplotlib:
```bash
pip install matplotlib>=3.8
```

### Issue: Optuna pruning warnings

**Solution**: This is normal. Reduce trials if too many are pruned:
```bash
mlf tune config.yaml --trials 5
```

### Issue: Databricks deployment fails

**Solution**: Verify credentials are set:
```bash
echo $DATABRICKS_HOST
echo $DATABRICKS_TOKEN
```

---

## Performance Baseline

For comparison, here are expected metrics for the churn prediction model:

**XGBoost (50 trees)**:
- Accuracy: 75-80%
- F1: 72-78%
- ROC-AUC: 82-87%

**Random Forest (30 trees)**:
- Accuracy: 73-77%
- F1: 70-75%
- ROC-AUC: 80-85%

**After Tuning (Optuna, 10 trials)**:
- Expected improvement: +2-5% on F1 score
- Best parameters typically: n_estimators 80-100, max_depth 6-8, learning_rate 0.10-0.15

---

## Advanced: Custom Test

You can create your own config file for testing specific scenarios:

```bash
cat > tests/manual/configs/my_test.yaml << 'EOF'
name: my_custom_test
pipeline_type: classification
data:
  source: tests/manual/data/churn_train.parquet
  format: parquet
  target_column: churn
  train_ratio: 0.8
features:
  numeric_impute_strategy: median
model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100
    max_depth: 5
mlflow:
  experiment_name: /my_tests/custom
EOF

mlf train tests/manual/configs/my_test.yaml
```

---

## Next Steps

After validating the framework:

1. **Customize for Your Data**: Replace `tests/manual/data/` with your datasets
2. **Tune Hyperparameters**: Use the tuning config to optimize for your metrics
3. **Monitor with MLflow**: Set up MLflow for production tracking
4. **Deploy Models**: Use Databricks deployer for serving endpoints
5. **Integrate into CI/CD**: Automate testing and deployment pipelines

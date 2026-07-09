# MLflow Model Registry Workflow for the MovieOps Sentiment Model

### 1. Train the model

Train the sentiment analysis model using the training dataset and log everything to MLflow.

```python
with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")

    # Log the trained model
    mlflow.sklearn.log_model(model, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
```

**MLflow UI:**
Navigate to **Experiments → movie-sentiment** and compare all training runs.

---

### 2. Select the best training run

Review the experiment results and choose the model with the best performance (for example, the highest validation accuracy and acceptable precision, recall, and F1-score).

Example:

```
Run #21
Accuracy: 91%
F1-score: 0.90
```

---

### 3. Register the model

Register the selected model in the MLflow Model Registry.

```python
model_uri = f"runs:/{run_id}/model"

mlflow.register_model(
    model_uri=model_uri,
    name="sentiment-model"
)
```

**MLflow UI:**
Open the run and click **Register Model**, then choose **sentiment-model**.

The registry now contains:

```
sentiment-model
Version 1
Stage: None
```

---

### 4. Move the model to Staging

After registration, promote the model to the **Staging** stage for testing.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="sentiment-model",
    version=1,
    stage="Staging"
)
```

**MLflow UI:**
Models → sentiment-model → Version 1 → Transition Stage → **Staging**

---

### 5. Test the staging model

Before serving real users, perform production-like validation:

* Verify API integration
* Check prediction accuracy on validation data
* Measure latency
* Test memory and CPU usage
* Verify preprocessing pipeline
* Perform end-to-end API tests

If any test fails, retrain or fix the model without affecting production.

---

### 6. Promote the model to Production

Once all validation passes, promote the model to **Production**.

```python
client.transition_model_version_stage(
    name="sentiment-model",
    version=1,
    stage="Production"
)
```

**MLflow UI:**
Models → sentiment-model → Version 1 → Transition Stage → **Production**

The registry now looks like:

```
sentiment-model

Version 1
Stage: Production
```

---

### 7. Serve the production model in FastAPI

Instead of loading a local pickle file, load the model directly from the registry.

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model(
    "models:/sentiment-model/Production"
)
```

Every prediction request now uses the model currently marked as **Production**.

---

### 8. Deploy a better model

Suppose a new model achieves 94% accuracy.

Repeat the workflow:

1. Train the new model
2. Log it to MLflow
3. Register it
4. Move it to **Staging**
5. Test it thoroughly
6. Promote it to **Production**

The registry becomes:

```
Version 1 → Archived

Version 2 → Production
```

The FastAPI application continues loading:

```python
models:/sentiment-model/Production
```

No application code changes are required.

---

### 9. Roll back if necessary

If Version 2 performs poorly in production, simply change the stages.

```python
client.transition_model_version_stage(
    name="sentiment-model",
    version=2,
    stage="Archived"
)

client.transition_model_version_stage(
    name="sentiment-model",
    version=1,
    stage="Production"
)
```

After the service restarts (or reloads the production model), it automatically serves Version 1 again.

---

## Complete Workflow

```
Train Model
      │
      ▼
Log Run to MLflow
      │
      ▼
Compare Experiment Results
      │
      ▼
Register Best Model
      │
      ▼
Stage = None
      │
      ▼
Promote to Staging
      │
      ▼
Test & Validate
      │
      ▼
Promote to Production
      │
      ▼
FastAPI loads:
models:/sentiment-model/Production
      │
      ▼
Users receive live predictions
      │
      ▼
New model available?
      │
      ▼
Repeat the workflow or roll back if needed.
```

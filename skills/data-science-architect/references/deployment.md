# Model Packaging & Deployment Reference

This guide covers model serialization, API creation, monitoring, and production lifecycle management.

---

## 1. Serialization Best Practices

Save trained scikit-learn models as full pipelines including feature preprocessing steps:

```python
import joblib

# Packaging model pipeline
model_artifact = {
    'pipeline': trained_pipeline,
    'feature_names': feature_list,
    'metrics': evaluation_metrics,
    'version': '1.0.0'
}

joblib.dump(model_artifact, 'outputs/model_v1.joblib', compress=3)
```

---

## 2. Model Serving Patterns

- **Batch Inference**: Offline prediction job executed on schedule (e.g. nightly SQL/PySpark jobs).
- **Real-Time REST API**: High-speed synchronous inference server using FastAPI / Flask.
- **ONNX (Open Neural Network Exchange)**: Convert scikit-learn or PyTorch models to ONNX runtime for ultra-low latency execution (<5ms).

---

## 3. Production Monitoring Checklist

1. **Prediction Drift**: Monitor output prediction distribution for shifts in mean or positive prediction rate.
2. **Feature Drift**: Measure PSI or Kolmogorov-Smirnov test scores on incoming features against training distributions.
3. **Latency & Throughput**: Track p95 / p99 request latency and error rate (5xx errors).
4. **Ground Truth Feedback Loop**: Compare predicted targets with actual labels once observed to track real-world precision/recall over time.

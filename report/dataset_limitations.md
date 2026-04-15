# Dataset Limitation Notes

The initial text-only TF-IDF + Multinomial Naive Bayes model produced perfect test accuracy, which is suspicious for a seven-class NLP problem. We checked for common evaluation bugs such as fitting on the full dataset, predicting on the training split, evaluating `y_test` against itself, including metadata as model input, and literal label leakage in the text column.

The main issue found so far is that the synthetic dataset contains repeated text patterns. We removed exact duplicate flattened conversations before train/test splitting, reducing the dataset from 10,000 rows to 8,455 unique conversations. However, message-level template overlap remains: in the deduplicated split, all 1,691 test examples contain at least one individual message line that also appears in the training set.

Because removing every repeated message template would require subjective and potentially destructive filtering, the current project treats this as a limitation of the synthetic benchmark. Model performance should be interpreted as classification performance on this generated dataset, not as evidence that the model generalizes to real-world manipulation detection.

The manual-example sanity check also showed weaker generalization than the held-out synthetic test score suggests. The model matched 5 of 7 manually written examples, misclassifying the direct coercion example as neutral and the passive-aggressive example as guilt-tripping. This supports the interpretation that the synthetic test score is inflated by repeated generation patterns.

For the final project, we will report:

- raw dataset size and deduplicated dataset size
- majority-class baseline on the deduplicated split
- TF-IDF model performance on the deduplicated split
- message-level overlap diagnostics
- predictions on manually written sanity-check examples

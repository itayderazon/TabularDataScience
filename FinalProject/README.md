# Feature Selection Hybrid Method

This repository contains the implementation of our hybrid feature selection method combining tree-based importance measures with correlation-based feature clustering.

## Important Runtime Notice

**⚠️ WARNING: The RFE (Recursive Feature Elimination) baseline method takes approximately 3 hours to run on the synthetic dataset due to its computational complexity.**

## Runtime Options for Method Comparison

When running our feature selection comparisons, you have three options for handling the computationally expensive RFE method:

### Option A: Skip the Synthetic Dataset

If you want quick results from all methods, you can run the code only on the smaller datasets:

```python
# Run only on Wine, Breast Cancer and Diabetes datasets
datasets = {
    'Wine': (load_wine(), 'classification'),
    'BreastCancer': (load_breast_cancer(), 'classification'),
    'Diabetes': (load_diabetes(), 'regression')
    # Synthetic dataset excluded
}
```

This will allow you to see comparison results across all methods on standard datasets within minutes.

### Option B: Exclude RFE for the Synthetic Dataset Only

If you want to include the synthetic dataset but avoid the 3-hour runtime, modify the `compare_methods` function for the synthetic dataset:

```python
    methods = {
        'Our Method': custom_method,
        # 'RFE': rfe_method,  # RFE excluded for synthetic dataset
        'PCA': pca_method
    }

```

We've already run the full comparison and documented the results in our paper, so you can trust those values without needing to reproduce them.

### Option C: Run Everything (Full Reproducibility)


Be prepared to wait approximately 3 hours for the RFE method to complete on the synthetic dataset. This option provides full reproducibility of our results but requires significant computation time.

**Our recommendation:** Use Option B for exploratory analysis, and Option C only if full reproduction is necessary.

## Running the Code

1. The main analysis is in `feature_selection.ipynb`
2. Choose one of the runtime options described above based on your time constraints
3. Execute the notebook cells in order

## Datasets

The code includes evaluations on four datasets:
- Wine Quality Dataset
- Breast Cancer Dataset
- Diabetes Dataset
- Synthetic Regression Dataset (150 features, 3000 samples)

## Interpreting Results

Our paper provides detailed analysis of the results, but here's a quick summary:
- Our method is significantly faster than RFE (up to 356× speedup on the synthetic dataset)
- Performance is comparable to RFE across all datasets
- Our method selects fewer features than RFE while maintaining performance
- The visualization components provide clear interpretability of feature relationships and selection decisions


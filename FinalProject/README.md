# Feature Selection Hybrid Method

This repository contains the implementation of our hybrid feature selection method combining tree-based importance measures with correlation-based feature clustering.

## Important Runtime Notice

**⚠️ WARNING: The RFE (Recursive Feature Elimination) baseline method takes approximately 3 hours to run on the synthetic dataset due to its computational complexity.**

To make the code more user-friendly, we have commented out the RFE method with `##` in the synthetic dataset evaluation section. This allows you to run the rest of the code and see our method's results without waiting for hours.

## Running the Code

1. The main analysis is in `feature_selection.ipynb`
2. To run without RFE on the synthetic dataset:
   - Leave the commented sections as they are
   - Results for RFE on synthetic data are pre-computed and will be displayed

3. If you want to run the full comparison including RFE on the synthetic dataset:
   - Uncomment the lines marked with `##` in the synthetic dataset section
   - Be prepared for a long runtime (approximately 3 hours)
   - We recommend running this overnight or on a powerful machine

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






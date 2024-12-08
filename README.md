# Credit_Risk_Model
Calculates the expected loss!
# Credit Risk Assessment Model

## Overview
A comprehensive credit risk assessment system that combines machine learning models to predict Expected Loss (EL) through its core components: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). The model achieves 89% accuracy in default prediction and 79% R-squared for exposure estimation, with robust performance metrics including 18.29% mean percentage error for EAD and 0.74 F1-score for default events.

## Key Features
- **Integrated Risk Assessment**: Combines PD, LGD, and EAD predictions for comprehensive Expected Loss calculation
- **Advanced Feature Engineering**: Implements sophisticated preprocessing and feature creation
- **Model Interpretability**: Includes risk binning and detailed output explanations
- **Robust Performance**: Demonstrated strong accuracy and reliability across different risk scenarios

## Model Components

### 1. Probability of Default (PD) Model
- XGBoost-based classifier with calibration
- Feature binning for credit scores, DTI, and LTV ratios
- Risk-based interaction features
- Comprehensive sanity checks and business rule overlays

### 2. Exposure at Default (EAD) Model
- XGBoost regressor for exposure prediction
- Leverage ratio considerations
- Property value and income relationship modeling
- Term and interest rate impact analysis

### 3. Loss Given Default (LGD) Calculation
- Proxy LGD calculation incorporating multiple risk factors
- Security type consideration
- Income and property value relationships
- Term and credit score impacts

## Getting Started

### Prerequisites
```python
pandas
numpy
scikit-learn
xgboost
```

### Basic Usage
```python
# Example usage for Expected Loss calculation
test_input = {
    'term': 360,
    'property_value': 500000,
    'income': 120000,
    'dtir1': 35,
    'Credit_Score': 740,
    'LTV': 80,
    'rate_of_interest': 5.5,
    'loan_type': 'type1',
    'loan_purpose': 'p1',
    'Credit_Worthiness': 'Good',
    'Secured_by': 'First lien',
    'Security_Type': 'Direct',
    'occupancy_type': 'Primary',
    'Region': 'Northeast'
}

result = predict_expected_loss(test_input)
```

## Model Training

### Handling Overfitting
The model implements several strategies to prevent overfitting:
1. Feature binning to reduce noise in continuous variables
2. Cross-validation during model training
3. Regularization in XGBoost models
4. Business rule overlays to ensure predictions align with domain knowledge
5. Interaction features to capture complex relationships without overfitting to individual variables

### Training Your Own Models
1. Start with data preprocessing:
   ```python
   df_processed = preprocess_data(your_data)
   ```

2. Create engineered features:
   ```python
   df_engineered = engineer_features(df_processed)
   df_engineered = create_interaction_features(df_engineered)
   ```

3. Train the PD model:
   ```python
   pd_model = train_pd_model(df_engineered)
   ```

4. Train the EAD model:
   ```python
   ead_model = train_ead_model(df_engineered)
   ```

5. Calculate Expected Loss:
   ```python
   el_result = calculate_expected_loss(pd_model, ead_model, test_input)
   ```

## Model Output Example
```python
Expected Loss Analysis:
Expected Loss Amount: $529.51
Component Breakdown:
- Probability of Default: 32.75%
- Loss Given Default: 3.06%
- Exposure at Default: $52,774.62
Risk Level: Low
```

## Performance Metrics
- PD Model Accuracy: 89%
- EAD R-squared: 79%
- Mean Percentage Error (EAD): 18.29%
- F1-Score (Default Events): 0.74

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

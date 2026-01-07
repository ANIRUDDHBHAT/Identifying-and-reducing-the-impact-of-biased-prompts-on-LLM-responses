# Chain of Thought (CoT) Prompt Engineering Implementation

This implementation adds Chain of Thought prompt engineering to reduce bias in LLM prompts through iterative refinement.

## Overview

The Chain of Thought (CoT) prompt engineering technique enhances prompts with step-by-step reasoning instructions to reduce bias. This implementation follows the control structure specified in the requirements.

## Features

### 1. Bias Classification
- **Sentiment Score**: Using TextBlob for polarity analysis
- **BERT-Based Bias Score**: Keyword-based approach with fallback to BERT embeddings
- **Diversity Score**: Vocabulary richness and unique word ratio

### 2. Iterative Refinement
- Maximum 4 iterations per prompt
- CoT templates that encourage systematic reasoning
- Stops when prompt becomes Neutral or max iterations reached

### 3. Output
- `train_df_chain_of_thought` DataFrame with columns:
  - `Original Prompt`: The original biased prompt
  - `Engineered Prompt (Latest)`: The final engineered version
  - `Current Classification (Neutral/Biased)`: Final classification
  - `iterations_till_neutral`: Number of iterations required

## Usage

### In Python Script:
```python
from cot_prompt_engineering import run_cot_prompt_engineering
import pandas as pd

# Load your training data
train_df = pd.read_excel('train_df.xlsx')

# Run CoT engineering
train_df_chain_of_thought = run_cot_prompt_engineering(train_df)

# Save results
train_df_chain_of_thought.to_excel('train_df_chain_of_thought.xlsx', index=False)
```

### In Jupyter Notebook:
The implementation has been integrated into `bias_classifier_R4.ipynb` as new cells at the end. Simply run the cells sequentially:

1. Import the CoT module
2. Run the engineering process
3. View statistics
4. Save results

## CoT Templates

The implementation uses four different CoT templates that rotate through iterations:

1. "Let's think through this step by step before concluding: {}"
2. "Let's reason systematically before answering: {}"
3. "Let's consider this carefully and objectively: {}"
4. "Let's approach this thoughtfully without assumptions: {}"

## Classification Criteria

A prompt is classified as **Biased** if any of the following conditions are met:
- BERT bias score > 0.1
- Absolute sentiment score > 0.4
- Diversity score < 0.4
- Contains explicit bias indicator keywords

Bias indicator keywords include: "obviously", "clearly", "always", "never", "must be", "should be", "stereotype", "typical", etc.

## Output Statistics

The implementation provides detailed statistics including:
- Number of prompts that became Neutral at each iteration
- Final count of Neutral vs. Biased prompts
- Percentage breakdown of results

## Files

- `cot_prompt_engineering.py`: Main implementation module
- `bias_classifier_R4.ipynb`: Jupyter notebook with integrated CoT cells
- `train_df.xlsx`: Input training data (28,656 prompts)
- `train_df_chain_of_thought.xlsx`: Output results (generated)

## Testing

Run a test with a sample of prompts:
```bash
python3 cot_prompt_engineering.py
```

This will process a sample from `train_df.xlsx` and display results.

## Requirements

- pandas
- numpy
- textblob
- sentence-transformers (optional, for BERT-based scoring)
- scikit-learn
- openpyxl

## Notes

- The implementation uses a keyword-based approach by default for faster processing
- BERT model loading can be enabled by setting `use_bert=True` in BiasClassifier initialization
- Processing all 28K+ prompts may take several minutes
- Results are saved after completion to preserve progress

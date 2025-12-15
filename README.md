# Group22-NLP-Final

# Job Experience Level Classification from Job Descriptions

Classify job postings into **Intern / Entry-level / Senior-level / Executive-level** using NLP models, comparing classic ML baselines (TF-IDF + Logistic Regression) with transformer-based models (section-aware BERT) and an LLM prompting “ceiling” baseline.

## Project Goals
- Build an end-to-end pipeline for **experience-level classification** from job descriptions
- Compare multiple approaches under the same evaluation setting
- Report **Accuracy + Macro-F1** (macro-F1 is key for class imbalance)
- Perform error analysis to understand confusions (e.g., Senior vs Executive)

## Dataset
We use a public Kaggle dataset of job postings (job descriptions + experience level labels).

### Label Space
- `Intern`
- `Entry-level`
- `Senior-level`
- `Executive-level`

## Methods Implemented
1. **Regex baseline**
   - Rule-based matching on keywords such as “intern”, “junior”, “senior”, “director”, etc.
   - If no match, fall back to a default class (e.g., majority class)

2. **TF-IDF + Logistic Regression**
   - Bag-of-words baseline with TF-IDF features
   - Logistic Regression classifier

3. **Section-aware BERT**
   - Split job descriptions into sections (e.g., Title / Responsibilities / Requirements / Others)
   - Encode each section with BERT, then aggregate (attention / pooling) for final prediction

4. **LLM prompting baseline**
   - Few-shot prompting/Fine-tuned as an upper-bound reference

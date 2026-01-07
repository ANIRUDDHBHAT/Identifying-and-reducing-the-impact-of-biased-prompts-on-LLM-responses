"""
Chain of Thought (CoT) Prompt Engineering for Bias Reduction

This module implements the CoT prompt engineering technique to reduce bias in prompts
through iterative refinement with step-by-step reasoning instructions.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BiasClassifier:
    """
    Classifier that evaluates prompts for bias using multiple metrics:
    - Sentiment Score (TextBlob)
    - BERT-based Bias Score (Sentence Transformers)
    - Diversity Score (vocabulary richness)
    """
    
    # Constants for bias scoring
    KEYWORD_SENSITIVITY = 2  # Multiplier to increase sensitivity to keyword matches
    RICHNESS_THRESHOLD = 20  # Number of unique words considered "rich"
    MAX_RICHNESS_BONUS = 0.3  # Maximum bonus for vocabulary richness
    
    def __init__(self, use_bert=False):
        """
        Initialize the bias classifier.
        
        Args:
            use_bert: Whether to use BERT model (requires download). Default False for simpler approach.
        """
        self.use_bert = use_bert
        
        # Define bias-related keywords/phrases for comparison
        self.bias_indicators = [
            "stereotype", "discriminate", "prejudice", "biased", "unfair",
            "assumption", "generalize", "obviously", "clearly", "typical",
            "always", "never", "all", "must be", "should be", "supposed to",
            "naturally", "inherently", "tend to"
        ]
        
        if use_bert:
            try:
                print("Loading BERT model for bias detection...")
                self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.bias_embeddings = self.bert_model.encode(
                    self.bias_indicators, 
                    convert_to_tensor=True
                )
                print("BiasClassifier initialized with BERT successfully.")
            except Exception as e:
                print(f"Warning: Failed to load BERT model: {e}")
                print("Falling back to keyword-based approach.")
                self.use_bert = False
        else:
            print("BiasClassifier initialized with keyword-based approach.")
    
    def calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment polarity using TextBlob.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment polarity score between -1 (negative) and 1 (positive)
        """
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except (ValueError, TypeError, AttributeError) as e:
            # Return neutral if text processing fails
            return 0.0
    
    def calculate_bert_bias_score(self, text: str) -> float:
        """
        Calculate bias score using BERT embeddings and cosine similarity
        with known bias indicators, or keyword matching if BERT is not available.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Bias score between 0 (less biased) and 1 (more biased)
        """
        if self.use_bert:
            try:
                # Get embedding for the text
                text_embedding = self.bert_model.encode(str(text), convert_to_tensor=True)
                
                # Calculate cosine similarity with bias indicators
                similarities = util.cos_sim(text_embedding, self.bias_embeddings)
                
                # Return the maximum similarity as bias score
                max_similarity = float(similarities.max())
                
                # Normalize to 0-1 range (cosine similarity is already -1 to 1)
                # We use abs to focus on strength of association
                return (max_similarity + 1) / 2
            except (AttributeError, ValueError, RuntimeError) as e:
                # Fall through to keyword-based approach if BERT fails
                pass
        
        # Fallback to keyword-based approach
        try:
            text_lower = str(text).lower()
            matches = sum(1 for indicator in self.bias_indicators if indicator in text_lower)
            # Normalize by total number of indicators, multiply by KEYWORD_SENSITIVITY to scale up detection
            # (since finding even 1-2 keywords should signal potential bias)
            return min(matches / len(self.bias_indicators) * self.KEYWORD_SENSITIVITY, 1.0)
        except (AttributeError, TypeError, ValueError) as e:
            # Return neutral default if text processing fails
            return 0.5
    
    def calculate_diversity_score(self, text: str) -> float:
        """
        Calculate diversity score based on unique word ratio and vocabulary richness.
        Higher diversity generally indicates less stereotypical language.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Diversity score between 0 (low diversity) and 1 (high diversity)
        """
        try:
            words = str(text).lower().split()
            if len(words) == 0:
                return 0.0
            
            unique_words = set(words)
            unique_ratio = len(unique_words) / len(words)
            
            # Bonus for longer unique word count (vocabulary richness)
            richness_bonus = min(len(unique_words) / self.RICHNESS_THRESHOLD, self.MAX_RICHNESS_BONUS)
            
            diversity = min(unique_ratio + richness_bonus, 1.0)
            return diversity
        except (AttributeError, TypeError, ZeroDivisionError) as e:
            # Return neutral default if text processing fails
            return 0.5
    
    def classify_prompt(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Classify a prompt as Biased or Neutral based on multiple scores.
        
        Classification logic:
        - Biased if: BERT bias score > 0.1 OR abs(sentiment) > 0.4 OR diversity < 0.4
                    OR contains bias indicator keywords
        - Neutral otherwise
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (classification, scores_dict)
        """
        sentiment_score = self.calculate_sentiment_score(text)
        bert_bias_score = self.calculate_bert_bias_score(text)
        diversity_score = self.calculate_diversity_score(text)
        
        scores = {
            'sentiment_score': sentiment_score,
            'bert_bias_score': bert_bias_score,
            'diversity_score': diversity_score
        }
        
        # Check for explicit bias keywords
        text_lower = str(text).lower()
        has_bias_keyword = any(indicator in text_lower for indicator in self.bias_indicators)
        
        # Classification logic - more sensitive to bias
        is_biased = (
            bert_bias_score > 0.1 or  # Lowered threshold
            abs(sentiment_score) > 0.4 or
            diversity_score < 0.4 or
            has_bias_keyword
        )
        
        classification = "Biased" if is_biased else "Neutral"
        
        return classification, scores


class CoTPromptEngineer:
    """
    Implements Chain of Thought (CoT) prompt engineering to reduce bias.
    """
    
    def __init__(self, bias_classifier: BiasClassifier):
        """Initialize with a bias classifier."""
        self.classifier = bias_classifier
        
        # CoT instruction templates
        self.cot_templates = [
            "Let's think through this step by step before concluding: {}",
            "Let's reason systematically before answering: {}",
            "Let's consider this carefully and objectively: {}",
            "Let's approach this thoughtfully without assumptions: {}"
        ]
    
    def apply_cot_engineering(self, prompt: str, iteration: int) -> str:
        """
        Apply CoT engineering by prepending a reasoning instruction.
        
        Note: We apply CoT templates to the original prompt each time (not cumulative)
        to avoid overly complex prompts. Each iteration uses a different template
        to provide varied reasoning approaches.
        
        Args:
            prompt: Original prompt text
            iteration: Current iteration number (0-3)
            
        Returns:
            Engineered prompt with CoT instruction
        """
        template_idx = iteration % len(self.cot_templates)
        cot_prompt = self.cot_templates[template_idx].format(prompt)
        return cot_prompt
    
    def process_prompts(self, train_df: pd.DataFrame, max_iterations: int = 4) -> pd.DataFrame:
        """
        Process prompts with iterative CoT engineering until neutral or max iterations reached.
        
        Args:
            train_df: DataFrame with 'prompt' column
            max_iterations: Maximum number of iterations (default 4)
            
        Returns:
            DataFrame with columns:
            - Original Prompt
            - Engineered Prompt (Latest)
            - Current Classification (Neutral/Biased)
            - iterations_till_neutral
        """
        print(f"\nStarting CoT prompt engineering on {len(train_df)} prompts...")
        print(f"Maximum iterations: {max_iterations}\n")
        
        results = []
        
        for idx, row in train_df.iterrows():
            original_prompt = row['prompt']
            current_prompt = original_prompt
            iterations_count = 0
            
            # Classify the original prompt first
            classification, scores = self.classifier.classify_prompt(original_prompt)
            
            # If already neutral, no iterations needed
            if classification == "Neutral":
                results.append({
                    'Original Prompt': original_prompt,
                    'Engineered Prompt (Latest)': original_prompt,
                    'Current Classification (Neutral/Biased)': classification,
                    'iterations_till_neutral': 0
                })
            else:
                # Prompt is biased, start iterative refinement
                for iteration in range(max_iterations):
                    # Apply CoT engineering
                    current_prompt = self.apply_cot_engineering(original_prompt, iteration)
                    
                    # Classify the engineered prompt
                    classification, scores = self.classifier.classify_prompt(current_prompt)
                    
                    if classification == "Neutral":
                        # Prompt became neutral, stop iterations
                        iterations_count = iteration + 1  # +1 because we count the iteration where it became neutral
                        break
                else:
                    # Max iterations reached without becoming neutral
                    iterations_count = max_iterations
                    # current_prompt already contains the last engineered version
                
                # Record result
                results.append({
                    'Original Prompt': original_prompt,
                    'Engineered Prompt (Latest)': current_prompt,
                    'Current Classification (Neutral/Biased)': classification,
                    'iterations_till_neutral': iterations_count if classification == "Neutral" else max_iterations
                })
            
            # Print progress every 1000 prompts
            if (idx + 1) % 1000 == 0:
                temp_df = pd.DataFrame(results)
                biased_count = len(temp_df[temp_df['Current Classification (Neutral/Biased)'] == 'Biased'])
                print(f"Processed {idx + 1}/{len(train_df)} prompts... (Current biased count: {biased_count})")
        
        # Create output dataframe
        result_df = pd.DataFrame(results)
        
        # Print iteration statistics
        print("\n" + "="*60)
        print("CoT Prompt Engineering Complete!")
        print("="*60)
        
        # Print after each iteration (conceptually - showing final distribution)
        print("\nIteration Statistics:")
        for iteration in range(max_iterations + 1):
            count = len(result_df[result_df['iterations_till_neutral'] == iteration])
            if count > 0:
                print(f"  After Iteration {iteration}: {count} prompts became Neutral")
        
        biased_count = len(result_df[result_df['Current Classification (Neutral/Biased)'] == 'Biased'])
        neutral_count = len(result_df[result_df['Current Classification (Neutral/Biased)'] == 'Neutral'])
        
        print(f"\nFinal Statistics:")
        print(f"  Neutral: {neutral_count} ({neutral_count/len(result_df)*100:.2f}%)")
        print(f"  Still Biased: {biased_count} ({biased_count/len(result_df)*100:.2f}%)")
        print("="*60)
        
        return result_df


def run_cot_prompt_engineering(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to run CoT prompt engineering on training data.
    
    Args:
        train_df: Training DataFrame with 'prompt' column
        
    Returns:
        train_df_chain_of_thought: DataFrame with CoT engineering results
    """
    # Initialize classifier
    classifier = BiasClassifier()
    
    # Initialize CoT engineer
    engineer = CoTPromptEngineer(classifier)
    
    # Process prompts with CoT engineering
    train_df_chain_of_thought = engineer.process_prompts(train_df, max_iterations=4)
    
    return train_df_chain_of_thought


# Example usage (commented out for notebook integration)
if __name__ == "__main__":
    # Load data
    import pandas as pd
    train_df = pd.read_excel('train_df.xlsx')
    
    # Run CoT engineering
    train_df_chain_of_thought = run_cot_prompt_engineering(train_df)
    
    # Display results
    print("\nSample results:")
    print(train_df_chain_of_thought.head(10))
    
    # Save results
    train_df_chain_of_thought.to_excel('train_df_chain_of_thought.xlsx', index=False)
    print("\nResults saved to 'train_df_chain_of_thought.xlsx'")

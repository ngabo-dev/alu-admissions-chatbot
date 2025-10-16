"""
ALU Chatbot Evaluation Script
Evaluates the fine-tuned Transformer model using NLP metrics
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotEvaluator:
    def __init__(self, model_path="../models/alu_chatbot_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load fine-tuned model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {model_path}")

    def calculate_perplexity(self, text):
        """Calculate perplexity for a given text"""
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
            loss = outputs.loss
            perplexity = math.exp(loss.item())

        return perplexity

    def calculate_bleu_score(self, reference, candidate):
        """Calculate BLEU score between reference and candidate"""
        smoothing = SmoothingFunction().method4

        # Tokenize
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())

        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens,
                                 smoothing_function=smoothing)

        return bleu_score

    def calculate_f1_score(self, reference, candidate):
        """Calculate F1 score based on word overlap"""
        ref_tokens = set(word_tokenize(reference.lower()))
        cand_tokens = set(word_tokenize(candidate.lower()))

        # Calculate precision and recall
        intersection = ref_tokens.intersection(cand_tokens)

        if len(cand_tokens) == 0:
            precision = 0
        else:
            precision = len(intersection) / len(cand_tokens)

        if len(ref_tokens) == 0:
            recall = 0
        else:
            recall = len(intersection) / len(ref_tokens)

        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1, precision, recall

    def evaluate_on_test_set(self, test_path="../data/test_dataset_public.csv"):
        """Evaluate model on test dataset"""
        test_df = pd.read_csv(test_path)

        results = {
            'bleu_scores': [],
            'f1_scores': [],
            'perplexities': [],
            'predictions': [],
            'references': []
        }

        logger.info(f"Evaluating on {len(test_df)} test samples...")

        for idx, row in test_df.iterrows():
            input_text = row['input_text']
            reference = row['target_text']

            # Generate prediction
            prediction = self.generate_response(input_text)

            # Calculate metrics
            bleu = self.calculate_bleu_score(reference, prediction)
            f1, _, _ = self.calculate_f1_score(reference, prediction)
            perplexity = self.calculate_perplexity(f"User: {input_text}\nBot: {prediction}")

            # Store results
            results['bleu_scores'].append(bleu)
            results['f1_scores'].append(f1)
            results['perplexities'].append(perplexity)
            results['predictions'].append(prediction)
            results['references'].append(reference)

            if idx % 5 == 0:
                logger.info(f"Processed {idx+1}/{len(test_df)} samples")

        # Calculate averages
        avg_bleu = np.mean(results['bleu_scores'])
        avg_f1 = np.mean(results['f1_scores'])
        avg_perplexity = np.mean(results['perplexities'])

        logger.info(".4f")
        logger.info(".4f")
        logger.info(".4f")

        return results, avg_bleu, avg_f1, avg_perplexity

    def generate_response(self, user_input, max_new_tokens=50, temperature=0.7):
        """Generate response for evaluation"""
        input_text = f"User: {user_input}\nBot:"

        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Bot:" in full_response:
            response = full_response.split("Bot:")[-1].strip()
        else:
            response = full_response.replace(input_text, "").strip()

        return response

    def qualitative_evaluation(self, test_questions):
        """Perform qualitative evaluation with sample questions"""
        print("\n" + "="*60)
        print("QUALITATIVE EVALUATION")
        print("="*60)

        for question in test_questions:
            response = self.generate_response(question)
            print(f"\nUser: {question}")
            print(f"Bot: {response}")
            print("-" * 40)

def main():
    """Main evaluation function"""
    evaluator = ChatbotEvaluator()

    # Quantitative evaluation
    results, avg_bleu, avg_f1, avg_perplexity = evaluator.evaluate_on_test_set()

    # Qualitative evaluation
    test_questions = [
        "What programmes do you offer?",
        "How much does ALU cost?",
        "I want to apply to ALU",
        "What are the entry requirements for IBT?",
        "When are intakes?",
        "Are there scholarships?"
    ]

    evaluator.qualitative_evaluation(test_questions)

    # Save results
    results_df = pd.DataFrame({
        'input': pd.read_csv('../data/test_dataset_public.csv')['input_text'],
        'reference': results['references'],
        'prediction': results['predictions'],
        'bleu_score': results['bleu_scores'],
        'f1_score': results['f1_scores'],
        'perplexity': results['perplexities']
    })

    results_df.to_csv('../evaluation_results.csv', index=False)
    logger.info("Evaluation results saved to evaluation_results.csv")

    return avg_bleu, avg_f1, avg_perplexity

if __name__ == "__main__":
    main()
"""
ALU University Admissions Chatbot - Transformer-based Implementation
Fine-tuned GPT-2 model for conversational AI in education domain
"""

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import pandas as pd
from datasets import Dataset
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ALUChatbot:
    def __init__(self, model_name="gpt2", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Add padding token for GPT-2
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)

        logger.info(f"Model loaded on device: {self.device}")

    def load_dataset(self, train_path="../data/train_dataset_public.csv", val_path="../data/val_dataset_public.csv"):
        """Load and preprocess the conversational dataset"""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Convert to conversational format
        train_conversations = self._create_conversations(train_df)
        val_conversations = self._create_conversations(val_df)

        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({"text": train_conversations})
        val_dataset = Dataset.from_dict({"text": val_conversations})

        return train_dataset, val_dataset

    def _create_conversations(self, df):
        """Convert DataFrame to conversational format"""
        conversations = []
        for _, row in df.iterrows():
            # Format: "User: {input} Bot: {response}"
            conversation = f"User: {row['input_text']}\nBot: {row['target_text']}"
            conversations.append(conversation)
        return conversations

    def tokenize_function(self, examples):
        """Tokenize the input texts"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

    def train(self, train_dataset, val_dataset, output_dir="../models/alu_chatbot_model",
              num_epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model"""

        # Tokenize datasets
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Not using masked language modeling
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")
        return trainer

    def load_fine_tuned_model(self, model_path="../models/alu_chatbot_model"):
        """Load the fine-tuned model"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        logger.info(f"Fine-tuned model loaded from {model_path}")

    def generate_response(self, user_input, max_new_tokens=50, temperature=0.7):
        """Generate a response to user input"""
        # Format input
        input_text = f"User: {user_input}\nBot:"

        # Tokenize
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # Generate response
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

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the bot's response
        if "Bot:" in full_response:
            response = full_response.split("Bot:")[-1].strip()
        else:
            response = full_response.replace(input_text, "").strip()

        return response

def main():
    """Main function for training and testing"""
    chatbot = ALUChatbot()

    # Load datasets
    train_dataset, val_dataset = chatbot.load_dataset()

    # Train the model
    trainer = chatbot.train(train_dataset, val_dataset)

    # Test the model
    test_inputs = [
        "What programs do you offer?",
        "How much does ALU cost?",
        "I want to apply to ALU"
    ]

    print("\nTesting the trained model:")
    for test_input in test_inputs:
        response = chatbot.generate_response(test_input)
        print(f"User: {test_input}")
        print(f"Bot: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
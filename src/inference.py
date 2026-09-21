import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class SentimentPredictor:

    MAX_LENGTH =256

    def __init__(self, model_path):

        try:
            self.model_path = model_path

            logger.info("Loading model from %s", self.model_path)

            # Use GPU when CUDA is available, otherwise use CPU.
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            logger.info("Using device: %s", self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

            self.model.eval()
            self.model.to(self.device)

            self.label_map = {
                0: "Negative",
                1: "Positive"
            }

            logger.info("Model loaded successfully")

        except Exception:
            logger.exception("Failed to load model from %s", self.model_path)
            raise
    
    

    def predict(self, text):

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[0, predicted_class].item()

        sentiment = self.label_map[predicted_class]

        return {
            "sentiment": sentiment,
            "confidence": confidence
        }

    

    def predict_batch(self, texts, batch_size=16):

        if not isinstance(texts, list):
            raise TypeError("Input must be a list of reviews.")

        if not texts:
            raise ValueError("Input list cannot be empty.")

        results = []

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            if not all(isinstance(text, str) for text in batch):
                raise TypeError("All reviews must be strings.")

            if any(not text.strip() for text in batch):
                raise ValueError("Reviews cannot be empty.")

            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.MAX_LENGTH,
                padding=True,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            predicted_classes = torch.argmax(
                probabilities,
                dim=-1
            )

            for j, predicted_class in enumerate(predicted_classes):

                class_id = predicted_class.item()

                results.append({
                    "sentiment": self.label_map[class_id],
                    "confidence": probabilities[j, class_id].item()
                })

        return results
import os
from typing import List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BatchEncoding


class NamedEntityRecogniser:

    def __init__(self, model_name: str = None):
        self.model_name = os.getenv("NER_MODEL_NAME") if model_name is None else model_name
        if self.model_name is None or self.model_name == "":
            raise ValueError(
                "A valid model name must be provided, or set as the NER_MODEL_NAME environment variable"
            )

    def classify_entities(self, input_text: str) -> dict:
        tokenised_input, input_tokens = self.tokenise_input_text(input_text)
        predictions = self.make_predictions(tokenised_input)
        output = self.map_input_tokens_to_predicted_labels(input_tokens, predictions)

        return output

    def tokenise_input_text(self, input_text: str) -> BatchEncoding:
        tokeniser = AutoTokenizer.from_pretrained(self.model_name)

        tokenised_input = tokeniser(input_text, return_tensors="pt")
        input_tokens = tokeniser.convert_ids_to_tokens(tokenised_input["input_ids"][0])

        return tokenised_input, input_tokens

    def make_predictions(self, tokenised_input: dict) -> List[str]:
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        with torch.no_grad():
            logits = model(**tokenised_input).logits

        predictions = torch.argmax(logits, dim=2)
        predicted_labels = [
            model.config.id2label[prediction.item()] for prediction in predictions[0]
        ]

        return predicted_labels

    @staticmethod
    def map_input_tokens_to_predicted_labels(
        input_tokens: List[int], predicted_labels: List[str]
    ) -> List[Tuple[str, str]]:
        mapped = list(zip(input_tokens, predicted_labels, strict=True))

        if mapped[0][0] == "[CLS]":
            del mapped[0]
        if mapped[-1][0] == "[SEP]":
            del mapped[-1]

        return mapped

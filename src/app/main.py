import argparse

from src.app.named_entity_recogniser import NamedEntityRecogniser

recogniser = NamedEntityRecogniser()


def classify_entities(input_text: str) -> dict:
    return recogniser.classify_entities(input_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify named entities in input text.")
    parser.add_argument("input_text", type=str, help="Input text in which to classify entities.")

    arguments = parser.parse_args()

    output = classify_entities(arguments.input_text)

    print(output)

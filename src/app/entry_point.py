import sys

from src.app.named_entity_recogniser import NamedEntityRecogniser

recogniser = NamedEntityRecogniser()


def classify_entities(input_text: str) -> dict:
    return recogniser.classify_entities(input_text)


if __name__ == "__main__":
    arguments = sys.argv[1:]

    if len(arguments) != 1:
        raise ValueError(
            "Too many arguments provided. Expected usage: python entry_point.py '<input_text>'"
        )

    input_text = arguments[0]
    output = classify_entities(input_text)

    print(output)

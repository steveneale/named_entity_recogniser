import pytest

from src.app.named_entity_recogniser import NamedEntityRecogniser
from src.tests.fixtures.consts import TestConsts
from src.tests.fixtures.utils import mock_model_name_env_var


class TestNamedEntityRecogniser:

    def setup_class(cls):
        cls.recogniser = NamedEntityRecogniser()

    def test_init_recogniser_is_successful(self, mock_model_name_env_var):
        # Act
        recogniser = NamedEntityRecogniser()

        # Assert
        assert isinstance(recogniser, NamedEntityRecogniser)
        assert recogniser.model_name == TestConsts.test_model_name

    @pytest.mark.parametrize(
        "model_name",
        [
            None,  # Null model name
            "",    # Empty model name
        ],
    )
    def test_init_recogniser_with_no_model_name_raises_error(
        self,
        monkeypatch,
        mock_model_name_env_var,
        model_name: str | None,
    ):
        # Arrange
        if model_name == None:
            monkeypatch.delenv("NER_MODEL_NAME")

        # Act
        with pytest.raises(ValueError) as err_info:
            _ = NamedEntityRecogniser(model_name=model_name)

        # Assert
        assert err_info.type == ValueError
        assert (
            err_info.value.args[0]
            == "A valid model name must be provided, or set as the NER_MODEL_NAME environment variable"
        )

    def test_map_input_tokens_to_predicted_labels(self):
        # Arrange
        input_tokens = ["[CLS]", "Barack", "Obama", "was", "the", "president", "of", "the", "United", "States", "[SEP]"]
        predicted_labels = ["O", "I-PER", "I-PER", "O", "O", "O", "O", "O", "I-LOC", "I-LOC", "O"]

        # Act
        mapped = NamedEntityRecogniser.map_input_tokens_to_predicted_labels(input_tokens, predicted_labels)

        # Assert
        assert len(mapped) == 9
        assert isinstance(mapped[0], tuple)
        assert mapped[0] == ("Barack", "I-PER")
        assert isinstance(mapped[8], tuple)
        assert mapped[8] == ("States", "I-LOC")

    def test_map_input_tokens_to_predicted_labels_raises_error_on_mismatch(self):
        # Arrange
        input_tokens = ["[CLS]", "Barack", "Obama", "was", "the", "president"]
        predicted_labels = ["O", "I-PER", "I-PER", "O", "O", "O", "O", "O", "I-LOC", "I-LOC", "O"]

        # Act
        with pytest.raises(ValueError) as err_info:
            NamedEntityRecogniser.map_input_tokens_to_predicted_labels(input_tokens, predicted_labels)

        # Assert
        assert err_info.type == ValueError

# named_entity_recogniser

`named_entity_recogniser` is a program for running named entity recognition on  
input text using fine-tuned, BERT-based models.


## Dependencies

`named_entity_recogniser` is written in [Python](https://www.python.org/) (version `3.13` recommended). Downloads for Python can be found at https://www.python.org/downloads/.

Dependencies can be found in `requirements.txt`. To install them, run the command:

```bash
pip install -r requirements.txt
```

Installing dependencies in a virtual environment such as [venv](https://docs.python.org/3/library/venv.html) or using a package  
manager such as [conda](https://anaconda.org/anaconda/conda) is recommended.


## Usage

Make sure that `NER_MODEL_NAME` in `.env` is set to the path of a BERT-variant  
fine-tuned for NER on [Hugging Face](https://huggingface.co/models), and then set the environment variables.  

For example, in `bash`:

```bash
set -a
. .env
set +a
```

To run inference on an input text, run the following command:

```bash
python -m src.app.main "Emmanuel Macron is the president of France."
```

Output will be a list of tuples, with one tuple for each classified token from  
the input text. The first item in each tuple is the token and the second item is  
a predicted label.

For example:

```bash
[('emmanuel', 'I-PER'), ('macro', 'I-PER'), ('##n', 'I-PER'), ('is', 'O'), ('the', 'O'), ('president', 'O'), ('of', 'O'), ('france', 'I-MISC'), ('.', 'O')]
```

## Logging

The logging level can be set in `.env`. For example:

```txt
LOG_LEVEL=INFO
```

If no `LOG_LEVEL` is set, the default logging level will be `INFO`.


## Tests

Tests can be found in the `src/tests` directory. To run them (using [pytest](https://docs.pytest.org/en/stable/)),  
run the following command:

```bash
python -m pytest -v src
```

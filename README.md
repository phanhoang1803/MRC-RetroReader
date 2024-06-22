# MRC-RetroReader

## Introduction

Machine Reading Comprehension (MRC) is one of the key challenges in Natural Language Understanding (NLU), where its task is to read and understand a given passage of text, and then answer questions based on that passage. It has various important applications such as question answering, dialogue systems, and search assistance. Recently, MRC has seen significant advancements due to the emergence of Transformers and Transformer-based language models such as BERT, GPT, T5, RoBERTa, DistilBert, and Electra.

Thanks to these advancements, MRC now requires the capability to differentiate unanswerable questions in order to provide appropriate responses instead of generating an incorrect answer. With language models that have strong contextual understanding, MRC tasks can be divided into two subtasks: 
1. determining the ability to answer questions (yes/no).
2. comprehending and providing the appropriate answer.

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributors](#contributors)
- [License](#license)

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/phanhoang1803/MRC-RetroReader.git
    cd MRC-RetroReader
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

- For notebooks: to running automatically, turn off wandb, warning if necessary:
```
wandb off
import warnings
warnings.filterwarnings('ignore')
``` 
- To train the model using the SQuAD v2 dataset:
```
python train_squad_v2.py --config path-to-yaml-file --module all --batch_size batch_size
```

- To evaluate the model on the SQuAD v2 dev set:
```
python evaluation_scriptv2.0.py path-to-dev-v2.0.json path-to-prediction.json
```

- To run demo application locally:
```
python app1.py
```

- To deploy demo application, run the following command and then follow the instructions to deploy:
```
gradio deploy
```


## Dependencies

- Python 3.x
- PyTorch
- Transformers

For a full list of dependencies, see `requirements.txt`.

## Configuration

Configuration files can be found in the `configs` directory. Adjust the parameters in these files to customize the model training and evaluation.

## Contributors
- khanhtuoimui
- phanhoang1803

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


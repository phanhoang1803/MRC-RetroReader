# MRC-RetroReader

## Introduction

Machine Reading Comprehension (MRC) is one of the key challenges in Natural Language Understanding (NLU), where its task is to read and understand a given passage of text, and then answer questions based on that passage. It has various important applications such as question answering, dialogue systems, and search assistance. Recently, MRC has seen significant advancements due to the emergence of Transformers and Transformer-based language models such as BERT, GPT, T5, RoBERTa, DistilBert, and Electra.

Thanks to these advancements, MRC now requires the capability to differentiate unanswerable questions in order to provide appropriate responses instead of generating an incorrect answer. With language models that have strong contextual understanding, MRC tasks can be divided into two subtasks: 
1. determining the ability to answer questions (yes/no).
2. comprehending and providing the appropriate answer.

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributors](#contributors)
- [License](#license)

## Demo
For a live demonstration of MRC-RetroReader in action, please visit our [demo page](https://huggingface.co/spaces/faori/HTK).

This demo allows you to interact with MRC-RetroReader by providing a passage of text and asking questions. It showcases the model's ability to comprehend the text and provide accurate answers based on the input.

In case the demo link is no longer valid or if you prefer to run the demo locally, you can follow the instructions in the [Usage](#usage) section to set up and use MRC-RetroReader on your own machine.

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
3. For pretrained weights:
MRC-RetroReader comes with pretrained weights that you can use to immediately leverage its capabilities without training from scratch. These pretrained weights are hosted on [Hugging Face Model Hub](https://huggingface.co/faori/retro_reeader/tree/main).
## Usage

- For notebooks: to running automatically, turn off wandb, warning if necessary:
```
wandb off
import warnings
warnings.filterwarnings('ignore')
``` 
- To train the model using the SQuAD v2 dataset, change the per_device_train_batch_size, and per_device_eval_batch_size in the config file to appropriate values based on your GPU memory:
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

- For Exploratory Data Analysis (EDA), please refer to "squad_eda.ipynb" for more details.

## Results

| **Backbone Encoder** 	| **Total** 	| **Total** | **HasAns** 	| **HasAns**| **NoAns** 	|
|----------------------	|:---------:	|:-----:	|:----------:	|:-----:	|:---------:	|
|                      	|     EM    	|   F1  	|     EM     	|   F1  	|   EM/F1   	|
| **DistilBert**       	|     0     	|   0   	|      0     	|   0   	|     0     	|
| **RoBERTa**          	|    78.2   	| 82.01 	|    76.9    	| 84.53 	|   79.49   	|
| **Electra Base**     	|    77.4   	| 80.97 	|    80.31   	| 87.46 	|    74.5   	|
| **Electra Large**    	|   84.46   	| 87.98 	|    81.81   	| 88.86 	|   87.11   	|


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


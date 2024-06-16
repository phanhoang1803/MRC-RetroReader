import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

from huggingface_hub import login
# login(token="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", add_to_git_credential=True)

from typing import Union, Any, Dict
# from datasets.arrow_dataset import Batch

import argparse
import datasets
from transformers.utils import logging, check_min_version
from transformers.utils.versions import require_version

from retro_reader import RetroReader
from retro_reader.constants import EXAMPLE_FEATURES
import torch

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0")

logger = logging.get_logger(__name__)


def schema_integrate(example) -> Union[Dict, Any]:
    title = example["title"]
    question = example["question"]
    context = example["context"]
    guid = example["id"]
    classtype = [""] * len(title)
    dataset_name = source = ["squad_v2"] * len(title)
    answers, is_impossible = [], []
    for answer_examples in example["answers"]:
        if answer_examples["text"]:
            answers.append(answer_examples)
            is_impossible.append(False)
        else:
            answers.append({"text": [""], "answer_start": [-1]})
            is_impossible.append(True)
    # The feature names must be sorted.
    return {
        "guid": guid,
        "question": question,
        "context": context,
        "answers": answers,
        "title": title,
        "classtype": classtype,
        "source": source,
        "is_impossible": is_impossible,
        "dataset": dataset_name,
    }


# data augmentation for multiple answers
def data_aug_for_multiple_answers(examples) -> Union[Dict, Any]:
    result = {key: [] for key in examples.keys()}
    
    def update(i, answers=None):
        for key in result.keys():
            if key == "answers" and answers is not None:
                result[key].append(answers)
            else:
                result[key].append(examples[key][i])
                
    for i, (answers, unanswerable) in enumerate(
        zip(examples["answers"], examples["is_impossible"])
    ):
        answerable = not unanswerable
        assert (
            len(answers["text"]) == len(answers["answer_start"]) or
            answers["answer_start"][0] == -1
        )
        if answerable and len(answers["text"]) > 1:
            for n_ans in range(len(answers["text"])):
                ans = {
                    "text": [answers["text"][n_ans]],
                    "answer_start": [answers["answer_start"][n_ans]],
                }
                update(i, ans)
        elif not answerable:
            update(i, {"text": [], "answer_start": []})
        else:
            update(i)
            
    return result


def main(args):
    # Load SQuAD V2.0 dataset
    print("Loading SQuAD v2.0 dataset ...")
    squad_v2 = datasets.load_dataset("squad_v2")
    
    # TODO: Visualize a sample from the dataset
    
    # Integrate into the schema used in this library
    # Note: The columns used for preprocessing are `question`, `context`, `answers`
    #       and `is_impossible`. The remaining columns are columns that exist to 
    #       process other types of data.
    
    # Minize the dataset for debugging
    if args.debug:
        squad_v2["validation"] = squad_v2["validation"].select(range(5))
    
    print("Integrating into the schema used in this library ...")
    squad_v2 = squad_v2.map(
        schema_integrate, 
        batched=True,
        remove_columns=squad_v2.column_names["train"],
        features=EXAMPLE_FEATURES,
    )
    # Load Retro Reader
    # features: parse arguments
    #           make train/eval dataset from examples
    #           load model from 🤗 hub
    #           set sketch/intensive reader and rear verifier
    print("Loading Retro Reader ...")
    retro_reader = RetroReader.load(
        config_file=args.configs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Train
    res = retro_reader.evaluate(squad_v2["validation"])
    print(res)
    logger.warning("Train retrospective reader Done.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", "-c", type=str, default="configs/inference_electra_base.yaml", help="config file path")
    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="batch size")
    parser.add_argument("--debug", "-d", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
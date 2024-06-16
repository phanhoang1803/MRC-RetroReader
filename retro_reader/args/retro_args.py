from dataclasses import dataclass, field
from .. import models

@dataclass
class RetroDataModelArguments:
    pass

@dataclass
class DataArguments(RetroDataModelArguments):
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "Maximum length of an answer (in tokens) to be generated. This is not "
            "a hard limit but the model's internal length limit."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    return_token_type_ids: bool = field(
        default=True,
        metadata={
            "help": "Whether to return token type ids."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    preprocessing_num_workers: int = field(
        default=5,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        },
    )
    version_2_with_negative: bool = field(
        default=True,
        metadata={
            "help": ""
        },
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "If null_score - best_non_null is greater than the threshold predict null."
        },
    )
    rear_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Rear threshold."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    use_choice_logits: bool = field(
        default=False,
        metadata={
            "help": "Whether to use choice logits."
        },
    )
    start_n_top: int = field(
        default=-1,
        metadata={
            "help": ""
        },
    )
    end_n_top: int = field(
        default=-1,
        metadata={
            "help": ""
        },
    )
    beta1: int = field(
        default=1,
        metadata={
            "help": ""
        },
    )
    beta2: int = field(
        default=1,
        metadata={
            "help": ""
        },
    )
    best_cof: int = field(
        default=1,
        metadata={
            "help": ""
        },
    )
    
@dataclass
class ModelArguments(RetroDataModelArguments):
    use_auth_token: bool = field(
        default=False,
        metadata={
            # "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            # "with private models)."
            "help": ""
        },
    )
    

@dataclass
class SketchModelArguments(ModelArguments):
    sketch_revision: str = field(
        default="main",
        metadata={
            "help": "The revision of the pretrained sketch model."
        },
    )
    sketch_model_name: str = field(
        default="monologg/koelectra-small-v3-discriminator",
        metadata={
            "help": "The name of the pretrained sketch model."
        },
    )
    sketch_model_mode: str = field(
        default="finetune",
        metadata={
            "help": "Choices = ['finetune', 'transfer']"
        },
    )
    sketch_tokenizer_name: str = field(
        default=None,
        metadata={
            "help": "The name of the pretrained sketch tokenizer."
        },
    )
    sketch_architectures: str = field(
        default="ElectraForSequenceClassification",
        metadata={
            "help": ""
        },
    )
    

@dataclass
class IntensiveModelArguments(ModelArguments):
    intensive_revision: str = field(
        default="main",
        metadata={
            "help": "The revision of the pretrained intensive model."
        },
    )
    intensive_model_name: str = field(
        default="monologg/koelectra-base-v3-discriminator",
        metadata={
            "help": "The name of the pretrained intensive model."
        },
    )
    intensive_model_mode: str = field(
        default="finetune",
        metadata={
            "help": "Choices = ['finetune', 'transfer']"
        },
    )
    intensive_tokenizer_name: str = field(
        default=None,
        metadata={
            "help": "The name of the pretrained intensive tokenizer."
        },
    )
    intensive_architectures: str = field(
        default="ElectraForQuestionAnsweringAVPool",
        metadata={
            "help": ""
        },
    )
    
@dataclass
class RetroArguments(DataArguments, SketchModelArguments, IntensiveModelArguments):
    def __post_init__(self):
        # Sketch 
        model_cls = getattr(models, self.sketch_architectures, None)
        if model_cls is None:
            raise ValueError(f"The sketch architecture '{self.sketch_architectures}' is not supported.")
            # raise AttributeError
        self.sketch_model_cls = model_cls
        self.sketch_model_type = model_cls.model_type
        if self.sketch_tokenizer_name is None:
            self.sketch_tokenizer_name = self.sketch_model_name
            
        # Intensive
        model_cls = getattr(models, self.intensive_architectures, None)
        if model_cls is None:
            raise AttributeError
        self.intensive_model_cls = model_cls
        self.intensive_model_type = model_cls.model_type
        
        # Tokenizer
        if self.intensive_tokenizer_name is None:
            self.intensive_tokenizer_name = self.intensive_model_name
        
        
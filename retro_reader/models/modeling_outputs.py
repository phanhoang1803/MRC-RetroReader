from typing import Optional, Tuple

import torch

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import QuestionAnsweringModelOutput

@dataclass
class QuestionAnsweringNaModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (:obj:`torch.FloatTensor`, `optional`):
            Loss of the output.
        start_logits (:obj:`torch.FloatTensor`):
            Span start logits.
        end_logits (:obj:`torch.FloatTensor`):
            Span end logits.
        has_logits (:obj:`torch.FloatTensor`):
            Has logits tensor.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`):
            Hidden states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`):
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    has_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

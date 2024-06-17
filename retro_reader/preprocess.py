import numpy as np
from .constants import (
    QUESTION_COLUMN_NAME,
    CONTEXT_COLUMN_NAME,
    ANSWER_COLUMN_NAME,
    ANSWERABLE_COLUMN_NAME,
    ID_COLUMN_NAME
)

def get_sketch_features(
    tokenizer, 
    mode, 
    data_args
):
    """
    Get the features for sketch model.

    Args:
        tokenizer (Tokenizer): Tokenizer for tokenizing input examples.
        mode (str): Mode of operation ("train", "eval", or "test").
        data_args (dict): Additional arguments for data loading.

    Returns:
        tuple: A tuple containing the function for preparing features and a boolean value indicating if labels are required.
    """

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def tokenize_fn(examples):
        """
        Tokenize input examples.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized examples.
        """
        # Tokenize the input examples using the provided tokenizer.
        # The tokenizer is configured to truncate sequences to a maximum length.
        # The tokenizer also returns the overflowing tokens, offsets mapping, and token type IDs.
        # The padding strategy is determined by the data_args.pad_to_max_length parameter.
        tokenized_examples = tokenizer(
            examples[QUESTION_COLUMN_NAME if pad_on_right else CONTEXT_COLUMN_NAME],
            examples[CONTEXT_COLUMN_NAME if pad_on_right else QUESTION_COLUMN_NAME],
            #truncation="only_second" if pad_on_right else "only_first",
            truncation=True,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            return_token_type_ids=data_args.return_token_type_ids,
            #padding="max_length" if data_args.pad_to_max_length else False,
            padding="max_length"
        )
        
        return tokenized_examples

    
    def prepare_train_features(examples):
        """
        Prepare training features by tokenizing the input examples and adding labels.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized and labeled examples.
        """
        # Tokenize the input examples using the provided tokenizer.
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Add labels to the tokenized examples.
        # The label is 0 for answerable and 1 for not answerable.
        tokenized_examples["labels"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sample_index = sample_mapping[i]
            
            # Determine if the example is answerable or not.
            is_impossible = examples[ANSWERABLE_COLUMN_NAME][sample_index]
            tokenized_examples["labels"].append(1 if is_impossible else 0)
        
        return tokenized_examples

    
    def prepare_eval_features(examples):
        """
        Prepare evaluation features by tokenizing the input examples and adding labels.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized and labeled examples.

        """
        # Tokenize the input examples using the provided tokenizer.
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Add example ids and labels to the tokenized examples.
        tokenized_examples["example_id"] = []
        tokenized_examples["labels"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            # Determine the sample index.
            sample_index = sample_mapping[i]
            
            # Extract the example id.
            id_col = examples[ID_COLUMN_NAME][sample_index]
            tokenized_examples["example_id"].append(id_col)
            
            # Determine the label.
            # answerable: 0, not answerable: 1.
            is_impossible = examples[ANSWERABLE_COLUMN_NAME][sample_index]
            tokenized_examples["labels"].append(1 if is_impossible else 0)
        
        return tokenized_examples

    
    def prepare_test_features(examples):
        """
        Prepare test features by tokenizing the input examples and adding example ids.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized and labeled examples.

        """
        # Tokenize the input examples using the provided tokenizer.
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Add example ids to the tokenized examples.
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            # Determine the sample index.
            sample_index = sample_mapping[i]
            
            # Extract the example id.
            id_col = examples[ID_COLUMN_NAME][sample_index]
            
            # Add the example id to the tokenized examples.
            tokenized_examples["example_id"].append(id_col)
        
        return tokenized_examples

    
    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_eval_features
    elif mode == "test":
        get_features_fn = prepare_test_features
    
    return get_features_fn, True

def get_intensive_features(
    tokenizer, 
    mode, 
    data_args
):
    """
    Generate intensive features for training, evaluation, or testing.

    Args:
        tokenizer (Tokenizer): The tokenizer used to tokenize the input examples.
        mode (str): The mode of operation. Must be one of "train", "eval", or "test".
        data_args (DataArguments): The data arguments containing the configuration for tokenization.

    Returns:
        tuple: A tuple containing the function to prepare the features and a boolean indicating if the tokenizer is beam-based.

    Raises:
        ValueError: If the mode is not one of "train", "eval", or "test".

    """
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    beam_based = data_args.intensive_model_type in ["xlnet", "xlm"]
    
    def tokenize_fn(examples):
        """
        Tokenize input examples.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized examples.
        """
        # Tokenize the input examples using the provided tokenizer.
        # The tokenizer is configured to truncate sequences to a maximum length.
        # The tokenizer also returns the overflowing tokens, offsets mapping, and token type IDs.
        # The padding strategy is determined by the data_args.pad_to_max_length parameter.
        
        examples[QUESTION_COLUMN_NAME] = examples[QUESTION_COLUMN_NAME].strip()
        tokenized_examples = tokenizer(
            examples[QUESTION_COLUMN_NAME if pad_on_right else CONTEXT_COLUMN_NAME],
            examples[CONTEXT_COLUMN_NAME if pad_on_right else QUESTION_COLUMN_NAME],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=data_args.return_token_type_ids,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        
        return tokenized_examples
    
    def prepare_train_features(examples):
        """
        Prepare training features by tokenizing the input examples and adding labels.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized and labeled examples.
        """
        # Tokenize the input examples using the provided tokenizer.
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Add start positions, end positions, and is_impossibles to the tokenized examples.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossibles"] = []
        
        if beam_based:
            # Add cls_index and p_mask to the tokenized examples if beam_based.
            tokenized_examples["cls_index"] = []
            tokenized_examples["p_mask"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            # Get the input_ids and cls_index for the current example.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Get the sequence_ids for the current example.
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            
            # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
            # The cls token gets 0.0 too (for predictions of empty answers).
            # Inspired by XLNet.
            if beam_based:
                tokenized_examples["cls_index"].append(cls_index)
                tokenized_examples["p_mask"].append(
                    [
                    0.0 if s == context_index or k == cls_index else 1.0
                    for s, k in enumerate(sequence_ids)
                    ]
                )
            
            # Get the sample_index, answers, and is_impossible for the current example.
            sample_index = sample_mapping[i]
            answers = examples[ANSWER_COLUMN_NAME][sample_index]
            is_impossible = examples[ANSWERABLE_COLUMN_NAME][sample_index]
            
            # If no answers are given, set the cls_index as answer.
            if is_impossible or len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossibles"].append(1.0) # unanswerable
            else:
                # Start and end token index of the current span in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # sequence_ids: 0 for question, 1 for context, None for others
                
                # Start token index of the current span in the tokenized context.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_index:
                    token_start_index += 1
                    
                # End token index of the current span in the tokenized context.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_index:
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and 
                        offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossibles"].append(1.0) # answerable
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (token_start_index < len(offsets) and 
                           offsets[token_start_index][0] <= start_char):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["is_impossibles"].append(0.0) # answerable
                    
        return tokenized_examples            
    

    def prepare_eval_features(examples):
        """
        Prepare evaluation features by tokenizing the input examples and adding labels.

        Args:
            examples (dict): Input examples.

        Returns:
            dict: Tokenized and labeled examples.
        """
        # Tokenize the input examples using the provided tokenizer.
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Add example ids to the tokenized examples.
        tokenized_examples["example_id"] = []
        
        if beam_based:
            # Add cls_index and p_mask to the tokenized examples if beam_based.
            tokenized_examples["cls_index"] = []
            tokenized_examples["p_mask"] = []
        
        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # Find the CLS index in the input_ids.
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            
            if beam_based:
                # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
                # The cls token gets 0.0 too (for predictions of empty answers).
                # Inspired by XLNet.
                tokenized_examples["cls_index"].append(cls_index)
                tokenized_examples["p_mask"].append(
                    [
                    0.0 if s == context_index or k == cls_index else 1.0
                    for s, k in enumerate(sequence_ids)
                    ]
                )
            
            sample_index = sample_mapping[i]
            id_col = examples[ID_COLUMN_NAME][sample_index]
            tokenized_examples["example_id"].append(id_col)
            
            # Set to None the offset mapping that are not part of the context
            # so it's easy to determine if a token position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_eval_features
    elif mode == "test":
        get_features_fn = prepare_eval_features
        
    return get_features_fn, True


import os
import gc
import time
import json
import math
import collections
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Callable, Any, Union

import torch
import numpy as np

# from transformers import (
#     is_datasets_available,
#     is_torch_tpu_available,
#     is_torch_xla_available,
# )

# from transformers.utils.import_utils import (
#     is_datasets_available,
#     is_torch_tpu_available,
#     is_torch_xla_available,
# )

from transformers.trainer_utils import (
    PredictionOutput,
    EvalPrediction,
    EvalLoopOutput,
    denumpify_detensorize,
    speed_metrics,
)

from transformers.utils import logging
from transformers.debug_utils import DebugOption

if is_datasets_available():
    import datasets
    
# if is_torch_xla_available():
#     import torch_xla.core.xla_model as xm       # type: ignore
#     import torch_xla.debug.metrics as met       # type: ignore

from transformers import Trainer

logger = logging.get_logger(__name__)

class ToMixin:
    def _optimizer_to(self, devide: str = "cpu"):
        """
        Move the optimizer state to the specified device.

        Args:
            devide (str, optional): The device to move the optimizer state to. Defaults to "cpu".
        """
        for param in self.optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(devide)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(devide)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(devide)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(devide)
    
    def _scheduler_to(self, devide: str = "cpu") -> None:
        """
        Move the scheduler state to the specified device.

        Args:
            devide (str, optional): The device to move the scheduler state to. Defaults to "cpu".

        Returns:
            None
        """
        for param in self.lr_scheduler.__dict__.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(devide)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(devide)
                
class BaseReader(Trainer, ToMixin):
    name: str = None
    
    def __init__(
        self, 
        *args,  # Passed to Trainer.__init__
        data_args = {},  # Additional arguments for data loading
        eval_examples: datasets.Dataset = None,  # Evaluation examples
        **kwargs  # Passed to Trainer.__init__
    ):
        """
        Initializes the BaseReader.

        Args:
            *args: Positional arguments passed to Trainer.__init__.
            data_args (dict): Additional arguments for data loading.
            eval_examples (datasets.Dataset): Evaluation examples.
            **kwargs: Keyword arguments passed to Trainer.__init__.
        """
        # Call the parent class's __init__ method with the given arguments
        super().__init__(*args, **kwargs)
        
        # Set the data_args attribute
        self.data_args = data_args
        
        # Set the eval_examples attribute
        self.eval_examples = eval_examples
    
    def free_memory(self):
        """
        Move the model, optimizer and scheduler state to the CPU, empty the CUDA cache and garbage collect.

        This method is useful to free up GPU memory before checkpointing the model or saving it to disk.
        """
        self.model.to("cpu")
        self._optimizer_to("cpu")
        self._scheduler_to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        
    def postprocess(
        self,
        output: EvalLoopOutput,
    ) -> Union[Any, PredictionOutput]:
        """
        Preprocess the evaluation loop output.

        This method is called after the evaluation loop has finished and before the evaluation metrics are computed.
        It receives the output of the evaluation loop and can be used to modify it before it is passed to the compute_metrics function.

        Args:
            output (EvalLoopOutput): The output of the evaluation loop.

        Returns:
            Union[Any, PredictionOutput]: The modified output that will be passed to the compute_metrics function.
        """
        return output

    
    def evaluate(
        self,
        eval_dataset: Optional[datasets.Dataset] = None,
        eval_examples: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Args:
            eval_dataset (Optional[datasets.Dataset], optional): The evaluation dataset. Defaults to None.
            eval_examples (Optional[datasets.Dataset], optional): The evaluation examples. Defaults to None.
            ignore_keys (Optional[List[str]], optional): Keys to ignore when calculating metrics. Defaults to None.
            metric_key_prefix (str, optional): The prefix for metric keys. Defaults to "eval".

        Returns:
            Dict[str, float]: The evaluation metrics.
        """
        
        # Start tracking memory usage
        self._memory_tracker.start()
        
        # Set eval_dataset and eval_dataloader
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Set eval_examples
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        
        # Start timing
        start_time = time.time()
        
        # Set compute_metrics
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        
        # Set eval_loop
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            # Run evaluation loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # Only gather predictions if there are metrics to compute
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # Restore compute_metrics
            self.compute_metrics = compute_metrics
        
        # Set eval_dataset format
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )
            
        # Postprocess output
        eval_preds = self.postprocess(output, eval_examples, eval_dataset, mode="evaluate")
        
        # Compute metrics
        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)
            
            # Make metrics JSON-serializable
            metrics = denumpify_detensorize(metrics)
            
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            
            # Add speed metrics
            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )
            
            # Log metrics
            self.log(metrics)
            
        # Log and save evaluation results
        filename = "eval_results.txt"
        eval_result_file = self.name + '_' + filename if self.name else filename
        with open(os.path.join(self.args.output_dir, eval_result_file), "w") as writer:
            logger.info(f"***** Eval results *****")
            writer.write("***** Eval results *****\n")
            writer.write(f"{datetime.now()}")
            for key in sorted(metrics.keys()):
                logger.info(f"  {key} = {metrics[key]}")
                writer.write(f"{key} = {metrics[key]}\n")
            writer.write("\n")
            
        # if DebugOption.TPU_METRICS_DEBUG and is_torch_xla_available():
        #     # Log debug metrics for PyTorch/XLA
        #     xm.master_print(met.metrics_report())
        
        # Call callback handler on evaluate
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        
        # Stop tracking memory usage and update metrics
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
            
    def predict(
        self,
        test_dataset: datasets.Dataset,
        test_examples: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        mode: bool = "predict",
    ) -> PredictionOutput:
        """
        Predicts on the given test dataset and returns the predictions.

        Args:
            test_dataset (datasets.Dataset): The test dataset.
            test_examples (Optional[datasets.Dataset], optional): The test examples. Defaults to None.
            ignore_keys (Optional[List[str]], optional): Keys to ignore when calculating metrics. Defaults to None.
            metric_key_prefix (str, optional): The prefix for metric keys. Defaults to "test".
            mode (bool, optional): The mode of prediction. Defaults to "predict".

        Returns:
            PredictionOutput: The predictions.
        """
        
        # Start tracking memory usage
        self._memory_tracker.start()
        
        # Get the test dataloader
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        
        # Set compute_metrics to None and store it for later use
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        
        # Get the evaluation loop
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            # Run the evaluation loop
            output = eval_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # Reset compute_metrics to its original value
            self.compute_metrics = compute_metrics
        
        # If the test dataset is a datasets.Dataset, set its format
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )
            
        # Postprocess the output and return the predictions
        predictions = self.postprocess(output, test_examples, test_dataset, mode=mode)
        
        # Stop tracking memory usage and update metrics
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        
        return predictions

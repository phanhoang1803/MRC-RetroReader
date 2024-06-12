import datasets
import evaluate
from transformers.trainer_utils import EvalPrediction

# accuracy = evaluate.load("accuracy").compute
# precision = evaluate.load("precision").compute
# recall = evaluate.load("recall").compute
# f1 = evaluate.load("f1").compute
# squad_v2 = evaluate.load("squad_v2").compute

accuracy = datasets.load_metric("accuracy").compute
precision = datasets.load_metric("precision").compute
recall = datasets.load_metric("recall").compute
f1 = datasets.load_metric("f1").compute
squad_v2 = datasets.load_metric("squad_v2").compute

def compute_classification_metric(p: EvalPrediction):
    """
    Compute classification metrics for a given prediction.

    Args:
        p (EvalPrediction): The prediction object.

    Returns:
        datasets.Metric: The metric object containing accuracy, precision,
        recall, and f1 score.
    """
    # Get the predicted class labels and the reference labels
    predictions = p.predictions.argmax(axis=1)
    references = p.label_ids    
    
    # Initialize the metric object
    metric = accuracy(predictions, references)
    
    # Update the metric with precision, recall, and f1 score
    metric.update(precision(predictions, references))
    metric.update(recall(predictions, references))
    metric.update(f1(predictions, references))
    
    # Return the metric object
    return metric


def compute_squad_v2(p: EvalPrediction):
    """
    Compute SQuAD v2 metrics for a given prediction.

    Args:
        p (EvalPrediction): The prediction object.

    Returns:
        datasets.Metric: The metric object containing SQuAD v2 metrics.
    """
    # Get the predicted answers and the reference answers
    predictions = p.predictions
    references = p.label_ids
    
    # Compute and return the SQuAD v2 metrics
    return squad_v2(predictions, references)


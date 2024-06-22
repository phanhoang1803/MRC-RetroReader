import gradio as gr
import io
import os
import yaml
import pyarrow
import tokenizers
from retro_reader import RetroReader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def from_library():
    from retro_reader import constants as C
    return C, RetroReader

C, RetroReader = from_library()

# Assuming RetroReader.load is a method from your imports
def load_model(config_path):
    return RetroReader.load(config_file=config_path)

# Loading models
model_electra_base = load_model("configs/inference_en_electra_base.yaml")
model_electra_large = load_model("configs/inference_en_electra_large.yaml")
model_roberta = load_model("configs/inference_en_roberta.yaml")
model_distilbert = load_model("configs/inference_en_distilbert.yaml")

def retro_reader_demo(query, context, model_choice):
    # Choose the model based on the model_choice
    if model_choice == "Electra Base":
        model = model_electra_base
    elif model_choice == "Electra Large":
        model = model_electra_large
    elif model_choice == "Roberta":
        model = model_roberta
    elif model_choice == "DistilBERT":
        model = model_distilbert
    else:
        return "Invalid model choice"

    # Generate outputs using the chosen model
    outputs = model(query=query, context=context, return_submodule_outputs=True)
    
    # Extract the answer
    answer = outputs[0]["id-01"] if outputs[0]["id-01"] else "No answer found"
    
    return answer

# Gradio app interface
iface = gr.Interface(
    fn=retro_reader_demo,
    inputs=[
        gr.Textbox(label="Query", placeholder="Type your query here..."),
        gr.Textbox(label="Context", placeholder="Provide the context here...", lines=10),
        gr.Radio(choices=["Electra Base", "Electra Large", "Roberta", "DistilBERT"], label="Model Choice")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Retrospective Reader Demo",
    description="This interface uses the RetroReader model to perform reading comprehension tasks."
)

if __name__ == "__main__":
    iface.launch(share=True)

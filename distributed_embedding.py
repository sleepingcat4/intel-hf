import torch
from transformers import AutoTokenizer
from accelerate import PartialState
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
import intel_extension_for_pytorch as ipex

sentences = ["what's the capital of England?", "what is the tallest mountain?", "who is the president of the USA?"]

model_name = "BAAI/bge-m3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="xpu", trust_remote_code=True, use_llm_runtime=False)

distributed_state = PartialState()

device = torch.device(f"xpu:{distributed_state.process_index}")
model.to(device)

if distributed_state.process_index == 0:
    subset_sentences = ["what's the capital of England?", "what is the tallest mountain?"]
elif distributed_state.process_index == 1:
    subset_sentences = ["who is the president of the USA?", "who is the president of the USA?"]
else:
    subset_sentences = []

if subset_sentences:
    subset_inputs = tokenizer(subset_sentences, return_tensors="pt", padding=True, truncation=True)
    subset_inputs = {key: tensor.to(device) for key, tensor in subset_inputs.items()}

    with torch.no_grad():
        outputs = model(**subset_inputs)
        logits = outputs.logits

    embeddings = logits.mean(dim=1)

    print(f"Process {distributed_state.process_index} embeddings:")
    print(embeddings)
else:
    print(f"Process {distributed_state.process_index} has no sentences to process.")

from sentence_transformers import SentenceTransformer
import torch
import os

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
model.eval()

transformer = model[0].auto_model
tokenizer = model.tokenizer

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

print("Exporting model to ONNX...")
output_file = "model.onnx"

torch.onnx.export(
    transformer,
    (inputs['input_ids'], inputs['attention_mask']),
    output_file,
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
    },
    opset_version=17 
)

print(f"Created '{output_file}'!")

print("Saving vocab...")
try:
    tokenizer.save_pretrained(".")
    if os.path.exists("vocab.txt"):
        print("Saved 'vocab.txt' (Method 1)")
    else:
        raise Exception("File not found")
except:
    print("Standard save failed, extracting manually...")
    with open("vocab.txt", "w", encoding="utf-8") as f:
        vocab = tokenizer.get_vocab() # Get the dict
        sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
        for token, index in sorted_vocab:
            f.write(token + "\n")
    print("Saved 'vocab.txt' (Method 2)")
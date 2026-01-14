import torch
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name)
model.eval()

class ModelWithPooling(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.transformer = original_model[0].auto_model

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

print("Exporting smarter model...")
full_model = ModelWithPooling(model)

#  input
text = "Hello world"
tokenizer = model.tokenizer
inputs = tokenizer(text, return_tensors="pt")

torch.onnx.export(
    full_model,
    (inputs['input_ids'], inputs['attention_mask']),
    "model.onnx", 
    input_names=['input_ids', 'attention_mask'],
    output_names=['sentence_embedding'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'}
    },
    opset_version=17
)

print("Created 'model.onnx' (With Mean Pooling baked in!)")
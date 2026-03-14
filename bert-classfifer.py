from transformers import AutoTokenizer, AutoModel
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig
from torchao.dtypes import PlainLayout
import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login



load_dotenv()

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-uncased',
    do_lower_case=True,
    model_max_length=512,
    padding_side="right",
    truncation_side="right",
    use_fast=True,
)

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path="bert-base-uncased",
    output_hidden_states=False,   # don't return all 12 layer hidden states
    output_attentions=False,      # don't return attention weight matrices
    add_pooling_layer=True,       # include the [CLS] pooler head on top
    torch_dtype=torch.float32,    # dtype before quantization; keep float32 here
    attn_implementation="eager",  # "eager" | "sdpa" | "flash_attention_2"
)

model.eval()

quantize_(
    model,
    Int8DynamicActivationInt8WeightConfig(
        layout=PlainLayout(),      # ✓ must be explicit, None crashes
    ),
)

def get_embeddings(texts: list[str]) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    with torch.no_grad():
        outputs = model(**encoded)

    # Mean pool over non-padding tokens
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return embeddings  # shape: (batch_size, 768)


texts = ["The cat sat on the mat.", "BERT embeddings are useful."]
embeddings = get_embeddings(texts)
print(embeddings)  # → torch.Size([2, 768])

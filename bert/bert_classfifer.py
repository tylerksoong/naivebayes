from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            do_lower_case=True,
            model_max_length=512,
            padding_side="right",
            truncation_side="right",
            use_fast=True,
        )
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            output_hidden_states=False,
            output_attentions=False,
            add_pooling_layer=True,       # exposes pooler_output ([CLS] → Linear + Tanh)
            attn_implementation="eager",  # sdpa is not fully stable on MPS yet
        ).to(device)

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # Single linear head as discussed — BERT's representations are rich
        # enough that a deeper MLP would overfit and destabilize fine-tuning
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, 1).to(device)  # single logit for BCEWithLogitsLoss

        for param in self.bert.parameters():
            param.requires_grad = False

        

        self.criterion = nn.BCEWithLogitsLoss()  # fuses sigmoid + BCE for numerical stability
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=1e-3
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # pooler_output: [CLS] token passed through Linear(768,768) + Tanh
        # shape: (batch_size, 768)
        cls_repr = self.dropout(outputs.pooler_output)
        return self.classifier(cls_repr).squeeze(-1)  # (batch_size,) — BCEWithLogitsLoss expects this shape

    def _tokenize(self, texts: list[str]):
        """Tokenize a list of strings and move tensors to device."""
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        # Move all tokenizer outputs to MPS in one call
        return {k: v.to(device) for k, v in encoded.items()}

    def fit(self, loader, epochs: int = 3, save_dir: str = "checkpoints"):
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.train()


        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device)
                labels = batch["labels"].float().to(device)

                self.optimizer.zero_grad()
                logits = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Print every 10 batches so you know it's alive
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(
                        f"Epoch {epoch+1}/{epochs} "
                        f"| Batch {batch_idx}/{len(loader)} "
                        f"| Avg Loss: {avg_loss:.4f}"
                    )
                    torch.save(
                        self.state_dict(),
                        os.path.join(save_dir, f"epoch{epoch+1}_batch{batch_idx}.pt"),
                    )

        print(f"Epoch {epoch+1} complete — avg loss: {total_loss / len(loader):.4f}")
        
    def predict(self, texts: list[str]) -> list[int]:
        self.eval()
        encoded = self._tokenize(texts)
        with torch.no_grad():
            logits = self(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["token_type_ids"],
            )
        # sigmoid > 0.5 threshold → binary prediction
        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().cpu().tolist()

    def predict_proba(self, texts: list[str]) -> list[float]:
        """Returns raw probabilities — useful for threshold tuning."""
        self.eval()
        encoded = self._tokenize(texts)
        with torch.no_grad():
            logits = self(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["token_type_ids"],
            )
        return torch.sigmoid(logits).cpu().tolist()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # Tokenize everything upfront
        encoded = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": torch.zeros(self.input_ids[idx].shape[0], dtype=torch.long),  # BERT expects this
            "labels": self.labels[idx],
        }
    
    def __len__(self):
        return len(self.labels)

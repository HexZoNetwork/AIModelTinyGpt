import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import ByteLevelBPETokenizer
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import argparse

# --- Impor model multimodal dari Canvas Anda ---
from modelvisual import MultimodalAgent

# ==========================================================
# BAGIAN 1: KELAS DATASET VISUAL
# Dirancang untuk memuat pasangan gambar dan caption.
# ==========================================================
class VisualDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, transform, seq_len=128):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.seq_len = seq_len
        
        # Baca file CSV yang berisi nama file gambar dan caption-nya
        print(f"📚 Membaca captions dari {captions_file}...")
        self.captions_df = pd.read_csv(captions_file)
        print(f"✅ Ditemukan {len(self.captions_df)} pasangan gambar-caption.")

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        # Ambil data dari baris ke-idx
        row = self.captions_df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_filename'])
        caption = row['caption']
        
        # Muat dan proses gambar
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except FileNotFoundError:
            print(f"⚠️ Gambar tidak ditemukan: {image_path}. Melewatkan item ini.")
            # Return item dummy jika gambar tidak ada
            return self.__getitem__((idx + 1) % len(self))

        # Tokenisasi caption
        tokenized_caption = self.tokenizer.encode(caption).ids
        
        # Siapkan input dan target, padding jika perlu
        padded_tokens = tokenized_caption + [0] * (self.seq_len - len(tokenized_caption))
        tokens = torch.tensor(padded_tokens[:self.seq_len], dtype=torch.long)
        
        # Target digeser satu posisi
        targets = torch.tensor(padded_tokens[1:self.seq_len+1], dtype=torch.long)
        
        return image_tensor, tokens, targets

# ==========================================================
# BAGIAN 2: FUNGSI TRAINING UTAMA
# ==========================================================
def train_visual_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🔥 Menggunakan device: {args.device}")

    # 1. Setup Tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_path, "vocab.json"),
        os.path.join(args.tokenizer_path, "merges.txt")
    )
    
    # 2. Setup Transformasi Gambar
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Setup Dataset
    full_dataset = VisualDataset(args.image_dir, args.captions_file, tokenizer, transform, args.max_seq_len)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4. Setup Model Multimodal
    model_config = {
        'vocab_size': args.vocab_size, 'n_emb': args.n_emb, 'n_layer': args.n_layer,
        'n_head': args.n_head, 'n_kv_heads': args.n_kv_heads, 'max_seq_len': args.max_seq_len,
        'num_experts': args.num_experts, 'num_experts_per_tok': args.num_experts_per_tok,
        'sliding_window_size': args.sliding_window_size
    }
    model = MultimodalAgent(model_config).to(args.device)
    
    # Opsional: Load bobot dari model bahasa yang sudah dilatih
    if args.load_llm_checkpoint:
        print(f"🧠 Memuat bobot LLM dari {args.load_llm_checkpoint}...")
        model.language_model.load_state_dict(torch.load(args.load_llm_checkpoint, map_location=args.device))

    # 5. Training Loop
    optim = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs)
    best_val_loss = float('inf')

    for ep in range(args.epochs):
        model.train()
        loop = tqdm(train_dl, desc=f"Epoch {ep+1}/{args.epochs}")
        for images, tokens, targets in loop:
            images, tokens, targets = images.to(args.device), tokens.to(args.device), targets.to(args.device)
            optim.zero_grad()
            _, loss = model(text_tokens=tokens, image=images, targets=targets)
            if loss is not None:
                loss.backward()
                optim.step()
                loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, tokens, targets in val_dl:
                images, tokens, targets = images.to(args.device), tokens.to(args.device), targets.to(args.device)
                _, loss = model(text_tokens=tokens, image=images, targets=targets)
                if loss is not None: val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dl)
        print(f"Epoch {ep+1}: Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "best_multimodal_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   🏆 Model multimodal terbaik disimpan ke {save_path}")

        scheduler.step()

# ==========================================================
# BAGIAN 3: ARGUMENT PARSER
# ==========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Multimodal Agent.")
    parser.add_argument("--image_dir", type=str, required=True, help="Direktori berisi gambar.")
    parser.add_argument("--captions_file", type=str, required=True, help="File .csv dengan kolom 'image_filename' dan 'caption'.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--output_dir", type=str, default="multimodal_checkpoints")
    parser.add_argument("--load_llm_checkpoint", type=str, default=None, help="Path opsional ke checkpoint LLM pre-trained.")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)

    # Model Config (sesuaikan dengan LLM yang di-load)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--n_emb", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--sliding_window_size", type=int, default=256)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_visual_model(args)

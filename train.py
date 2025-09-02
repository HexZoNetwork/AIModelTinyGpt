import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import ByteLevelBPETokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import glob

# --- ✨ UPGRADE: Impor model v6 dari file Canvas Anda ---
# Pastikan file ini berada di direktori yang sama dengan 'rag_memory_system.py'
from model import TinyGPT_v6

# ==========================================================
# BAGIAN 1: KELAS DATASET
# Menggunakan LineDataset yang Anda berikan untuk tokenisasi yang efisien.
# ==========================================================
class LineDataset(Dataset):
    def __init__(self, files, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.ids = []
        
        # Tokenisasi file satu per satu untuk manajemen memori yang lebih baik
        print("💾 Memulai tokenisasi dataset...")
        for f in tqdm(files, desc="Processing files"):
            with open(f, 'r', encoding='utf-8') as fh:
                # Membaca seluruh file lalu tokenisasi, lebih cepat untuk banyak file kecil
                text = fh.read()
                tokens = tokenizer.encode(text).ids
                self.ids.extend(tokens)
        print(f"✅ Tokenisasi selesai. Total token: {len(self.ids):,}")

    def __len__(self):
        # Pastikan kita tidak mencoba mengakses indeks di luar batas
        return len(self.ids) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Input (x) adalah sekuens dari start_idx
        # Target (y) adalah sekuens yang digeser satu token ke kanan
        x = torch.tensor(self.ids[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(self.ids[start_idx+1:end_idx+1], dtype=torch.long)
        return x, y

# ==========================================================
# BAGIAN 2: KONFIGURASI TRAINING
# Semua parameter penting ada di sini agar mudah diubah.
# ==========================================================
class TrainingConfig:
    # Lokasi data & model
    FILES = sorted(glob.glob("data/wiki_part*.txt")) or ["data/wiki.txt"]
    TOKENIZER_PATH = "tokenizer" # Folder berisi vocab.json & merges.txt
    CHECKPOINT_DIR = "checkpoints"

    # Hyperparameters Training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 2
    
    # Konfigurasi Model (sesuaikan dengan model terbaik dari hyper-tuning)
    VOCAB_SIZE = 30000 
    MAX_SEQ_LEN = 256
    N_EMB = 512
    N_LAYER = 8
    N_HEAD = 8
    N_KV_HEADS = 4
    NUM_EXPERTS = 8
    NUM_EXPERTS_PER_TOK = 2
    SLIDING_WINDOW_SIZE = 256

    # Early Stopping
    PATIENCE = 3

# ==========================================================
# BAGIAN 3: SCRIPT TRAINING UTAMA
# ==========================================================
def main():
    config = TrainingConfig()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"🔥 Menggunakan device: {config.DEVICE}")

    # 1. Setup Tokenizer
    if not os.path.exists(os.path.join(config.TOKENIZER_PATH, "vocab.json")):
        raise FileNotFoundError("Tokenizer tidak ditemukan! Latih tokenizer terlebih dahulu.")
    
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(config.TOKENIZER_PATH, "vocab.json"),
        os.path.join(config.TOKENIZER_PATH, "merges.txt")
    )

    # 2. Setup Dataset & DataLoader
    full_dataset = LineDataset(config.FILES, tokenizer, seq_len=config.MAX_SEQ_LEN)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"📚 Ukuran dataset: {len(train_ds)} train, {len(val_ds)} validasi.")

    # 3. Setup Model, Optimizer, & Scheduler
    model = TinyGPT_v6(
        vocab_size=config.VOCAB_SIZE,
        n_emb=config.N_EMB,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_kv_heads=config.N_KV_HEADS,
        max_seq_len=config.MAX_SEQ_LEN,
        num_experts=config.NUM_EXPERTS,
        num_experts_per_tok=config.NUM_EXPERTS_PER_TOK,
        sliding_window_size=config.SLIDING_WINDOW_SIZE
    ).to(config.DEVICE)
    print(f"🤖 Jumlah parameter model: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optim = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optim, T_max=config.EPOCHS)

    # 4. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0

    for ep in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_dl, desc=f"Epoch {ep+1}/{config.EPOCHS}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            optim.zero_grad()
            
            _, loss = model(xb, targets=yb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            
            train_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                _, loss = model(xb, targets=yb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dl)

        scheduler.step()
        print(f"Epoch {ep+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"   🏆 Model terbaik disimpan ke {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"   ⏹️ Early stopping terpicu setelah {config.PATIENCE} epoch tanpa peningkatan.")
            break

    print("\n🎉 Proses training selesai!")

if __name__ == "__main__":
    main()

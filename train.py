import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import ByteLevelBPETokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import glob

# --- ✨ UPGRADE: Impor model v6 dari file Canvas Anda ---
# Pastikan file ini berada di direktori yang sama dengan 'model.py'
from model import TinyGPT_v6

# ==========================================================
# BAGIAN 1: KELAS DATASET
# ==========================================================
class LineDataset(Dataset):
    def __init__(self, files, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.ids = []
        
        print("💾 Memulai tokenisasi dataset...")
        for f in tqdm(files, desc="Processing files"):
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read()
                tokens = tokenizer.encode(text).ids
                self.ids.extend(tokens)
        print(f"✅ Tokenisasi selesai. Total token: {len(self.ids):,}")

    def __len__(self):
        return len(self.ids) // (self.seq_len + 1)

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        x = torch.tensor(self.ids[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(self.ids[start_idx+1:end_idx+1], dtype=torch.long)
        return x, y

# ==========================================================
# BAGIAN 2: KONFIGURASI TRAINING
# ==========================================================
class TrainingConfig:
    FILES = sorted(glob.glob("data/wiki_part*.txt")) or ["data/wiki.txt"]
    TOKENIZER_PATH = "tokenizer"
    CHECKPOINT_DIR = "checkpoints"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    
    # --- ✨ FITUR BARU: Konfigurasi Adaptif ---
    # Default untuk GPU
    BATCH_SIZE = 2
    NUM_WORKERS = 2 
    VOCAB_SIZE = 30000 
    MAX_SEQ_LEN = 256
    N_EMB = 512
    N_LAYER = 8
    N_HEAD = 8
    N_KV_HEADS = 4
    NUM_EXPERTS = 8
    NUM_EXPERTS_PER_TOK = 2
    SLIDING_WINDOW_SIZE = 256
    PATIENCE = 3

    def __init__(self):
        # Jika terdeteksi CPU, gunakan konfigurasi "ringan"
        if self.DEVICE == 'cpu':
            print("⚠️ Terdeteksi environment CPU. Menggunakan konfigurasi 'ringan' untuk mempercepat proses.")
            print("   Model akan lebih kecil dan batch size dikurangi. Ini cocok untuk debugging atau environment terbatas.")
            self.BATCH_SIZE = 1
            self.NUM_WORKERS = 0 # Di CPU, num_workers > 0 bisa memperlambat
            self.N_EMB = 256    # Kurangi ukuran embedding
            self.N_LAYER = 4    # Kurangi jumlah layer
            self.N_HEAD = 4     # Kurangi jumlah head
            self.N_KV_HEADS = 2 # Sesuaikan KV heads
            self.NUM_EXPERTS = 4 # Kurangi jumlah expert

# ==========================================================
# BAGIAN 3: SCRIPT TRAINING UTAMA
# ==========================================================
def main():
    config = TrainingConfig()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"🔥 Menggunakan device: {config.DEVICE}")

    tokenizer = ByteLevelBPETokenizer(
        os.path.join(config.TOKENIZER_PATH, "vocab.json"),
        os.path.join(config.TOKENIZER_PATH, "merges.txt")
    )

    full_dataset = LineDataset(config.FILES, tokenizer, seq_len=config.MAX_SEQ_LEN)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    print(f"📚 Ukuran dataset: {len(train_ds)} train, {len(val_ds)} validasi.")

    model = TinyGPT_v6(
        vocab_size=config.VOCAB_SIZE, n_emb=config.N_EMB, n_layer=config.N_LAYER,
        n_head=config.N_HEAD, n_kv_heads=config.N_KV_HEADS, max_seq_len=config.MAX_SEQ_LEN,
        num_experts=config.NUM_EXPERTS, num_experts_per_tok=config.NUM_EXPERTS_PER_TOK,
        sliding_window_size=config.SLIDING_WINDOW_SIZE
    ).to(config.DEVICE)
    print(f"🤖 Jumlah parameter model: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optim = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optim, T_max=config.EPOCHS)

    start_epoch = 0
    best_val_loss = float('inf')
    resume_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pt")

    if os.path.exists(resume_checkpoint_path):
        print(f"🔄 Melanjutkan training dari checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=config.DEVICE)
        # ✨ FIX: Periksa jika arsitektur checkpoint cocok dengan konfigurasi saat ini
        # Ini mencegah error saat beralih dari mode CPU (model kecil) ke GPU (model besar) atau sebaliknya
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"   Melanjutkan dari epoch {start_epoch}, best_val_loss sebelumnya: {best_val_loss:.4f}")
        except RuntimeError as e:
            print(f"⚠️ Gagal memuat checkpoint karena arsitektur tidak cocok. Memulai dari awal. Error: {e}")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        print("🚀 Memulai training dari awal.")
    
    patience_counter = 0
    for ep in range(start_epoch, config.EPOCHS):
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

        checkpoint = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, resume_checkpoint_path)

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


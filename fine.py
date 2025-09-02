import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import ByteLevelBPETokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import glob
import argparse

# --- ✨ UPGRADE: Impor model v6 dari file Canvas Anda ---
from model import TinyGPT_v6

# ==========================================================
# BAGIAN 1: KELAS DATASET
# ==========================================================
class LineDataset(Dataset):
    def __init__(self, files, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.ids = []
        
        print("💾 Memulai tokenisasi dataset fine-tuning...")
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
# BAGIAN 2: FUNGSI FINE-TUNING UTAMA
# ==========================================================
def fine_tune_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🔥 Menggunakan device: {args.device}")

    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_path, "vocab.json"),
        os.path.join(args.tokenizer_path, "merges.txt")
    )

    # --- ✨ FITUR BARU: Deteksi CPU Adaptif ---
    num_workers = 2 if args.device == 'cuda' else 0
    print(f"   Menggunakan {num_workers} worker processes.")
    
    fine_tune_files = [args.dataset_path]
    full_dataset = LineDataset(fine_tune_files, tokenizer, seq_len=args.max_seq_len)
    
    val_size = min(int(0.1 * len(full_dataset)), 500)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    print(f"📚 Ukuran dataset fine-tuning: {len(train_ds)} train, {len(val_ds)} validasi.")

    model = TinyGPT_v6(
        vocab_size=args.vocab_size, n_emb=args.n_emb, n_layer=args.n_layer,
        n_head=args.n_head, n_kv_heads=args.n_kv_heads, max_seq_len=args.max_seq_len,
        num_experts=args.num_experts, num_experts_per_tok=args.num_experts_per_tok,
        sliding_window_size=args.sliding_window_size
    ).to(args.device)

    optim = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs)
    
    # --- ✨ FITUR BARU: Sistem Checkpoint & Resume ---
    start_epoch = 0
    best_val_loss = float('inf')
    resume_checkpoint_path = os.path.join(args.output_dir, "latest_finetune_checkpoint.pt")

    # Muat checkpoint model pre-trained terlebih dahulu
    if os.path.exists(args.checkpoint_path):
        print(f"✅ Memuat checkpoint pre-trained dari: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    else:
        print(f"⚠️ Peringatan: Checkpoint pre-trained tidak ditemukan. Memulai fine-tuning dari nol.")

    # Kemudian, periksa apakah ada checkpoint proses fine-tuning untuk dilanjutkan
    if os.path.exists(resume_checkpoint_path):
        print(f"🔄 Melanjutkan proses fine-tuning dari: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict']) # Timpa lagi dengan state terakhir
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Melanjutkan dari epoch {start_epoch}, best_val_loss sebelumnya: {best_val_loss:.4f}")

    print(f"🤖 Jumlah parameter model: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    for ep in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_dl, desc=f"Fine-Tuning Epoch {ep+1}/{args.epochs}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(args.device), yb.to(args.device)
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
                xb, yb = xb.to(args.device), yb.to(args.device)
                _, loss = model(xb, targets=yb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dl) if len(val_dl) > 0 else 0.0

        scheduler.step()
        print(f"Epoch {ep+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Simpan checkpoint komprehensif
        checkpoint = {
            'epoch': ep, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, resume_checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "fine_tuned_best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   🏆 Model fine-tuned terbaik disimpan ke {save_path}")

    print("\n🎉 Proses fine-tuning selesai!")


# ==========================================================
# BAGIAN 4: ARGUMENT PARSER & ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained TinyGPT-v6 model.")
    
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Path ke model pre-trained yang akan di-fine-tune.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path ke file .txt dataset untuk fine-tuning.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer", help="Path ke folder tokenizer.")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Direktori untuk menyimpan model hasil fine-tuning.")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch untuk fine-tuning.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (biasanya lebih kecil untuk fine-tuning).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    # Model Architecture Arguments (HARUS SESUAI DENGAN CHECKPOINT)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--n_emb", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--sliding_window_size", type=int, default=256)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fine_tune_model(args)


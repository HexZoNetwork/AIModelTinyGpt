import itertools
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import glob
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Impor dari file proyek Anda ---
from model import TinyGPT_v6
from dataset import TextDataset

# ==========================================================
# ✨ UPGRADE: Hyperparameter grid disesuaikan untuk TinyGPT_v6
# ==========================================================
hp_configs = [
    {
        "run_name": "Small_Model_Fast_LR",
        "lr": 5e-4, "batch_size": 2, "max_seq_len": 256,
        "n_emb": 384, "n_layer": 6, "n_head": 6, "n_kv_heads": 2,
        "num_experts": 4, "num_experts_per_tok": 2, "sliding_window_size": 128
    },
    {
        "run_name": "Medium_Model_Slow_LR",
        "lr": 1e-4, "batch_size": 2, "max_seq_len": 256,
        "n_emb": 512, "n_layer": 8, "n_head": 8, "n_kv_heads": 4,
        "num_experts": 8, "num_experts_per_tok": 2, "sliding_window_size": 256
    },
    {
        "run_name": "Large_Context_Model",
        "lr": 3e-4, "batch_size": 1, "max_seq_len": 512,
        "n_emb": 512, "n_layer": 8, "n_head": 8, "n_kv_heads": 2,
        "num_experts": 8, "num_experts_per_tok": 2, "sliding_window_size": 256
    },
]

# Konfigurasi Global
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 30000 
MAX_EPOCHS = 10
FILES = sorted(glob.glob("data/wiki_part*.txt")) or ["data/wiki.txt"]
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================
# Jalankan semua kombinasi
# ==========================
for run_id, hp in enumerate(hp_configs):
    run_num = run_id + 1
    print(f"\n🚀 Memulai Run {run_num}: {hp['run_name']}")
    print(f"   Hyperparameters: {hp}")

    logdir = f"runs/{hp['run_name']}_run{run_num}"
    writer = SummaryWriter(logdir=logdir)

    # --- ✨ FITUR BARU: Deteksi CPU Adaptif ---
    num_workers = 2 if DEVICE == 'cuda' else 0
    print(f"   Menggunakan device '{DEVICE}' dengan {num_workers} worker processes.")

    # Dataset
    ds = TextDataset(FILES, seq_len=hp["max_seq_len"])
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False, num_workers=num_workers)

    # Model
    model = TinyGPT_v6(
        vocab_size=VOCAB_SIZE, n_emb=hp["n_emb"], n_layer=hp["n_layer"],
        n_head=hp["n_head"], n_kv_heads=hp["n_kv_heads"], max_seq_len=hp["max_seq_len"],
        num_experts=hp["num_experts"], num_experts_per_tok=hp["num_experts_per_tok"],
        sliding_window_size=hp["sliding_window_size"]
    ).to(DEVICE)
    print(f"   Jumlah parameter: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optim = AdamW(model.parameters(), lr=hp["lr"])
    scheduler = CosineAnnealingLR(optim, T_max=MAX_EPOCHS)
    
    # --- ✨ FITUR BARU: Sistem Checkpoint Per-Run ---
    start_epoch = 0
    best_val_loss = float("inf")
    resume_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"latest_checkpoint_run{run_num}.pt")

    if os.path.exists(resume_checkpoint_path):
        print(f"🔄 Melanjutkan Run {run_num} dari checkpoint...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Melanjutkan dari epoch {start_epoch}, best_val_loss sebelumnya: {best_val_loss:.4f}")
    else:
        print("   Memulai run dari awal.")

    # Training loop
    for ep in range(start_epoch, MAX_EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Run{run_num} Ep{ep+1}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            _, loss = model(xb, targets=yb)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0.0
        writer.add_scalar("Loss/train", avg_train_loss, ep)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                _, loss = model(xb, targets=yb)
                if loss is not None:
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dl) if len(val_dl) > 0 else 0.0
        writer.add_scalar("Loss/val", avg_val_loss, ep)

        print(f"[Run{run_num}] Ep{ep+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Simpan checkpoint komprehensif
        checkpoint = {
            'epoch': ep, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, resume_checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CHECKPOINT_DIR, f"best_model_run{run_num}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ Model terbaik disimpan ke {save_path} (val_loss={best_val_loss:.4f})")

        scheduler.step()

    writer.add_hparams(
        {k: v for k, v in hp.items() if isinstance(v, (int, float, str))},
        {"hparam/best_val_loss": best_val_loss}
    )
    writer.close()

print("\n🎉 Semua proses hyperparameter tuning selesai.")


import torch
from tokenizers import ByteLevelBPETokenizer
import argparse
import os

# --- ✨ UPGRADE: Impor model v6 dari file Canvas Anda ---
# Pastikan file ini berada di direktori yang sama dengan 'rag_memory_system.py'
from model import TinyGPT_v6

# ==========================================================
# BAGIAN 1: FUNGSI GENERASI UTAMA
# ==========================================================
@torch.no_grad()
def generate_text(args):
    """Fungsi utama untuk memuat model dan menghasilkan teks."""

    print(f"🔥 Menggunakan device: {args.device}")

    # 1. Setup Tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_path, "vocab.json"),
        os.path.join(args.tokenizer_path, "merges.txt")
    )

    # 2. Setup Model dengan arsitektur yang sesuai
    model = TinyGPT_v6(
        vocab_size=args.vocab_size,
        n_emb=args.n_emb,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        sliding_window_size=args.sliding_window_size
    ).to(args.device)

    # 3. Load Checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di {args.checkpoint_path}. Harap periksa path-nya.")
    
    print(f"✅ Memuat checkpoint dari: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    model.eval() # Set model ke mode evaluasi

    # 4. Tokenisasi Prompt
    prompt_ids = tokenizer.encode(args.prompt).ids
    context_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    # 5. Generate Teks menggunakan method .generate() dari model
    print("\n🚀 Menghasilkan teks...")
    generated_tokens = model.generate(
        context_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    # 6. Decode dan Tampilkan Hasil
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    
    print("-" * 50)
    print("💬 Output Model:")
    print(generated_text)
    print("-" * 50)


# ==========================================================
# BAGIAN 2: ARGUMENT PARSER & ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained TinyGPT-v6 model.")
    
    # Path Arguments
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Path ke model .pt yang sudah dilatih.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer", help="Path ke folder tokenizer.")

    # Generation Arguments
    parser.add_argument("--prompt", type=str, required=True, help="Teks awal untuk memulai generasi.")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Jumlah maksimum token baru yang akan digenerate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Mengontrol kreativitas. Nilai lebih tinggi = lebih acak.")
    parser.add_argument("--top_k", type=int, default=50, help="Sampling dari k token paling mungkin. 0 untuk menonaktifkan.")

    # Model Architecture Arguments (HARUS SESUAI DENGAN CHECKPOINT)
    # Ini adalah parameter dari model terbaik yang kita latih sebelumnya
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

    generate_text(args)

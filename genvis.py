import torch
from tokenizers import ByteLevelBPETokenizer
from torchvision import transforms
from PIL import Image
import argparse
import os

# --- Impor model multimodal dari Canvas Anda ---
from modelvisual import MultimodalAgent

# ==========================================================
# BAGIAN 1: FUNGSI GENERASI VISUAL
# ==========================================================
@torch.no_grad()
def generate_visual_text(args):
    """Memuat model multimodal dan menghasilkan deskripsi untuk sebuah gambar."""
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

    # 3. Setup Model
    model_config = {
        'vocab_size': args.vocab_size, 'n_emb': args.n_emb, 'n_layer': args.n_layer,
        'n_head': args.n_head, 'n_kv_heads': args.n_kv_heads, 'max_seq_len': args.max_seq_len,
        'num_experts': args.num_experts, 'num_experts_per_tok': args.num_experts_per_tok,
        'sliding_window_size': args.sliding_window_size
    }
    model = MultimodalAgent(model_config).to(args.device)
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint_path}")
    print(f"✅ Memuat checkpoint dari {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    model.eval()

    # 4. Muat dan Proses Gambar Input
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(args.device)

    # 5. Tokenisasi Prompt Awal
    prompt_tokens = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long, device=args.device)

    # 6. ✨ LOGIKA GENERASI AUTO-REGRESIF ✨
    # Dapatkan embedding visual sekali saja
    visual_embeddings = model.vision_encoder(image_tensor)
    
    # Mulai dengan prompt teks
    text_tokens = prompt_tokens
    
    print("\n🚀 Menghasilkan teks...")
    for _ in range(args.max_new_tokens):
        # Dapatkan embedding teks untuk token saat ini
        text_embeddings = model.language_model.token_emb(text_tokens)
        
        # Gabungkan embedding visual dan teks
        combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # Proses melalui LLM untuk mendapatkan logits
        h = model.language_model.blocks(combined_embeddings)
        h = model.language_model.norm(h)
        logits = model.language_model.head(h)
        
        # Ambil logit untuk token terakhir saja
        next_token_logits = logits[:, -1, :]
        next_token_logits /= args.temperature
        
        # Terapkan top-k sampling
        if args.top_k is not None:
            v, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

        # Ambil sampel token berikutnya
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Tambahkan token baru ke urutan teks
        text_tokens = torch.cat([text_tokens, next_token], dim=1)
        
        # Berhenti jika bertemu token akhir (jika ada)
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break

    # 7. Decode dan Tampilkan Hasil
    generated_ids = text_tokens[0, prompt_tokens.shape[1]:].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    print("-" * 50)
    print(f"🖼️ Gambar: {args.image_path}")
    print(f"✍️ Prompt: {args.prompt}")
    print("-" * 50)
    print(f"💬 Jawaban AI:\n{generated_text}")
    print("-" * 50)


# ==========================================================
# BAGIAN 3: ARGUMENT PARSER
# ==========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text from an image using a Multimodal Agent.")
    parser.add_argument("--image_path", type=str, required=True, help="Path ke gambar input.")
    parser.add_argument("--prompt", type=str, default="Deskripsikan gambar ini:", help="Prompt teks awal.")
    parser.add_argument("--checkpoint_path", type=str, default="multimodal_checkpoints/best_multimodal_model.pt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    
    # Generation Config
    parser.add_argument("--max_new_tokens", type=int, default=75)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)

    # Model Config (harus sama dengan model yang dilatih)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--n_emb", type=int, default=512)
    # ... (sisa parameter model sama seperti di train_visual.py) ...
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--sliding_window_size", type=int, default=256)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generate_visual_text(args)

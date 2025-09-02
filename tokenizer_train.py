from tokenizers import ByteLevelBPETokenizer
import os
import glob

# ==========================================================
# BAGIAN 1: KONFIGURASI
# ==========================================================
class TokenizerConfig:
    # Path ke file-file teks mentah untuk training tokenizer
    # Secara otomatis akan menggunakan semua file wiki_part*.txt jika ada,
    # atau kembali ke wiki.txt jika tidak ada.
    FILES = sorted(glob.glob("data/wiki_part*.txt")) or ["data/wiki.txt"]

    # Direktori tempat menyimpan tokenizer yang sudah dilatih
    OUTPUT_DIR = "tokenizer"

    # Hyperparameters untuk tokenizer
    VOCAB_SIZE = 30000
    MIN_FREQUENCY = 2

    # Token spesial yang penting untuk model
    SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# ==========================================================
# BAGIAN 2: SCRIPT TRAINING TOKENIZER
# ==========================================================
def train_tokenizer():
    """Fungsi utama untuk melatih dan menyimpan tokenizer."""
    config = TokenizerConfig()
    
    # Pastikan direktori output ada
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Inisialisasi tokenizer baru
    tokenizer = ByteLevelBPETokenizer()
    
    print(f"📚 Memulai training tokenizer dari {len(config.FILES)} file...")
    print(f"   - Ukuran Vokabulari: {config.VOCAB_SIZE}")
    print(f"   - Frekuensi Minimum: {config.MIN_FREQUENCY}")

    # 2. Latih tokenizer
    # Proses ini mungkin memakan waktu beberapa menit tergantung ukuran dataset
    tokenizer.train(
        files=config.FILES, 
        vocab_size=config.VOCAB_SIZE, 
        min_frequency=config.MIN_FREQUENCY, 
        special_tokens=config.SPECIAL_TOKENS
    )
    
    # 3. Simpan tokenizer ke direktori
    tokenizer.save_model(config.OUTPUT_DIR)
    
    print("\n🎉 Training tokenizer selesai!")
    print(f"   Tokenizer (vocab.json & merges.txt) disimpan di folder '{config.OUTPUT_DIR}'.")
    print("   Sekarang Anda siap untuk menjalankan 'train.py'.")

# ==========================================================
# BAGIAN 3: ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    train_tokenizer()

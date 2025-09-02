import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Impor "Otak" Bahasa dari model yang sudah kamu buat ---
# Pastikan file ini ada di direktori yang sama
from model import TinyGPT_v6

# ==========================================================
# BAGIAN 1: "MATA" UNTUK AI (VISION ENCODER)
# Tugasnya adalah mengubah gambar menjadi embedding yang bisa dimengerti oleh 'otak' AI.
# ==========================================================
class SimpleVisionEncoder(nn.Module):
    """
    Encoder visual sederhana menggunakan Convolutional Neural Network (CNN).
    Ini akan memproses gambar dan mengubahnya menjadi serangkaian token visual.
    Input: Gambar (batch, channels, height, width), misal: (1, 3, 256, 256)
    Output: Urutan embedding visual (batch, num_tokens, embedding_dim)
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Lapisan konvolusi untuk mengekstrak fitur dari gambar
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0), # (B, 32, 63, 63)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # (B, 64, 30, 30)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # (B, 64, 28, 28)
            nn.ReLU(),
        )
        # Proyeksi akhir untuk mengubah fitur menjadi dimensi embedding yang sama dengan model bahasa
        self.final_proj = nn.Linear(64 * 28 * 28, embedding_dim)

    def forward(self, image):
        # 1. Ekstrak fitur dengan CNN
        x = self.conv_net(image)
        
        # 2. Ratakan output fitur menjadi satu dimensi panjang
        batch_size, _, _, _ = x.shape
        x_flat = x.view(batch_size, -1)
        
        # 3. Proyeksikan ke dimensi embedding yang benar
        visual_embedding = self.final_proj(x_flat)
        
        # 4. Tambahkan dimensi 'sequence' agar konsisten (hanya 1 token visual sederhana)
        return visual_embedding.unsqueeze(1)


# ==========================================================
# BAGIAN 2: MODEL MULTIMODAL (GABUNGAN MATA + OTAK)
# ==========================================================
class MultimodalAgent(nn.Module):
    """
    Menggabungkan Vision Encoder ("Mata") dengan Language Model ("Otak").
    """
    def __init__(self, model_config):
        super().__init__()
        # Inisialisasi "Mata"
        self.vision_encoder = SimpleVisionEncoder(embedding_dim=model_config['n_emb'])
        
        # Inisialisasi "Otak" Bahasa
        self.language_model = TinyGPT_v6(**model_config)

    def forward(self, text_tokens, image, targets=None):
        # 1. Dapatkan embedding dari gambar menggunakan "Mata"
        #    Output: (batch_size, num_visual_tokens, embedding_dim)
        visual_embeddings = self.vision_encoder(image)
        
        # 2. Dapatkan embedding dari teks menggunakan "Otak"
        #    Output: (batch_size, num_text_tokens, embedding_dim)
        text_embeddings = self.language_model.token_emb(text_tokens)

        # 3. ✨ KUNCI UTAMA: Gabungkan embedding visual dan teks menjadi satu urutan
        #    [IMG_TOKEN, a, photo, of, a, cat]
        combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # 4. Proses urutan gabungan melalui sisa dari model bahasa
        #    Kita perlu sedikit memodifikasi TinyGPT agar bisa menerima embedding langsung
        #    Untuk sekarang, kita panggil komponen internalnya secara manual
        h = self.language_model.blocks(combined_embeddings)
        h = self.language_model.norm(h)
        logits = self.language_model.head(h)
        
        # 5. Hitung loss jika ada target
        loss = None
        if targets is not None:
            # Kita hanya menghitung loss pada bagian teks, bukan pada token gambar
            # Ambil logits yang sesuai dengan posisi teks
            text_logits = logits[:, visual_embeddings.shape[1]:, :]
            loss = F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)), targets.view(-1))
            
        return logits, loss

# --- Contoh Penggunaan ---
if __name__ == '__main__':
    # Konfigurasi harus sama dengan model yang sudah dilatih
    config = {
        'vocab_size': 30000,
        'n_emb': 512,
        'n_layer': 8,
        'n_head': 8,
        'n_kv_heads': 4,
        'max_seq_len': 256,
        'num_experts': 8,
        'num_experts_per_tok': 2,
        'sliding_window_size': 256
    }

    # Buat model multimodal
    multimodal_model = MultimodalAgent(model_config=config)

    # Buat data dummy
    # Gambar acak berukuran 256x256 dengan 3 channel (RGB)
    dummy_image = torch.randn(1, 3, 256, 256) 
    
    # Prompt teks (misalnya, hasil tokenisasi dari "Apa yang ada di gambar ini?")
    dummy_text = torch.randint(0, config['vocab_size'], (1, 10))
    
    # Target untuk teks (digeser satu)
    dummy_targets = torch.randint(0, config['vocab_size'], (1, 10))


    # Jalankan model
    print("🤖 Menjalankan model multimodal dengan gambar dan teks...")
    logits, loss = multimodal_model(text_tokens=dummy_text, image=dummy_image, targets=dummy_targets)
    
    print(f"✅ Berhasil! Bentuk output Logits: {logits.shape}")
    if loss is not None:
        print(f"   Loss yang dihitung: {loss.item():.4f}")

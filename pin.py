import os
import csv
import argparse
from bing_image_downloader import downloader

# ==========================================================
# BAGIAN 1: FUNGSI UTAMA PENGUNDUH GAMBAR
# ==========================================================
def download_images_and_create_captions(args):
    """
    Mengunduh gambar berdasarkan query dan secara otomatis membuat
    file captions.csv yang kompatibel dengan train_visual.py.
    """
    
    # 1. Unduh gambar menggunakan library
    print(f"🚀 Memulai pengunduhan untuk query: '{args.query}'")
    print(f"   Jumlah gambar: {args.limit}")
    print(f"   Menyimpan ke: {args.output_dir}")
    
    downloader.download(
        args.query,
        limit=args.limit,
        output_dir=args.output_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True
    )

    print("\n✅ Pengunduhan gambar selesai.")

    # 2. Buat file captions.csv
    image_folder = os.path.join(args.output_dir, args.query)
    caption_file_path = os.path.join(args.output_dir, "captions.csv")
    
    print(f"✍️  Membuat file captions di: {caption_file_path}")

    # Kumpulkan semua nama file gambar yang valid
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    with open(caption_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_filename', 'caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        # Buat caption sederhana untuk setiap gambar
        for filename in image_files:
            # Path di CSV harus relatif terhadap folder utama (misal: 'kucing/Image_1.jpg')
            relative_path = os.path.join(args.query, filename)
            
            # Caption bisa dibuat lebih bervariasi, namun untuk awal kita gunakan query
            caption_text = f"sebuah foto {args.query}"
            
            writer.writerow({'image_filename': relative_path, 'caption': caption_text})

    print(f"🎉 Berhasil! {len(image_files)} gambar dan caption telah dibuat.")
    print(f"   Sekarang Anda bisa menggunakan folder '{args.output_dir}' dan file '{caption_file_path}' untuk 'train_visual.py'.")


# ==========================================================
# BAGIAN 2: ARGUMENT PARSER & ENTRYPOINT
# ==========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unduh dataset gambar dari internet dan buat file caption.")
    
    parser.add_argument("--query", type=str, required=True, help="Kata kunci pencarian gambar (misal: 'kucing lucu').")
    parser.add_argument("--limit", type=int, default=100, help="Jumlah gambar yang ingin diunduh.")
    parser.add_argument("--output_dir", type=str, default="visual_dataset", help="Folder utama untuk menyimpan gambar dan file caption.")

    args = parser.parse_args()
    
    # Menjalankan fungsi utama
    download_images_and_create_captions(args)

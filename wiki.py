import wikipediaapi
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

wiki = wikipediaapi.Wikipedia(
    user_agent='XenoGPT/0.3 (https://github.com/xeno)',
    language='id'
)

# Buat folder data kalau belum ada
os.makedirs("data", exist_ok=True)

categories = [
    "Sejarah Indonesia",
   "Tokoh Indonesia",
  "Kerajaan di Indonesia",
  "Teknologi",
  "Ilmu komputer",
  "Kecerdasan buatan",
  "Pemrograman",
  "Bahasa pemrograman",
  "Geografi Indonesia",
  "Flora Indonesia",
  "Fauna Indonesia",
  "Ekonomi Indonesia",
  "Politik Indonesia",
  "Budaya Indonesia",
  "Agama di Indonesia",
    "Fisika",
    "Kimia",
    "Biologi",
    "Matematika",
    "Astronomi",
    "Kedokteran",
    "Geologi",
    "Sastra Indonesia",
    "Musik",
    "Film",
    "Filsafat",
    "Mitologi",
    "Arsitektur",
    "Psikologi",
    "Sosiologi",
    "Pendidikan",
    "Hukum di Indonesia",
    "Antropologi",
    "Sejarah dunia",
    "Perang Dunia II",
    "Negara di dunia",
    "Penjelajahan",
    "Transportasi",
    "Pertanian",
    "Teknik sipil",
    "Robotika",
    "Olahraga",
    "Masakan Indonesia",
    "Permainan video",
    "Pariwisata di Indonesia",
    "Jaringan komputer",
    "Keamanan komputer",
    "Perangkat lunak",
    "Sistem operasi",
    "Basis data",
    "Internet",
    "Telekomunikasi",
    "Pengembangan web",
    "Sejarah kuno",
    "Abad Pertengahan",
    "Renaisans",
    "Revolusi Industri",
    "Peradaban Romawi Kuno",
    "Peradaban Yunani Kuno",
    "Peradaban Mesir Kuno",
    "Perang Dingin",
    "Ibu kota negara di dunia",
    "Sungai di Indonesia",
    "Gunung di Indonesia",
    "Samudra",
    "Gurun",
    "Asia Tenggara",
    "Benua",
    "Ekonomi makro",
    "Ekonomi mikro",
    "Perbankan",
    "Investasi",
    "Pemasaran",
    "Manajemen",
    "Perusahaan",
    "Mata uang",
    "Fotografi",
    "Teater",
    "Tarian",
    "Alat musik",
    "Masakan dunia",
    "Olimpiade",
    "Sepak bola",
    "Genetika",
    "Ekologi",
    "Botani",
    "Zoologi",
    "Mekanika kuantum",
    "Kimia organik",
    "Vulkanologi",
    "Kementerian di Indonesia",
    "Lembaga pemerintah Indonesia",
    "Hubungan internasional",
    "Konstitusi",
    "Sistem politik",
    "Ideologi politik",
    "Pemilihan umum",
    "Hukum internasional",
    "Penyakit",
    "Anatomi manusia",
    "Obat-obatan",
    "Gizi",
    "Kesehatan masyarakat",
    "Virus",
    "Bakteri",
    "Psikiatri",
    "Ilmu material",
    "Teknik elektro",
    "Bioteknologi",
    "Ilmu lingkungan",
    "Meteorologi",
    "Oseanografi",
    "Ilmuwan",
    "Linguistik",
    "Arkeologi",
    "Peninggalan sejarah di Indonesia",
    "Warisan Dunia UNESCO di Indonesia",
    "Genre musik",
    "Genre film",
    "Konsep filosofis",
    "Teori ilmiah",
    "Hukum fisika",
    "Paradoks",
    "Model matematika",
    "Satuan pengukuran",
    "Daftar tokoh menurut profesi",
    "Daftar peristiwa menurut tahun",
    "Daftar negara menurut",
    "Daftar bahasa menurut rumpun",
    "Daftar penemuan",
    "Daftar penghargaan",
]

def get_category_articles(category_name, max_depth=2, max_articles=200):
    """Ambil artikel dari kategori Wikipedia (rekursif sampai sub-kategori)"""
    cat = wiki.page("Kategori:" + category_name)
    if not cat.exists():
        return []

    pages = []

    def crawl(c, depth):
        print(f"[Crawl] {c.title} depth={depth} total={len(pages)}", flush=True)
        if depth > max_depth:
            return
        for title, member in c.categorymembers.items():
            if member.ns == 0:  # Artikel
                pages.append(title)
            elif member.ns == 14:  # Subkategori
                crawl(member, depth + 1)

    crawl(cat, 0)
    random.shuffle(pages)
    return pages[:max_articles]

def fetch_page(title):
    """Ambil konten artikel"""
    page = wiki.page(title)
    if page.exists() and len(page.text) > 500:  # skip artikel kosong/pendek
        return f"## {page.title}\n{page.text}\n\n"
    return None

if __name__ == "__main__":
    topics = []
    for cat in categories:
        topics.extend(get_category_articles(cat, max_depth=2, max_articles=200))

    topics = list(set(topics))  # hapus duplikat

    print(f"\n[INFO] Total topik terkumpul: {len(topics)}")

    # Parallel fetch biar lebih cepat
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_page, t): t for t in topics}
        for future in as_completed(futures):
            try:
                data = future.result()
                if data:
                    results.append(data)
                    print(f"[OK] {futures[future]} berhasil ditambahkan", flush=True)
                else:
                    print(f"[SKIP] {futures[future]} kosong/pendek", flush=True)
            except Exception as e:
                print(f"[ERROR] {futures[future]} gagal: {e}", flush=True)

    # Simpan hasil
    with open("data/wiki.txt", "w", encoding="utf-8") as f:
        f.writelines(results)

    print(f"\n✅ Dataset final terkumpul: {len(results)} artikel")

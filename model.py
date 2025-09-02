import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import re
from typing import List, Dict, Any

# --- BAGIAN 1: ARSITEKTUR MODEL INTI (TinyGPT_v6) ---
# 'Otak' pemrosesan bahasa dari AI. Kode ini tetap sama.
# ... (Kode TinyGPT_v6 lengkap dari file sebelumnya disisipkan di sini)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, device=inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    def forward(self, x):
        seq_len = x.shape[2]
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def apply_rotary_pos_emb(q, k, cos, sin):
    q_half1, q_half2 = q.chunk(2, dim=-1)
    k_half1, k_half2 = k.chunk(2, dim=-1)
    q_rotated = torch.cat((-q_half2, q_half1), dim=-1)
    k_rotated = torch.cat((-k_half2, k_half1), dim=-1)
    return (q * cos + q_rotated * sin), (k * cos + k_rotated * sin)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1: return x
    return x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim).reshape(bs, n_kv_heads * n_rep, slen, head_dim)

class SlidingWindowAttention(nn.Module):
    def __init__(self, n_emb, n_head, n_kv_heads, max_seq_len, dropout, sliding_window_size=None):
        super().__init__()
        self.n_head, self.n_kv_heads = n_head, n_kv_heads
        self.n_rep = n_head // n_kv_heads
        self.head_dim = n_emb // n_head
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(n_emb, d, bias=False) for d in [n_emb, n_kv_heads * self.head_dim, n_kv_heads * self.head_dim, n_emb]]
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        self.resid_dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size
        self.mask = None
        if sliding_window_size is not None:
            mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            for i in range(max_seq_len): mask[:, :, i, :max(0, i - sliding_window_size + 1)] = float("-inf")
            self.register_buffer("mask", mask)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        attn_mask = self.mask[:, :, :T, :T] if self.sliding_window_size is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.sliding_window_size is None)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.w1, self.w2, self.w3 = [nn.Linear(d_in, d_out, bias=False) for d_in, d_out in [(dim, hidden_dim), (hidden_dim, dim), (dim, hidden_dim)]]
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MixtureOfExperts(nn.Module):
    def __init__(self, n_emb, num_experts, num_experts_per_tok, ff_mult, dropout):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_emb, n_emb * ff_mult, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(n_emb, num_experts, bias=False)
        self.num_experts_per_tok = num_experts_per_tok
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        router_logits = self.gate(x_flat)
        weights, selected_experts = torch.topk(F.softmax(router_logits, dim=1, dtype=torch.float), self.num_experts_per_tok, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        final_output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=len(self.experts)).permute(2, 0, 1)
        for i, expert in enumerate(self.experts):
            token_indices, _ = torch.where(expert_mask[i])
            if token_indices.numel() > 0:
                final_output.index_add_(0, token_indices, (expert(x_flat[token_indices]) * weights[expert_mask[i]].sum(1, keepdim=True)).to(x.dtype))
        return final_output.view(B, T, C)

class Block(nn.Module):
    def __init__(self, n_emb, n_head, n_kv_heads, max_seq_len, ff_mult, dropout, num_experts, num_experts_per_tok, sliding_window_size):
        super().__init__()
        self.attention = SlidingWindowAttention(n_emb, n_head, n_kv_heads, max_seq_len, dropout, sliding_window_size)
        self.feed_forward = MixtureOfExperts(n_emb, num_experts, num_experts_per_tok, ff_mult, dropout)
        self.attention_norm, self.ffn_norm = RMSNorm(n_emb), RMSNorm(n_emb)
    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))

class TinyGPT_v6(nn.Module):
    def __init__(self, vocab_size, n_emb=384, n_layer=8, n_head=8, n_kv_heads=4, max_seq_len=128, dropout=0.1, ff_mult=4, num_experts=8, num_experts_per_tok=2, sliding_window_size=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head, n_kv_heads, max_seq_len, ff_mult, dropout, num_experts, num_experts_per_tok, sliding_window_size) for _ in range(n_layer)])
        self.norm = RMSNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        self.token_emb.weight = self.head.weight
        self.max_seq_len = max_seq_len
    def forward(self, x, targets=None):
        if x.size(1) > self.max_seq_len: x = x[:, -self.max_seq_len:]
        h = self.token_emb(x)
        h = self.blocks(h)
        logits = self.head(self.norm(h))
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if stop_token is not None and idx_next.item() == stop_token: break
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
# --- END OF MODEL ARCHITECTURE ---


# --- BAGIAN 2: MODUL EKSTERNAL (MEMORI & ALAT) ---
class KnowledgeBase:
    """ ✨ UPGRADE v8: Basis Pengetahuan dengan Caching & Retrieval Berbobot. """
    def __init__(self, embedding_model):
        self.memories: List[Dict] = []
        self.embedding_model = embedding_model
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self.type_weights = {"USER_PROFILE": 1.5, "SUMMARY": 1.2, "GENERAL_FACT": 1.0}

    @torch.no_grad()
    def _get_embedding(self, text):
        if text in self.embedding_cache: return self.embedding_cache[text]
        tokens = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
        emb = self.embedding_model.token_emb(tokens).mean(dim=1)
        self.embedding_cache[text] = emb
        return emb

    def add_memory(self, text: str, type: str):
        print(f"[Memori] ✨ Menambahkan ingatan [{type}]: '{text}'")
        self.memories.append({'text': text, 'type': type, 'timestamp': time.time()})

    def retrieve_relevant_memories(self, query_text, top_k=3):
        if not self.memories: return []
        query_vector = self._get_embedding(query_text)
        
        scores = []
        for i, mem in enumerate(self.memories):
            mem_vector = self._get_embedding(mem['text'])
            similarity = F.cosine_similarity(query_vector, mem_vector).item()
            
            # Recency Weighting (semakin baru, semakin tinggi skor)
            recency = math.exp(-0.01 * (time.time() - mem['timestamp']) / 3600) # Decay per jam
            # Type Weighting
            type_weight = self.type_weights.get(mem['type'], 1.0)
            
            final_score = similarity * recency * type_weight
            scores.append((final_score, i))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for score, idx in scores[:top_k]]
        return [self.memories[i] for i in top_indices]

class ToolBox:
    """ ✨ FITUR BARU v8: Kotak alat untuk memperluas kemampuan AI. """
    def _web_search(self, query: str) -> str:
        print(f"[Alat] 🔍 Melakukan pencarian web untuk: '{query}'")
        # Simulasi: Di dunia nyata, ini akan memanggil API Google/Bing
        if "cuaca" in query: return f"Hasil pencarian: Cuaca di Jakarta hari ini cerah berawan."
        if "presiden" in query: return f"Hasil pencarian: Presiden Indonesia saat ini adalah Joko Widodo."
        return f"Hasil pencarian: Tidak ditemukan hasil relevan untuk '{query}'."

    def _calculator(self, expression: str) -> str:
        print(f"[Alat] 🧮 Menghitung: '{expression}'")
        try:
            # Peringatan keamanan: eval() berbahaya. Ini hanya untuk demonstrasi.
            return f"Hasil kalkulasi: {eval(expression)}"
        except Exception as e:
            return f"Error kalkulasi: {e}"

    def use_tool(self, tool_call: str) -> str:
        match = re.match(r"(\w+)\((.+)\)", tool_call)
        if not match: return "Format alat tidak valid."
        tool_name, tool_arg = match.groups()
        tool_arg = tool_arg.strip("'\"")

        if tool_name == "web_search": return self._web_search(tool_arg)
        if tool_name == "calculator": return self._calculator(tool_arg)
        return f"Alat '{tool_name}' tidak ditemukan."

# --- BAGIAN 3: AGEN OTONOM DENGAN REASONING LOOP ---
class AutonomousAgent:
    """ ✨ UPGRADE v8: Menggunakan 'Reason-Act Loop' untuk interaksi cerdas. """
    def __init__(self, model: TinyGPT_v6):
        self.model = model
        self.knowledge_base = KnowledgeBase(embedding_model=model)
        self.toolbox = ToolBox()

    def _generate_text(self, prompt: str, max_tokens: int) -> str:
        """ Helper function untuk memanggil model. """
        tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long)
        generated = self.model.generate(tokens, max_tokens)
        return "".join([chr(i) for i in generated[0][len(tokens[0]):]])

    def _reasoning_step(self, user_query: str, chat_history: List[str]) -> str:
        """ Menghasilkan rencana tindakan (thought process). """
        prompt = f"History: {' '.join(chat_history)}\nQuery: {user_query}\nThink: What is the user's intent? Do I need memory or a tool? Formulate a plan. Plan:"
        plan = self._generate_text(prompt, 80)
        print(f"[Agen] 🤔 Rencana: {plan}")
        return plan

    def _execute_plan(self, plan: str) -> Dict[str, Any]:
        """ Mengeksekusi rencana, mengambil data dari memori atau alat. """
        results = {"memory": [], "tool": []}
        # Cari panggilan alat
        tool_calls = re.findall(r"TOOL\[(\w+\(.+\))\]", plan)
        for call in tool_calls:
            results["tool"].append(self.toolbox.use_tool(call))
        
        # Cari query memori
        memory_queries = re.findall(r"MEMORY\[(.+?)\]", plan)
        for query in memory_queries:
            results["memory"].extend([mem['text'] for mem in self.knowledge_base.retrieve_relevant_memories(query)])
        
        return results

    def respond(self, user_query: str, chat_history: List[str] = []):
        print(f"\n[User] 🗣️  '{user_query}'")
        
        # 1. REASON: AI berpikir dan membuat rencana
        plan = self._reasoning_step(user_query, chat_history)

        # 2. ACT: AI mengeksekusi rencana
        execution_results = self._execute_plan(plan)

        # 3. RESPOND: AI mensintesis jawaban akhir
        final_prompt = (
            f"Anda adalah asisten AI. Gunakan informasi berikut untuk menjawab.\n"
            f"Informasi dari Memori: {' '.join(execution_results['memory']) or 'Tidak ada'}\n"
            f"Informasi dari Alat: {' '.join(execution_results['tool']) or 'Tidak ada'}\n"
            f"Pertanyaan Pengguna: {user_query}\nJawaban:"
        )
        response = self._generate_text(final_prompt, 100)
        print(f"[AI] 💬 '{response}'")
        return response

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import os
import sys
import io
import json
import base64
import uuid
import time
import asyncio
import httpx
import csv
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

load_dotenv()

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================
# 目录配置
# ============================================================
TEMP_UPLOAD_DIR = Path(os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")).resolve()
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_FILE_TTL = int(os.getenv("TEMP_FILE_TTL", 3600))
TEMP_MAX_FILE_SIZE = 20 * 1024 * 1024
TEMP_ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp", "image/gif"}
TEMP_ALLOWED_VIDEO = {"video/mp4", "video/webm", "video/quicktime"}
TEMP_META_FILE = TEMP_UPLOAD_DIR / "_meta.json"

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
for _sub in ("main", "sub", "video", "detail", "proxy"):
    (UPLOAD_DIR / _sub).mkdir(parents=True, exist_ok=True)

KB_DIR = Path(os.getenv("KB_DIR", "knowledge_base")).resolve()
KB_DIR.mkdir(parents=True, exist_ok=True)
(KB_DIR / "files").mkdir(parents=True, exist_ok=True)
KB_META_FILE = KB_DIR / "_meta.json"
KB_ALLOWED_EXT = {".txt", ".md", ".markdown", ".json", ".csv", ".pdf", ".docx"}
KB_MAX_FILE_SIZE = 20 * 1024 * 1024

ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets_lib")).resolve()
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
(ASSETS_DIR / "images").mkdir(exist_ok=True)
(ASSETS_DIR / "videos").mkdir(exist_ok=True)
ASSETS_META_FILE = ASSETS_DIR / "_meta.json"
ASSET_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
ASSET_VIDEO_EXT = {".mp4", ".webm", ".mov", ".quicktime"}
ASSET_IMAGE_MAX = 10 * 1024 * 1024
ASSET_VIDEO_MAX = 100 * 1024 * 1024

SESSIONS_FILE = Path(os.getenv("SESSIONS_FILE", "sessions.json")).resolve()
SESSION_MAX_TURNS = 12
SESSION_TTL = 3600 * 6
SESSION_CLEANUP_INTERVAL = 1800

# ★ 新增文件
COUPON_FILE = Path(os.getenv("COUPON_FILE", "coupons.json")).resolve()
CARD_CLICK_FILE = Path(os.getenv("CARD_CLICK_FILE", "card_clicks.json")).resolve()


def get_public_base_url(request: Request) -> str:
    env_url = os.getenv("PUBLIC_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    forwarded_host = request.headers.get("x-forwarded-host", "")
    if forwarded_host:
        proto = forwarded_proto or "https"
        return f"{proto}://{forwarded_host}".rstrip("/")
    base = str(request.base_url).rstrip("/")
    if base.startswith("http://") and "localhost" not in base and "127.0.0.1" not in base:
        base = "https://" + base[len("http://"):]
    return base


# ============================================================
# 临时图床
# ============================================================
def _load_temp_meta():
    if TEMP_META_FILE.exists():
        try:
            with open(TEMP_META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_temp_meta(meta):
    try:
        with open(TEMP_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


temp_meta = _load_temp_meta()


def _cleanup_expired():
    now = time.time()
    expired = []
    for file_id, info in temp_meta.items():
        if now - info.get("created_at", 0) > TEMP_FILE_TTL:
            expired.append(file_id)
    for file_id in expired:
        info = temp_meta.pop(file_id, None)
        if info:
            try:
                Path(info["path"]).unlink(missing_ok=True)
            except Exception:
                pass
    if expired:
        _save_temp_meta(temp_meta)


async def _periodic_cleanup():
    while True:
        try:
            _cleanup_expired()
        except Exception as e:
            print("[cleanup] error:", e)
        await asyncio.sleep(600)


# ============================================================
# 会话
# ============================================================
def _load_sessions():
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_sessions(sessions):
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[_save_sessions] 失败: {e}")


sessions_store = _load_sessions()


def _cleanup_sessions():
    now = time.time()
    expired = [sid for sid, s in sessions_store.items()
               if now - s.get("updated_at", 0) > SESSION_TTL]
    for sid in expired:
        sessions_store.pop(sid, None)
    if expired:
        _save_sessions(sessions_store)


async def _periodic_session_cleanup():
    while True:
        try:
            _cleanup_sessions()
        except Exception as e:
            print("[session cleanup] error:", e)
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL)


def get_or_create_session(session_id: str) -> dict:
    if not session_id:
        session_id = uuid.uuid4().hex[:16]
    s = sessions_store.get(session_id)
    if not s:
        s = {
            "id": session_id,
            "turns": [],
            "last_generated": None,
            "last_product": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        sessions_store[session_id] = s
    s["updated_at"] = time.time()
    return s


def append_turn(session: dict, role: str, text: str, **extra):
    turn = {
        "role": role,
        "text": text or "",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    turn.update(extra)
    session["turns"].append(turn)
    if len(session["turns"]) > SESSION_MAX_TURNS * 2:
        session["turns"] = session["turns"][-SESSION_MAX_TURNS * 2:]
    session["updated_at"] = time.time()
    _save_sessions(sessions_store)


# ============================================================
# 知识库
# ============================================================
DEFAULT_KB_CATEGORIES = [
    {"id": "cat_faq", "name": "FAQ", "icon": "❓", "parent_id": None,
     "children": [
         {"id": "cat_faq_product", "name": "商品咨询", "icon": "📦", "parent_id": "cat_faq", "children": []},
         {"id": "cat_faq_logistics", "name": "物流咨询", "icon": "🚚", "parent_id": "cat_faq", "children": []},
         {"id": "cat_faq_after_sales", "name": "售后咨询", "icon": "🛠️", "parent_id": "cat_faq", "children": []},
     ]},
    {"id": "cat_product_doc", "name": "产品文档", "icon": "📖", "parent_id": None,
     "children": [
         {"id": "cat_product_intro", "name": "产品介绍", "icon": "🌟", "parent_id": "cat_product_doc", "children": []},
         {"id": "cat_product_spec", "name": "规格参数", "icon": "📐", "parent_id": "cat_product_doc", "children": []},
         {"id": "cat_product_usage", "name": "使用教程", "icon": "🎓", "parent_id": "cat_product_doc", "children": []},
     ]},
    {"id": "cat_after_sales", "name": "售后政策", "icon": "🛡️", "parent_id": None,
     "children": [
         {"id": "cat_after_return", "name": "退换货", "icon": "↩️", "parent_id": "cat_after_sales", "children": []},
         {"id": "cat_after_refund", "name": "退款说明", "icon": "💰", "parent_id": "cat_after_sales", "children": []},
         {"id": "cat_after_quality", "name": "质量问题", "icon": "⚠️", "parent_id": "cat_after_sales", "children": []},
     ]},
    {"id": "cat_logistics", "name": "物流说明", "icon": "🚛", "parent_id": None,
     "children": [
         {"id": "cat_logi_delivery", "name": "配送范围", "icon": "🗺️", "parent_id": "cat_logistics", "children": []},
         {"id": "cat_logi_time", "name": "时效说明", "icon": "⏰", "parent_id": "cat_logistics", "children": []},
     ]},
    {"id": "cat_other", "name": "其他", "icon": "📝", "parent_id": None, "children": []},
]


def _load_kb_meta():
    if KB_META_FILE.exists():
        try:
            with open(KB_META_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "items" not in data: data["items"] = []
                if "next_id" not in data: data["next_id"] = 1
                if "categories" not in data: data["categories"] = DEFAULT_KB_CATEGORIES
                if "stats" not in data: data["stats"] = {}
                return data
        except Exception:
            pass
    return {"items": [], "next_id": 1, "categories": DEFAULT_KB_CATEGORIES, "stats": {}}


def _save_kb_meta(meta):
    try:
        with open(KB_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[_save_kb_meta] 失败: {e}")


def _extract_text_from_file(file_path: Path, ext: str) -> str:
    try:
        if ext in (".txt", ".md", ".markdown"):
            return file_path.read_text(encoding="utf-8", errors="replace")

        if ext == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                lines = []
                for item in data:
                    if isinstance(item, dict):
                        q = item.get("question") or item.get("q") or ""
                        a = item.get("answer") or item.get("a") or ""
                        if q or a:
                            lines.append(f"Q: {q}\nA: {a}")
                        else:
                            lines.append(json.dumps(item, ensure_ascii=False))
                    else:
                        lines.append(str(item))
                return "\n\n".join(lines)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if ext == ".csv":
            lines = []
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append(" | ".join(row))
            return "\n".join(lines)

        if ext == ".pdf":
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(str(file_path)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                return "\n\n".join(text_parts)
            except ImportError:
                print("[_extract_text] pdfplumber 未安装，跳过 PDF 解析")
                return ""
            except Exception as e:
                print(f"[_extract_text] PDF 解析失败: {e}")
                return ""

        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(file_path))
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                print("[_extract_text] python-docx 未安装，跳过 DOCX 解析")
                return ""
            except Exception as e:
                print(f"[_extract_text] DOCX 解析失败: {e}")
                return ""

    except Exception as e:
        print(f"[_extract_text] 失败: {e}")
        return ""
    return ""


# ============================================================
# 向量检索
# ============================================================
_kb_vector_cache = {"matrix": None, "vectorizer": None, "ids": []}


def _rebuild_kb_vectors():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
    except ImportError:
        print("[_rebuild_kb_vectors] sklearn 未安装，退化为关键词匹配")
        _kb_vector_cache["matrix"] = None
        _kb_vector_cache["vectorizer"] = None
        _kb_vector_cache["ids"] = []
        return

    meta = _load_kb_meta()
    items = [x for x in meta.get("items", []) if x.get("enabled", True)]
    if not items:
        _kb_vector_cache["matrix"] = None
        _kb_vector_cache["vectorizer"] = None
        _kb_vector_cache["ids"] = []
        return

    corpus = []
    ids = []
    for item in items:
        text = (item.get("title", "") + " " + item.get("content", "")[:2000]).strip()
        if not text:
            continue
        corpus.append(text)
        ids.append(item["id"])

    if not corpus:
        _kb_vector_cache["matrix"] = None
        _kb_vector_cache["vectorizer"] = None
        _kb_vector_cache["ids"] = []
        return

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1, max_features=5000)
        matrix = vectorizer.fit_transform(corpus)
        _kb_vector_cache["matrix"] = matrix
        _kb_vector_cache["vectorizer"] = vectorizer
        _kb_vector_cache["ids"] = ids
        print(f"[_rebuild_kb_vectors] 已重建索引，共 {len(ids)} 条")
    except Exception as e:
        print(f"[_rebuild_kb_vectors] 失败: {e}")
        _kb_vector_cache["matrix"] = None
        _kb_vector_cache["vectorizer"] = None
        _kb_vector_cache["ids"] = []


def _vector_search(query: str, top_k: int = 5):
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None

    matrix = _kb_vector_cache.get("matrix")
    vectorizer = _kb_vector_cache.get("vectorizer")
    ids = _kb_vector_cache.get("ids", [])

    if matrix is None or vectorizer is None or not ids:
        return None

    try:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix)[0]
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])[:top_k]
        meta = _load_kb_meta()
        items_map = {x["id"]: x for x in meta.get("items", [])}
        results = []
        for idx, score in ranked:
            if score <= 0.01:
                continue
            item_id = ids[idx]
            item = items_map.get(item_id)
            if item:
                results.append({"item": item, "score": float(score)})
        return results
    except Exception as e:
        print(f"[_vector_search] 失败: {e}")
        return None


def _tokenize_keyword(text: str):
    if not text:
        return set()
    text = text.lower()
    tokens = set()
    en_words = re.findall(r"[a-z0-9]+", text)
    tokens.update(en_words)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chinese_chars) - 1):
        tokens.add(chinese_chars[i] + chinese_chars[i + 1])
    tokens.update(chinese_chars)
    return tokens


def _keyword_search(query: str, top_k: int = 5):
    meta = _load_kb_meta()
    items = [x for x in meta.get("items", []) if x.get("enabled", True)]
    if not items:
        return []
    q_tokens = _tokenize_keyword(query)
    if not q_tokens:
        return []

    scored = []
    for item in items:
        content = item.get("content", "")
        title = item.get("title", "")
        tags = item.get("tags", [])
        haystack = (title + " " + content + " " + " ".join(tags)).lower()
        h_tokens = _tokenize_keyword(haystack)
        if not h_tokens:
            continue
        common = q_tokens & h_tokens
        if not common:
            continue
        score = len(common) / max(len(q_tokens), 1)
        if any(t in title.lower() for t in q_tokens if len(t) >= 2):
            score += 0.5
        for tag in tags:
            if tag.lower() in query.lower():
                score += 0.3
        if item.get("pinned"):
            score += 0.2
        scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    return [{"item": item, "score": score} for score, item in scored[:top_k]]


def search_knowledge_hybrid(query: str, top_k: int = 3):
    vector_results = _vector_search(query, top_k=top_k * 2)
    keyword_results = _keyword_search(query, top_k=top_k * 2)

    merged = {}
    if vector_results:
        for r in vector_results:
            item_id = r["item"]["id"]
            merged[item_id] = {"item": r["item"], "score": r["score"] * 1.2, "match_type": "vector"}
    if keyword_results:
        for r in keyword_results:
            item_id = r["item"]["id"]
            if item_id in merged:
                merged[item_id]["score"] += r["score"] * 0.8
                merged[item_id]["match_type"] = "hybrid"
            else:
                merged[item_id] = {"item": r["item"], "score": r["score"], "match_type": "keyword"}

    results = sorted(merged.values(), key=lambda x: -x["score"])[:top_k]
    return results


def search_knowledge(query: str, top_k: int = 3):
    results = search_knowledge_hybrid(query, top_k)
    return [r["item"] for r in results]


def record_kb_hit(item_id: str):
    meta = _load_kb_meta()
    if "stats" not in meta:
        meta["stats"] = {}
    stats = meta["stats"].get(item_id, {"hit_count": 0, "last_hit": "", "satisfied": 0, "unsatisfied": 0})
    stats["hit_count"] = stats.get("hit_count", 0) + 1
    stats["last_hit"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["stats"][item_id] = stats
    _save_kb_meta(meta)


def record_kb_feedback(item_id: str, satisfied: bool):
    meta = _load_kb_meta()
    if "stats" not in meta:
        meta["stats"] = {}
    stats = meta["stats"].get(item_id, {"hit_count": 0, "last_hit": "", "satisfied": 0, "unsatisfied": 0})
    if satisfied:
        stats["satisfied"] = stats.get("satisfied", 0) + 1
    else:
        stats["unsatisfied"] = stats.get("unsatisfied", 0) + 1
    meta["stats"][item_id] = stats
    _save_kb_meta(meta)


def _find_category_name(categories, cat_id):
    for node in categories:
        if node["id"] == cat_id:
            return node["name"]
        if node.get("children"):
            name = _find_category_name(node["children"], cat_id)
            if name:
                return name
    return cat_id or "未分类"


# ============================================================
# 资产库
# ============================================================
def _load_assets_meta():
    if ASSETS_META_FILE.exists():
        try:
            with open(ASSETS_META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"items": [], "next_id": 1}
    return {"items": [], "next_id": 1}


def _save_assets_meta(meta):
    try:
        with open(ASSETS_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[_save_assets_meta] 失败: {e}")


def search_assets(query: str, asset_type: str = "all", top_k: int = 6):
    meta = _load_assets_meta()
    items = meta.get("items", [])
    if not items:
        return []
    results = []
    q = (query or "").lower().strip()
    q_tokens = _tokenize_keyword(q)
    for item in items:
        if asset_type != "all" and item.get("type") != asset_type:
            continue
        name = item.get("name", "").lower()
        tags = [t.lower() for t in item.get("tags", [])]
        product = (item.get("product_name") or "").lower()
        category = (item.get("category") or "").lower()
        if not q:
            results.append((0, item))
            continue
        haystack = name + " " + " ".join(tags) + " " + product + " " + category
        h_tokens = _tokenize_keyword(haystack)
        common = q_tokens & h_tokens
        score = len(common)
        if q in name: score += 3
        if any(q in t for t in tags): score += 2
        if q in product: score += 2
        if q in category: score += 1
        if score > 0:
            results.append((score, item))
    results.sort(key=lambda x: -x[0])
    return [item for _, item in results[:top_k]]


# ============================================================
# ★ 优惠券
# ============================================================
def load_coupons():
    if COUPON_FILE.exists():
        try:
            with open(COUPON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        {"id": "cp1", "code": "NEW20", "name": "新客专享券", "discount": 20, "threshold": 100,
         "desc": "新人首单立减", "expire_at": "2026-12-31"},
        {"id": "cp2", "code": "VIP50", "name": "VIP 会员券", "discount": 50, "threshold": 300,
         "desc": "会员专享 满300减50", "expire_at": "2026-12-31"},
        {"id": "cp3", "code": "FREESHIP", "name": "包邮券", "discount": 8, "threshold": 0,
         "desc": "全店包邮", "expire_at": "2026-12-31"},
    ]


def save_coupons(coupons):
    with open(COUPON_FILE, "w", encoding="utf-8") as f:
        json.dump(coupons, f, ensure_ascii=False, indent=2)


# ============================================================
# ★ 卡片点击追踪
# ============================================================
def load_card_clicks():
    if CARD_CLICK_FILE.exists():
        try:
            with open(CARD_CLICK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_shown": 0, "total_clicked": 0, "by_product": {}}


def save_card_clicks(data):
    with open(CARD_CLICK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    task1 = asyncio.create_task(_periodic_cleanup())
    task2 = asyncio.create_task(_periodic_session_cleanup())
    try:
        _rebuild_kb_vectors()
    except Exception as e:
        print(f"[lifespan] 向量索引重建失败: {e}")
    yield
    task1.cancel()
    task2.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "").strip() or "placeholder-no-key",
    base_url="https://api.deepseek.com"
)

agnes_client = OpenAI(
    api_key=os.getenv("AGNES_API_KEY", "").strip() or "placeholder-no-key",
    base_url="https://apihub.agnes-ai.com/v1"
)

DATA_FILE = "products.json"
LOG_FILE = "logs.json"
TAG_FILE = "tags.json"
LOG_MAX = 1000


# ============================================================
# 调试
# ============================================================
@app.get("/api/debug/env")
async def debug_env():
    agnes_key = os.getenv("AGNES_API_KEY", "").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    public_url = os.getenv("PUBLIC_BASE_URL", "").strip()

    def mask(k):
        if not k: return "(未设置)"
        if len(k) <= 12: return f"{k[:4]}...（共 {len(k)} 位）"
        return f"{k[:8]}...{k[-4:]}（共 {len(k)} 位）"

    return {
        "status": "ok",
        "AGNES_API_KEY": mask(agnes_key),
        "AGNES_API_KEY_length": len(agnes_key),
        "DEEPSEEK_API_KEY": mask(deepseek_key),
        "DEEPSEEK_API_KEY_length": len(deepseek_key),
        "PUBLIC_BASE_URL": public_url or "(未设置)"
    }


@app.get("/api/debug/agnes")
async def debug_agnes():
    api_key = os.getenv("AGNES_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "message": "AGNES_API_KEY 未设置", "key_length": 0}
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.post(
                "https://apihub.agnes-ai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "agnes-image-2.5-flash", "prompt": "一杯绿茶，白色背景，简单", "size": "1K", "ratio": "1:1", "extra_body": {"response_format": "url"}}
            )
            if r.status_code != 200:
                return {"success": False, "message": f"Agnes 返回 {r.status_code}", "error": r.text[:500]}
            data = r.json()
            return {"success": True, "message": "Agnes Key 有效！", "agnes_image_url": data["data"][0].get("url") if data.get("data") else None}
    except Exception as e:
        return {"success": False, "message": f"调用异常：{str(e)}"}


# ============================================================
# 类目树
# ============================================================
CATEGORY_TREE = [
    {"id": "cat_food", "name": "食品", "icon": "🍜", "children": [
        {"id": "cat_food_nuts", "name": "坚果", "children": [
            {"id": "cat_food_nuts_pistachio", "name": "开心果"}, {"id": "cat_food_nuts_walnut", "name": "核桃"}, {"id": "cat_food_nuts_almond", "name": "巴旦木"},
        ]},
        {"id": "cat_food_tea", "name": "茶饮", "children": [
            {"id": "cat_food_tea_green", "name": "绿茶"}, {"id": "cat_food_tea_black", "name": "红茶"}, {"id": "cat_food_tea_oolong", "name": "乌龙茶"},
        ]},
        {"id": "cat_food_snack", "name": "零食", "children": [
            {"id": "cat_food_snack_candy", "name": "糖果"}, {"id": "cat_food_snack_dried", "name": "果干"}, {"id": "cat_food_snack_meat", "name": "肉干"},
        ]},
        {"id": "cat_food_health", "name": "滋补", "children": [
            {"id": "cat_food_health_tea", "name": "养生茶"}, {"id": "cat_food_health_soup", "name": "汤料"},
        ]},
    ]},
    {"id": "cat_handicraft", "name": "手工艺", "icon": "🎨", "children": [
        {"id": "cat_handicraft_weave", "name": "编织", "children": [
            {"id": "cat_handicraft_weave_bamboo", "name": "竹编"}, {"id": "cat_handicraft_weave_rattan", "name": "藤编"}, {"id": "cat_handicraft_weave_cloth", "name": "布艺"},
        ]},
        {"id": "cat_handicraft_ceramic", "name": "陶瓷", "children": [
            {"id": "cat_handicraft_ceramic_cup", "name": "陶瓷杯"}, {"id": "cat_handicraft_ceramic_plate", "name": "陶瓷盘"}, {"id": "cat_handicraft_ceramic_teapot", "name": "陶瓷壶"},
        ]},
        {"id": "cat_handicraft_wood", "name": "木艺", "children": [
            {"id": "cat_handicraft_wood_box", "name": "木盒"}, {"id": "cat_handicraft_wood_toy", "name": "木制玩具"},
        ]},
    ]},
    {"id": "cat_tea", "name": "茶叶", "icon": "🍃", "children": [
        {"id": "cat_tea_green", "name": "绿茶", "children": [
            {"id": "cat_tea_green_longjing", "name": "龙井"}, {"id": "cat_tea_green_biluochun", "name": "碧螺春"},
        ]},
        {"id": "cat_tea_black", "name": "红茶", "children": [
            {"id": "cat_tea_black_junshan", "name": "君山银针"}, {"id": "cat_tea_black_keemun", "name": "祁门红茶"},
        ]},
        {"id": "cat_tea_oolong", "name": "乌龙茶", "children": [
            {"id": "cat_tea_oolong_tieguanyin", "name": "铁观音"}, {"id": "cat_tea_oolong_dahongpao", "name": "大红袍"},
        ]},
    ]},
    {"id": "cat_cultural", "name": "文创", "icon": "📚", "children": [
        {"id": "cat_cultural_paper", "name": "纸艺", "children": [
            {"id": "cat_cultural_paper_fan", "name": "折扇"}, {"id": "cat_cultural_paper_card", "name": "明信片"}, {"id": "cat_cultural_paper_bookmark", "name": "书签"},
        ]},
        {"id": "cat_cultural_fabric", "name": "布艺", "children": [
            {"id": "cat_cultural_fabric_scarf", "name": "丝巾"}, {"id": "cat_cultural_fabric_bag", "name": "布包"},
        ]},
        {"id": "cat_cultural_soap", "name": "香道", "children": [
            {"id": "cat_cultural_soap_handmade", "name": "手工皂"}, {"id": "cat_cultural_soap_scent", "name": "香薰"},
        ]},
    ]},
]

CATEGORY_MAP = {}


def build_category_map(tree, parent_path=None):
    if parent_path is None: parent_path = []
    for node in tree:
        node_id = node["id"]
        path = parent_path + [{"id": node_id, "name": node["name"]}]
        CATEGORY_MAP[node_id] = {"name": node["name"], "path": path, "level": len(path), "parent_id": parent_path[-1]["id"] if parent_path else None}
        if "children" in node:
            build_category_map(node["children"], path)


build_category_map(CATEGORY_TREE)

# ============================================================
# 默认商品
# ============================================================
DEFAULT_PRODUCTS = [
    {"id": 1, "name": "云山茶叶礼盒", "price": 128, "stock": 234, "category": "茶叶", "platform": "抖音", "status": "在售", "sales": 1247, "icon": "fa-leaf", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "500g×2", "sku": "YS-2026-001", "rating": "4.9星", "subcat": "茶饮", "third": "礼盒装", "tags": ["春茶", "送礼首选", "高端款"], "category_path": ["cat_tea", "cat_tea_green", "cat_tea_green_longjing"]},
    {"id": 2, "name": "手工竹编包", "price": 89, "stock": 247, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 856, "icon": "fa-bag-shopping", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "35×28cm", "sku": "ZB-2026-002", "rating": "4.8星", "subcat": "编织", "third": "手工", "tags": ["手工", "环保"], "category_path": ["cat_handicraft", "cat_handicraft_weave", "cat_handicraft_weave_bamboo"]},
    {"id": 3, "name": "山核桃仁 250g", "price": 45, "stock": 156, "category": "食品", "platform": "拼多多", "status": "在售", "sales": 2345, "icon": "fa-seedling", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "250g/袋", "sku": "HT-2026-003", "rating": "4.7星", "subcat": "坚果", "third": "散装", "tags": ["零食", "健康"], "category_path": ["cat_food", "cat_food_nuts", "cat_food_nuts_walnut"]},
    {"id": 4, "name": "云山手工皂套装", "price": 79, "stock": 89, "category": "文创", "platform": "京东", "status": "在售", "sales": 567, "icon": "fa-soap", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "3块装", "sku": "SZ-2026-004", "rating": "4.5星", "subcat": "纸艺", "third": "定制", "tags": ["送礼", "礼盒"], "category_path": ["cat_cultural", "cat_cultural_soap", "cat_cultural_soap_handmade"]},
    {"id": 5, "name": "云山陶瓷杯", "price": 58, "stock": 143, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 1876, "icon": "fa-mug-saucer", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "350ml", "sku": "TC-2026-005", "rating": "4.9星", "subcat": "陶瓷", "third": "手工", "tags": ["手工", "茶具"], "category_path": ["cat_handicraft", "cat_handicraft_ceramic", "cat_handicraft_ceramic_cup"]},
    {"id": 6, "name": "手写书法折扇", "price": 35, "stock": 0, "category": "文创", "platform": "抖音", "status": "下架", "sales": 234, "icon": "fa-scroll", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "10寸", "sku": "FS-2026-006", "rating": "4.6星", "subcat": "纸艺", "third": "定制", "tags": ["文创", "国风"], "category_path": ["cat_cultural", "cat_cultural_paper", "cat_cultural_paper_fan"]},
    {"id": 7, "name": "手工红糖姜茶", "price": 29.9, "stock": 210, "category": "食品", "platform": "淘宝", "status": "在售", "sales": 3456, "icon": "fa-candy-cane", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "300g/盒", "sku": "JC-2026-007", "rating": "4.9星", "subcat": "茶饮", "third": "礼盒装", "tags": ["养生", "暖身"], "category_path": ["cat_food", "cat_food_health", "cat_food_health_tea"]},
    {"id": 8, "name": "云山国风丝巾", "price": 68, "stock": 76, "category": "文创", "platform": "拼多多", "status": "在售", "sales": 789, "icon": "fa-palette", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "90×90cm", "sku": "SJ-2026-008", "rating": "4.8星", "subcat": "编织", "third": "手工", "tags": ["国风", "送礼"], "category_path": ["cat_cultural", "cat_cultural_fabric", "cat_cultural_fabric_scarf"]},
]


def load_products():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for p in data:
                    if "main_image" not in p: p["main_image"] = ""
                    if "sub_images" not in p: p["sub_images"] = []
                    if "video" not in p: p["video"] = ""
                    if "detail_html" not in p: p["detail_html"] = ""
                    if "spec" not in p: p["spec"] = ""
                    if "sku" not in p: p["sku"] = ""
                    if "rating" not in p: p["rating"] = ""
                    if "subcat" not in p: p["subcat"] = ""
                    if "third" not in p: p["third"] = ""
                    if "tags" not in p: p["tags"] = []
                    if "category_path" not in p: p["category_path"] = []
                    if "url" not in p: p["url"] = ""
                    if "original_price" not in p: p["original_price"] = None
                return data
        except Exception:
            return DEFAULT_PRODUCTS
    return DEFAULT_PRODUCTS


def save_products(products):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)


products = load_products()


def _build_product_url(p):
    """根据平台生成商品跳转链接"""
    if p.get("url"):
        return p["url"]
    platform = p.get("platform", "")
    sku = p.get("sku", "")
    import urllib.parse
    name = urllib.parse.quote(p.get("name", ""))
    if "淘宝" in platform:
        return f"https://s.taobao.com/search?q={name}"
    elif "抖音" in platform:
        return f"https://haohuo.jinritemai.com/views/product/detail?id={sku}"
    elif "拼多多" in platform:
        return f"https://mobile.yangkeduo.com/search_result.html?search_key={name}"
    elif "京东" in platform:
        return f"https://search.jd.com/Search?keyword={name}"
    return ""


# ============================================================
# 标签
# ============================================================
DEFAULT_TAGS = [
    {"id": "tag_1", "name": "春茶", "color": "#0d7c4f"},
    {"id": "tag_2", "name": "送礼首选", "color": "#b55a1a"},
    {"id": "tag_3", "name": "高端款", "color": "#7b4fa0"},
    {"id": "tag_4", "name": "爆款", "color": "#b51a3f"},
    {"id": "tag_5", "name": "清仓", "color": "#5f7d95"},
    {"id": "tag_6", "name": "手工", "color": "#2a7faa"},
    {"id": "tag_7", "name": "环保", "color": "#0d7c4f"},
    {"id": "tag_8", "name": "国风", "color": "#b51a5a"},
    {"id": "tag_9", "name": "养生", "color": "#b55a1a"},
    {"id": "tag_10", "name": "礼盒", "color": "#7b4fa0"},
]


def load_tags():
    if os.path.exists(TAG_FILE):
        try:
            with open(TAG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_TAGS
    return DEFAULT_TAGS


def save_tags(tags):
    with open(TAG_FILE, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)


tags_store = load_tags()

# ============================================================
# 日志
# ============================================================
def load_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_logs(logs):
    logs = logs[:LOG_MAX]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def add_log(action_type: str, args: dict, result: str, success: bool):
    type_names = {
        "update_price": "修改价格", "update_stock": "修改库存", "query_low_stock": "查询低库存",
        "publish_to_platforms": "多平台上架", "update_title": "修改标题", "take_off_shelf": "下架商品",
        "put_on_shelf": "上架商品", "query_sales": "查询销量", "create_coupon": "创建优惠券",
        "batch_take_off": "批量下架", "batch_update_price": "批量调价", "query_by_price_range": "价格区间查询",
        "batch_take_off_zero_stock": "清空零库存", "batch_markup_all": "全店加价", "query_sort_by_sales": "销量排行",
        "add_product": "新增商品", "delete_product": "删除商品", "batch_update_stock": "批量改库存",
        "query_by_platform": "按平台查询", "query_by_status": "按状态查询", "batch_put_on_shelf": "批量上架",
        "duplicate_product": "复制商品", "query_by_category": "按分类查询", "export_products": "导出数据",
        "search_product": "关键词搜索", "batch_delete_by_category": "批量删除",
        "query_restock_alert": "库存预警", "export_to_excel": "导出CSV",
        "set_main_image": "设置主图", "add_sub_image": "添加副图",
        "remove_sub_image": "删除副图", "clear_sub_images": "清空副图",
        "query_images": "查看图片", "delete_main_image": "删除主图",
        "set_video": "设置视频", "delete_video": "删除视频",
        "update_detail": "保存图文详情",
        "update_product": "更新商品", "create_product": "创建商品",
        "create_tag": "新建标签", "delete_tag": "删除标签",
        "sync_materials": "同步素材到商品",
        "upload_temp": "上传临时文件",
        "extract_frames": "视频抽帧",
        "decompose_video": "视频反解",
        "parse_generation": "AI 意图识别",
        "match_products": "商品智能匹配",
        "smart_match": "智能匹配商品",
        "market_analysis": "AI电商策略分析",
        "proxy_image": "代理下载图片",
        "proxy_video": "代理下载视频",
        "agnes_generate_image": "后端代调Agnes图片",
        "agnes_generate_video": "后端代调Agnes视频",
        "remember_generation": "记录生成结果",
        "kb_upload": "上传知识库",
        "kb_delete": "删除知识库",
        "kb_update": "编辑知识库",
        "kb_batch_upload": "批量上传知识库",
        "kb_auto_generate": "AI自动提炼FAQ",
        "kb_feedback": "知识库反馈",
        "asset_upload": "上传资产",
        "asset_update": "编辑资产",
        "asset_delete": "删除资产",
        "track_card_click": "卡片点击追踪",
        "recommend_products": "AI推荐商品",
    }
    logs = load_logs()
    log_entry = {
        "id": len(logs) + 1,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": action_type,
        "type_name": type_names.get(action_type, action_type),
        "args": args,
        "result": result,
        "success": success,
    }
    logs.insert(0, log_entry)
    save_logs(logs)


def find_product(name: str):
    if not name:
        return None
    clean = name.replace(" ", "").strip()
    for p in products:
        if p["name"].replace(" ", "") == clean:
            return p
    for p in products:
        pn = p["name"].replace(" ", "")
        if pn in clean or clean in pn:
            return p
    aliases = {
        "茶叶": "茶叶", "礼盒": "茶叶", "竹编": "竹编", "核桃": "核桃",
        "皂": "皂", "陶瓷": "陶瓷", "杯": "陶瓷", "折扇": "扇",
        "扇": "扇", "姜茶": "姜茶", "红糖": "姜茶", "丝巾": "丝巾",
    }
    for key, val in aliases.items():
        if key in clean:
            for p in products:
                if val in p["name"]:
                    return p
    return None


def _delete_file_by_url(url: str):
    if not url or not isinstance(url, str):
        return
    if "/uploads/" not in url:
        return
    try:
        relative = url.split("/uploads/", 1)[1]
        relative = relative.split("?", 1)[0].split("#", 1)[0]
        file_path = (UPLOAD_DIR / relative).resolve()
        if not str(file_path).startswith(str(UPLOAD_DIR)):
            return
        file_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[_delete_file_by_url] 删除失败 {url}: {e}")


# ============================================================
# 商品智能匹配
# ============================================================
class SmartMatchRequest(BaseModel):
    keyword: str
    max_results: int = 5


@app.post("/api/products/smart-match")
async def smart_match_products(req: SmartMatchRequest):
    keyword = req.keyword.strip()
    if not keyword:
        return {"success": False, "message": "缺少关键词", "has_match": False, "matched": []}

    def extract_core_chars(kw):
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '及', '等', '个', '款', '种'}
        chars = set()
        for ch in kw:
            if '\u4e00' <= ch <= '\u9fff' and ch not in stopwords:
                chars.add(ch)
        return chars

    core_chars = extract_core_chars(keyword)
    matched_names = set()
    matches = []

    def add_match(p, score, reason, level):
        if p["name"] in matched_names: return
        matched_names.add(p["name"])
        matches.append({
            "product_name": p["name"], "score": score, "reason": reason, "level": level,
            "price": p.get("price", 0), "stock": p.get("stock", 0), "category": p.get("category", "")
        })

    for p in products:
        if keyword in p["name"]: add_match(p, 100, f"商品名包含「{keyword}」", "L1")
    for p in products:
        if keyword == p.get("category", "") or keyword in p.get("category", ""):
            add_match(p, 80, f"分类为「{p.get('category', '')}」", "L2")
    if len(matches) < req.max_results * 2:
        for p in products:
            if p["name"] in matched_names: continue
            name_chars = set(p["name"])
            common = core_chars & name_chars
            if common and len(common) >= 1:
                score = 70 if len(common) >= 2 else 65
                add_match(p, score, f"包含核心字「{''.join(common)}」", "L2.5")
    for p in products:
        subcat = p.get("subcat", "")
        if subcat and (keyword in subcat or subcat in keyword):
            add_match(p, 65, f"子分类为「{subcat}」", "L3")
    for p in products:
        tags = p.get("tags", [])
        if any(keyword in t or t in keyword for t in tags):
            hit_tags = [t for t in tags if keyword in t or t in keyword]
            add_match(p, 60, f"标签含「{'/'.join(hit_tags)}」", "L4")
    if len(matches) < req.max_results:
        for p in products:
            if p["name"] in matched_names: continue
            name = p["name"]
            if any(ch in name for ch in keyword):
                add_match(p, 50, f"名称部分匹配", "L5")

    matches.sort(key=lambda x: (-x["score"], x["product_name"]))
    matches = matches[:req.max_results]
    has_match = len(matches) > 0
    add_log("smart_match", {"keyword": keyword}, f"匹配 {len(matches)} 个（共 {len(products)} 商品）", True)
    return {"success": True, "has_match": has_match, "keyword": keyword, "matched": matches, "total_products": len(products), "suggest_create": not has_match}


class SuggestCategoryRequest(BaseModel):
    keyword: str


@app.post("/api/products/suggest-category")
async def suggest_category(req: SuggestCategoryRequest):
    keyword = req.keyword.strip()
    CATEGORY_KEYWORDS = {
        "茶叶": ["茶", "龙井", "碧螺春", "铁观音", "大红袍", "毛尖", "普洱", "红茶", "绿茶", "乌龙"],
        "食品": ["糕", "饼", "糖", "果", "肉", "零食", "坚果", "核桃", "开心果", "巴旦木", "干", "蜜饯", "果脯"],
        "手工艺": ["杯", "瓷", "陶", "竹", "编", "木", "手工", "织", "绣", "雕刻", "壶", "碗"],
        "文创": ["丝巾", "折扇", "书签", "明信片", "文创", "国风", "香薰", "手工皂", "笔", "本", "画"],
    }
    suggested = "文创"
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in keyword:
                suggested = cat
                break
        else:
            continue
        break
    return {"success": True, "suggested_category": suggested, "all_categories": ["茶叶", "食品", "手工艺", "文创"]}


# ============================================================
# Agnes 代调
# ============================================================
class AgnesImageRequest(BaseModel):
    prompt: str
    size: str = "1K"
    ratio: str = "1:1"
    reference_images: Optional[List[str]] = None


class AgnesVideoRequest(BaseModel):
    prompt: str
    seconds: str = "5"
    aspect_ratio: str = "9:16"
    reference_images: Optional[List[str]] = None


@app.post("/api/agnes/generate-image")
async def agnes_generate_image(req: AgnesImageRequest, request: Request):
    api_key = os.getenv("AGNES_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "message": "AGNES_API_KEY 环境变量未设置"}
    body = {
        "model": "agnes-image-2.5-flash", "prompt": req.prompt, "size": req.size, "ratio": req.ratio,
        "extra_body": {"response_format": "url"}
    }
    if req.reference_images and len(req.reference_images) > 0:
        body["extra_body"]["image"] = req.reference_images[:5]
        if "参考" not in req.prompt and "reference" not in req.prompt.lower():
            body["prompt"] = req.prompt + "（参考上传的图片风格与主体）"
    try:
        async with httpx.AsyncClient(timeout=180) as http:
            r = await http.post(
                "https://apihub.agnes-ai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body
            )
            if r.status_code != 200:
                return {"success": False, "message": f"Agnes 返回 {r.status_code}: {r.text[:500]}"}
            data = r.json()
    except Exception as e:
        return {"success": False, "message": f"调用 Agnes 异常：{str(e)}"}

    agnes_url = None
    b64_data = None
    if data.get("data") and data["data"][0]:
        agnes_url = data["data"][0].get("url")
        b64_data = data["data"][0].get("b64_json")

    content = None
    ext = ".png"
    if agnes_url:
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
                img_r = await http.get(agnes_url, headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Ecommerce-Bot/1.0)"})
                if img_r.status_code == 200:
                    content = img_r.content
                else:
                    return {"success": False, "message": f"下载 Agnes 图片失败：{img_r.status_code}"}
        except Exception as e:
            return {"success": False, "message": f"下载 Agnes 图片异常：{str(e)}"}
    elif b64_data:
        try:
            content = base64.b64decode(b64_data)
        except Exception as e:
            return {"success": False, "message": f"解码 base64 失败：{str(e)}"}
    else:
        return {"success": False, "message": "Agnes 未返回图片", "raw": str(data)[:500]}

    proxy_dir = UPLOAD_DIR / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:16]
    file_path = proxy_dir / f"{file_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)

    base_url = get_public_base_url(request)
    public_url = f"{base_url}/uploads/proxy/{file_id}{ext}"
    add_log("agnes_generate_image", {"prompt": req.prompt[:50]}, f"已生成 {public_url}", True)
    return {"success": True, "url": public_url}


@app.post("/api/agnes/generate-video")
async def agnes_generate_video(req: AgnesVideoRequest, request: Request):
    api_key = os.getenv("AGNES_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "message": "AGNES_API_KEY 环境变量未设置"}
    body = {
        "model": "agnes-video-2.5-flash", "prompt": req.prompt, "seconds": req.seconds,
        "mode": "reference" if req.reference_images else "text",
        "size": "720P", "aspect_ratio": req.aspect_ratio, "n": 1
    }
    if req.reference_images:
        body["images"] = req.reference_images[:5]
        if "参考" not in req.prompt and "picture" not in req.prompt.lower():
            body["prompt"] = req.prompt + "。以 <Picture 1> 中的商品外观、色调和风格为参考，保持主体一致性。"

    create_data = None
    last_err_msg = ""
    max_create_attempts = 8
    start_time = time.time()
    total_timeout = 240

    for attempt in range(max_create_attempts):
        elapsed = time.time() - start_time
        if elapsed > total_timeout:
            return {"success": False, "message": f"视频队列持续繁忙，已等待 {int(elapsed)}s。请 5-10 分钟后再试。", "queue_full": True}
        try:
            async with httpx.AsyncClient(timeout=60) as http:
                r = await http.post(
                    "https://apihub.agnes-ai.com/v1/videos",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body
                )
                if r.status_code == 200:
                    create_data = r.json()
                    break
                else:
                    err_text = r.text[:500]
                    last_err_msg = err_text
                    if "video_queue_full" in err_text or "queue is full" in err_text.lower():
                        wait = min(30 + 30 * attempt, 60)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        return {"success": False, "message": f"创建视频任务失败：{err_text}"}
        except Exception as e:
            last_err_msg = str(e)
            await asyncio.sleep(5)
            continue

    if not create_data:
        return {"success": False, "message": f"视频队列持续繁忙，已尝试 {max_create_attempts} 次。请稍后重试。", "queue_full": True, "last_error": last_err_msg[:200]}

    video_id = create_data.get("video_id")
    if not video_id:
        return {"success": False, "message": "未返回 video_id"}

    last_status = ""
    for i in range(300):
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                qr = await http.get(
                    f"https://apihub.agnes-ai.com/agnesapi?video_id={video_id}&model_name=agnes-video-2.5-flash",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if qr.status_code != 200:
                    continue
                qd = qr.json()
        except Exception:
            continue

        status = qd.get("status")
        last_status = status or last_status
        if status == "completed":
            video_url = qd.get("url")
            if not video_url:
                return {"success": False, "message": "视频完成但无 URL"}
            try:
                async with httpx.AsyncClient(timeout=180, follow_redirects=True) as http:
                    v_r = await http.get(video_url)
                    if v_r.status_code != 200:
                        return {"success": False, "message": f"下载视频失败：{v_r.status_code}"}
                    content = v_r.content
            except Exception as e:
                return {"success": False, "message": f"下载视频异常：{str(e)}"}

            proxy_dir = UPLOAD_DIR / "proxy"
            proxy_dir.mkdir(parents=True, exist_ok=True)
            file_id = uuid.uuid4().hex[:16]
            file_path = proxy_dir / f"{file_id}.mp4"
            with open(file_path, "wb") as f:
                f.write(content)

            base_url = get_public_base_url(request)
            public_url = f"{base_url}/uploads/proxy/{file_id}.mp4"
            add_log("agnes_generate_video", {"prompt": req.prompt[:50]}, f"已生成 {public_url}", True)
            return {"success": True, "url": public_url}

        if status == "failed":
            err = qd.get("error") or "未知错误"
            return {"success": False, "message": f"视频生成失败：{err}"}

    return {"success": False, "message": f"视频生成超时（最后状态：{last_status}）"}


# ============================================================
# 代理
# ============================================================
class ProxyMediaRequest(BaseModel):
    media_url: str


@app.post("/api/proxy/image")
async def proxy_image(req: ProxyMediaRequest, request: Request):
    if not req.media_url:
        return {"success": False, "message": "缺少 media_url"}
    if not req.media_url.startswith(("http://", "https://")):
        return {"success": False, "message": "URL 格式错误"}
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
            r = await http.get(req.media_url, headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Ecommerce-Bot/1.0)"})
            if r.status_code != 200:
                return {"success": False, "message": f"下载失败：HTTP {r.status_code}"}
            content = r.content
            content_type = r.headers.get("content-type", "image/png").lower()
    except Exception as e:
        return {"success": False, "message": f"下载异常：{str(e)}"}
    if len(content) > 20 * 1024 * 1024:
        return {"success": False, "message": "图片超过 20MB"}
    if "png" in content_type: ext = ".png"
    elif "webp" in content_type: ext = ".webp"
    elif "gif" in content_type: ext = ".gif"
    else: ext = ".jpg"
    proxy_dir = UPLOAD_DIR / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:16]
    file_path = proxy_dir / f"{file_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)
    base_url = get_public_base_url(request)
    public_url = f"{base_url}/uploads/proxy/{file_id}{ext}"
    add_log("proxy_image", {"source": req.media_url[:80]}, f"已代理到 {public_url}", True)
    return {"success": True, "url": public_url, "size": len(content)}


@app.post("/api/proxy/video")
async def proxy_video(req: ProxyMediaRequest, request: Request):
    if not req.media_url:
        return {"success": False, "message": "缺少 media_url"}
    if not req.media_url.startswith(("http://", "https://")):
        return {"success": False, "message": "URL 格式错误"}
    try:
        async with httpx.AsyncClient(timeout=180, follow_redirects=True) as http:
            r = await http.get(req.media_url, headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Ecommerce-Bot/1.0)"})
            if r.status_code != 200:
                return {"success": False, "message": f"下载失败：HTTP {r.status_code}"}
            content = r.content
            content_type = r.headers.get("content-type", "video/mp4").lower()
    except Exception as e:
        return {"success": False, "message": f"下载异常：{str(e)}"}
    if len(content) > 100 * 1024 * 1024:
        return {"success": False, "message": "视频超过 100MB"}
    if "webm" in content_type: ext = ".webm"
    elif "quicktime" in content_type or "mov" in content_type: ext = ".mov"
    else: ext = ".mp4"
    proxy_dir = UPLOAD_DIR / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:16]
    file_path = proxy_dir / f"{file_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)
    base_url = get_public_base_url(request)
    public_url = f"{base_url}/uploads/proxy/{file_id}{ext}"
    add_log("proxy_video", {"source": req.media_url[:80]}, f"已代理到 {public_url}", True)
    return {"success": True, "url": public_url, "size": len(content)}


# ============================================================
# AI 工具
# ============================================================
tools = [
    {"type": "function", "function": {"name": "update_price", "description": "修改指定商品的价格。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "new_price": {"type": "number"}}, "required": ["product_name", "new_price"]}}},
    {"type": "function", "function": {"name": "update_stock", "description": "修改库存。action: increase/decrease/set。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "action": {"type": "string", "enum": ["increase", "decrease", "set"]}, "amount": {"type": "integer"}}, "required": ["product_name", "action", "amount"]}}},
    {"type": "function", "function": {"name": "query_low_stock", "description": "查询库存低于阈值的商品。", "parameters": {"type": "object", "properties": {"threshold": {"type": "integer"}}, "required": ["threshold"]}}},
    {"type": "function", "function": {"name": "publish_to_platforms", "description": "把商品上架到多个平台。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "platforms": {"type": "array", "items": {"type": "string"}}}, "required": ["product_name", "platforms"]}}},
    {"type": "function", "function": {"name": "update_title", "description": "修改商品标题。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "new_title": {"type": "string"}}, "required": ["product_name", "new_title"]}}},
    {"type": "function", "function": {"name": "take_off_shelf", "description": "下架单个商品。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "put_on_shelf", "description": "上架单个商品。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "query_sales", "description": "查询商品销量。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "create_coupon", "description": "创建优惠券。", "parameters": {"type": "object", "properties": {"threshold": {"type": "integer"}, "discount": {"type": "integer"}, "days": {"type": "integer"}}, "required": ["threshold", "discount"]}}},
    {"type": "function", "function": {"name": "batch_take_off", "description": "批量下架分类商品。", "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "batch_update_price", "description": "批量调价。adjust_type: percent/fixed。", "parameters": {"type": "object", "properties": {"category": {"type": "string"}, "adjust_type": {"type": "string", "enum": ["percent", "fixed"]}, "adjust_value": {"type": "number"}}, "required": ["category", "adjust_type", "adjust_value"]}}},
    {"type": "function", "function": {"name": "query_by_price_range", "description": "查询价格区间商品。", "parameters": {"type": "object", "properties": {"min_price": {"type": "number"}, "max_price": {"type": "number"}}, "required": ["min_price", "max_price"]}}},
    {"type": "function", "function": {"name": "batch_take_off_zero_stock", "description": "下架所有零库存商品。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "batch_markup_all", "description": "全店加价百分比。", "parameters": {"type": "object", "properties": {"percent": {"type": "number"}}, "required": ["percent"]}}},
    {"type": "function", "function": {"name": "query_sort_by_sales", "description": "按销量排序。", "parameters": {"type": "object", "properties": {"order": {"type": "string", "enum": ["desc", "asc"]}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "add_product", "description": "新增商品。", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "price": {"type": "number"}, "stock": {"type": "integer"}, "category": {"type": "string"}, "platform": {"type": "string"}, "spec": {"type": "string"}, "sku": {"type": "string"}}, "required": ["name", "price"]}}},
    {"type": "function", "function": {"name": "delete_product", "description": "删除单个商品。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "batch_update_stock", "description": "批量改库存。action: increase/decrease/set。", "parameters": {"type": "object", "properties": {"category": {"type": "string"}, "action": {"type": "string", "enum": ["increase", "decrease", "set"]}, "amount": {"type": "integer"}}, "required": ["category", "action", "amount"]}}},
    {"type": "function", "function": {"name": "query_by_platform", "description": "按平台查询。", "parameters": {"type": "object", "properties": {"platform": {"type": "string"}}, "required": ["platform"]}}},
    {"type": "function", "function": {"name": "query_by_status", "description": "按状态查询。", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["在售", "下架", "待审核"]}}, "required": ["status"]}}},
    {"type": "function", "function": {"name": "batch_put_on_shelf", "description": "批量上架分类商品。", "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "duplicate_product", "description": "复制商品。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "new_name": {"type": "string"}}, "required": ["product_name", "new_name"]}}},
    {"type": "function", "function": {"name": "query_by_category", "description": "按分类查询。", "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "export_products", "description": "导出商品数据。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search_product", "description": "按关键词搜索商品。", "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}}},
    {"type": "function", "function": {"name": "batch_delete_by_category", "description": "批量删除某个分类的所有商品。", "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "query_restock_alert", "description": "查询需要补货的商品。", "parameters": {"type": "object", "properties": {"threshold": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "export_to_excel", "description": "导出商品数据为CSV。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "query_images", "description": "查看某商品的所有图片。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "delete_main_image", "description": "删除某商品的主图。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "remove_sub_image", "description": "删除某商品指定序号的副图。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "index": {"type": "integer"}}, "required": ["product_name", "index"]}}},
    {"type": "function", "function": {"name": "clear_sub_images", "description": "清空某商品的所有副图。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "clear_all_images", "description": "清空某商品的所有图片。", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
]


# ============================================================
# AI 意图识别
# ============================================================
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    history: Optional[List[dict]] = None
    last_generated: Optional[dict] = None
    reference_context: Optional[dict] = None
    conv_id: Optional[str] = None


def _build_history_messages(session: Optional[dict], fallback_history: Optional[List[dict]]):
    msgs = []
    system_parts = [
        "你是电商运营助手。用户会用自然语言下达指令。",
        "你需要判断应该调用哪个工具，并提取参数。",
        "如果用户只是闲聊或问问题，不要调用工具，直接回复。",
        "",
        "【上下文规则】",
        "1. 若用户说'改成红色''再来一张''换成XX''继续'等，参考上文理解其指代对象。",
        "2. 若上文提到某商品，'它''这个''刚才那个'都指该商品。",
        "3. 若用户明确说出新商品名，以新商品为准。",
    ]
    if session:
        lg = session.get("last_generated")
        if lg:
            system_parts.append("")
            system_parts.append(f"【最近一次生成】类型：{lg.get('kind')}，主题：{lg.get('theme')}，商品：{lg.get('product_name') or '无'}")
        if session.get("last_product"):
            system_parts.append(f"【最近操作的商品】{session['last_product']}")
    msgs.append({"role": "system", "content": "\n".join(system_parts)})

    turns = []
    if session and session.get("turns"):
        turns = session["turns"]
    elif fallback_history:
        turns = [{"role": t.get("role"), "text": t.get("text", "")} for t in fallback_history]

    recent = turns[-SESSION_MAX_TURNS * 2:]
    for t in recent:
        role = t.get("role")
        text = t.get("text", "")
        if not text:
            continue
        if role == "user":
            msgs.append({"role": "user", "content": text})
        elif role == "ai":
            msgs.append({"role": "assistant", "content": text[:200]})
    return msgs


@app.post("/api/ai/parse")
async def parse_intent(req: ChatRequest):
    session = None
    if req.session_id:
        session = get_or_create_session(req.session_id)

    messages = _build_history_messages(session, req.history)

    if req.last_generated and messages and messages[0]["role"] == "system":
        lg = req.last_generated
        extra = f"\n【前端上报的上次生成】类型：{lg.get('kind')}，主题：{lg.get('theme')}，商品：{lg.get('product_name') or '无'}"
        messages[0]["content"] += extra

    if req.reference_context and messages and messages[0]["role"] == "system":
        rc = req.reference_context
        extra = f"\n【用户当前参考素材】图片 {rc.get('images', 0)} 张，视频关键帧 {rc.get('keyframes', 0)} 张"
        messages[0]["content"] += extra

    kb_results = search_knowledge_hybrid(req.text, top_k=3)
    kb_hits_full = []
    if kb_results and messages and messages[0]["role"] == "system":
        kb_ctx = "\n\n【企业知识库参考资料】\n"
        for i, r in enumerate(kb_results, 1):
            item = r["item"]
            kb_ctx += f"{i}. 《{item['title']}》({item.get('category_name', item.get('category', ''))})：{item.get('content', '')[:500]}\n"
            kb_hits_full.append({
                "id": item["id"], "title": item["title"],
                "category": item.get("category", ""), "category_name": item.get("category_name", ""),
                "score": round(r["score"], 3), "match_type": r["match_type"]
            })
            try:
                record_kb_hit(item["id"])
            except Exception:
                pass
        kb_ctx += "如果用户问题与以上资料相关，请优先依据资料回答，不要编造。\n"
        messages[0]["content"] += kb_ctx

    assets_hits = []
    trigger_words = ['图', '视频', '素材', '看看', '发我', '发个', '展示', '什么样', '实拍']
    if any(w in req.text for w in trigger_words):
        assets_hits = search_assets(req.text, asset_type="all", top_k=4)
        if assets_hits and messages and messages[0]["role"] == "system":
            a_ctx = "\n\n【可用素材库（可推荐给用户）】\n"
            for a in assets_hits:
                a_ctx += f"- [{a['type']}] {a['name']} → {a['url']}\n"
            a_ctx += "如用户要求看图/视频，可从上面挑选匹配的发送，回复末尾附上素材链接。\n"
            messages[0]["content"] += a_ctx

    messages.append({"role": "user", "content": req.text})

    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        if session is not None:
            append_turn(session, "user", req.text)

        if msg.tool_calls:
            call = msg.tool_calls[0]
            try:
                args = json.loads(call.function.arguments)
            except Exception:
                args = {}
            if session is not None:
                append_turn(session, "ai", f"[调用 {call.function.name}]", type=call.function.name, args=args)
                pn = args.get("product_name") or args.get("name")
                if pn:
                    session["last_product"] = pn
                    _save_sessions(sessions_store)
            return {"type": call.function.name, "args": args, "session_id": session["id"] if session else None, "kb_hits": kb_hits_full, "assets": assets_hits}
        else:
            reply_text = msg.content or ""
            if session is not None:
                append_turn(session, "ai", reply_text, type="chat")
            return {"type": "chat", "text": reply_text, "session_id": session["id"] if session else None, "kb_hits": kb_hits_full, "assets": assets_hits}
    except Exception as e:
        return {"type": "error", "text": str(e)}


# ============================================================
# 记录生成结果
# ============================================================
class RememberGenRequest(BaseModel):
    session_id: Optional[str] = None
    kind: str
    theme: Optional[str] = ""
    product_name: Optional[str] = ""
    urls: Optional[List[str]] = None
    video_url: Optional[str] = None
    poster_url: Optional[str] = None


@app.post("/api/ai/remember-generation")
async def remember_generation(req: RememberGenRequest):
    if not req.session_id:
        return {"success": False, "message": "缺少 session_id"}
    session = get_or_create_session(req.session_id)
    session["last_generated"] = {
        "kind": req.kind, "theme": req.theme or "", "product_name": req.product_name or "",
        "urls": req.urls or [], "videoUrl": req.video_url or "", "posterUrl": req.poster_url or "",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_sessions(sessions_store)
    add_log("remember_generation", {"kind": req.kind, "theme": req.theme}, "已记录", True)
    return {"success": True, "message": "已记录", "last_generated": session["last_generated"]}


@app.get("/api/ai/session/{session_id}")
async def get_session(session_id: str):
    s = sessions_store.get(session_id)
    if not s:
        return {"success": False, "message": "会话不存在或已过期"}
    return {"success": True, "session": {"id": s["id"], "turns": s["turns"][-20:], "last_generated": s.get("last_generated"), "last_product": s.get("last_product")}}


@app.post("/api/ai/session/clear")
async def clear_session(payload: dict):
    sid = payload.get("session_id")
    if sid and sid in sessions_store:
        sessions_store.pop(sid, None)
        _save_sessions(sessions_store)
    return {"success": True}


# ============================================================
# 生成意图识别
# ============================================================
class ParseGenRequest(BaseModel):
    text: str
    available_products: Optional[List[str]] = []
    has_reference: Optional[bool] = False
    ref_types: Optional[List[str]] = []
    session_id: Optional[str] = None
    last_generated: Optional[dict] = None


@app.post("/api/ai/parse-generation")
async def parse_generation(req: ParseGenRequest):
    text = req.text or ''
    available_products = req.available_products or []
    has_reference = req.has_reference or False
    ref_types = req.ref_types or []

    if not text:
        return {"kind": None, "is_bound": False, "product_name": ""}

    last_gen_hint = ""
    last_product_hint = ""
    if req.session_id and req.session_id in sessions_store:
        s = sessions_store[req.session_id]
        lg = s.get("last_generated")
        if lg:
            last_gen_hint = f"上一次生成了{lg.get('kind')}，主题「{lg.get('theme')}」，商品「{lg.get('product_name') or '无'}」"
        if s.get("last_product"):
            last_product_hint = f"上一次操作的商品是「{s['last_product']}」"
    elif req.last_generated:
        lg = req.last_generated
        last_gen_hint = f"上一次生成了{lg.get('kind')}，主题「{lg.get('theme')}」，商品「{lg.get('product_name') or '无'}」"

    products_str = '、'.join(available_products) if available_products else '（暂无）'
    ref_str = f'有（{"/".join(ref_types)}）' if has_reference else '无'

    prompt = f"""你是电商素材生成助手。请分析用户输入，判断生成意图。

用户输入：{text}
有无参考素材：{ref_str}
可用商品列表：{products_str}
{last_gen_hint}
{last_product_hint}

【上下文规则】
- 若用户说"再来一张/再来一个/继续/同样的"，沿用上一次的 kind / theme / product_name
- 若用户说"改成XX色/换成XX风格"，只改 extra_desc，其它沿用
- 若用户明说新商品，覆盖上次的商品

请严格返回以下 JSON 格式（不要 markdown 代码块，不要任何解释）：
{{
  "kind": "images" | "video" | "poster" | null,
  "is_bound": true / false,
  "product_name": "商品名，仅当 is_bound=true 时填，必须是上面列表中的一个；否则填空字符串",
  "count": 数字（图片张数，默认1）,
  "seconds": "视频时长字符串，默认5",
  "poster_type": "促销海报 | 新品海报 | 国潮海报 | 简约海报 | 自由海报",
  "free_theme": "自由主题关键词，如'茶叶'、'母亲节'、'国潮'",
  "ai_suggestion": "一句给用户的建议"
}}

判断规则：
1. 若文本含"给XXX生成/为XXX制作/帮XXX做"，且 XXX 在商品列表中 → is_bound=true, product_name=XXX
2. 若文本含"给XXX生成"，但 XXX 不在列表中 → is_bound=false, free_theme=XXX
3. 若文本无明确商品指向 → is_bound=false，kind 根据文本判断
4. 若用户上传了参考素材 → 优先推荐生成同类素材
5. 若文本含"海报" → kind=poster；含"视频" → kind=video；含"图/主图" → kind=images
6. 若文本既无"生成"意图也无"操作"意图 → kind=null
7. 若文本含"再来一张/继续/同样"，沿用 {last_gen_hint}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith('```'):
            parts = raw.split('```')
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith('json'):
                    raw = raw[4:]
                raw = raw.strip()
        result = json.loads(raw)
        add_log("parse_generation", {"text": text[:50]}, f"kind={result.get('kind')}", True)
        return result
    except Exception as e:
        print(f"[parse_generation] 失败: {e}")
        return {"kind": None, "is_bound": False, "product_name": "", "error": str(e)}


# ============================================================
# 商品匹配（旧版）
# ============================================================
class MatchRequest(BaseModel):
    keywords: List[str]
    generation_context: Optional[str] = ""


@app.post("/api/products/match")
async def match_products(req: MatchRequest):
    if not products or not req.keywords:
        return {"success": True, "matches": []}
    matches = []
    seen = set()
    for kw in req.keywords:
        clean = kw.replace(" ", "").strip().lower()
        for p in products:
            pn = p["name"].replace(" ", "").lower()
            if pn == clean:
                key = (p["name"], "exact")
                if key not in seen:
                    seen.add(key)
                    matches.append({"product_name": p["name"], "match_level": "exact", "score": 1.0, "reason": f"精确匹配「{kw}」"})
    if matches:
        add_log("match_products", {"keywords": req.keywords}, f"L1 精确匹配 {len(matches)} 个", True)
        return {"success": True, "matches": matches}

    products_summary = [{"name": p["name"], "category": p.get("category", ""), "subcat": p.get("subcat", ""), "tags": p.get("tags", []), "spec": p.get("spec", ""), "price": p.get("price", 0)} for p in products]
    try:
        prompt = f"""你是电商商品匹配助手。

用户输入：{', '.join(req.keywords)}
生成内容类型：{req.generation_context or '图片'}

可选商品列表：
{json.dumps(products_summary, ensure_ascii=False, indent=2)}

请判断用户输入最可能对应哪个（或哪些）商品。考虑：
1. 商品名称的语义相关性
2. 商品的分类
3. 商品的子分类
4. 商品的标签
5. 商品的规格、用途

严格返回 JSON（不要 markdown 代码块，不要任何解释）：
{{
  "matches": [
    {{
      "product_name": "商品名（必须是列表中的一个）",
      "score": 0.85,
      "reason": "简短说明为什么匹配（15字以内）"
    }}
  ]
}}

规则：
- 只返回 score >= 0.5 的匹配
- 最多返回 3 个匹配，按 score 降序
"""
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        ai_result = json.loads(raw)
        ai_matches = ai_result.get("matches", [])
        valid_names = {p["name"] for p in products}
        for m in ai_matches:
            if m.get("product_name") in valid_names:
                key = (m["product_name"], "semantic")
                if key not in seen:
                    seen.add(key)
                    matches.append({"product_name": m["product_name"], "match_level": "semantic", "score": float(m.get("score", 0.6)), "reason": m.get("reason", "语义相关")})
    except Exception as e:
        print(f"[match_products] AI 语义匹配失败: {e}")

    if not matches:
        for kw in req.keywords:
            clean = kw.replace(" ", "").lower()
            for p in products:
                p_category = p.get("category", "").lower()
                p_subcat = p.get("subcat", "").lower()
                p_tags = [t.lower() for t in p.get("tags", [])]
                hit = False
                if clean in p_category or clean in p_subcat: hit = True
                if any(clean in t for t in p_tags): hit = True
                if clean in p["name"].lower(): hit = True
                if hit:
                    key = (p["name"], "category")
                    if key not in seen:
                        seen.add(key)
                        matches.append({"product_name": p["name"], "match_level": "category", "score": 0.6, "reason": f"命中分类/标签"})
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:5]
    add_log("match_products", {"keywords": req.keywords}, f"共匹配 {len(matches)} 个商品", True)
    return {"success": True, "matches": matches}


# ============================================================
# AI 电商市场顾问
# ============================================================
class MarketAnalysisRequest(BaseModel):
    theme: str
    kind: str
    platform: Optional[str] = "全平台"


@app.post("/api/ai/market-analysis")
async def market_analysis(req: MarketAnalysisRequest):
    if not req.theme:
        return {"success": False, "message": "缺少主题"}
    now = datetime.now()
    month = now.month
    if month in (3, 4, 5): season_hint = "春季"; festival_hint = "母亲节/五一"
    elif month in (6, 7, 8): season_hint = "夏季"; festival_hint = "618/端午节/暑期"
    elif month in (9, 10, 11): season_hint = "秋季"; festival_hint = "中秋节/国庆/双11"
    else: season_hint = "冬季"; festival_hint = "双12/圣诞/元旦/年货节"
    kind_names = {"images": "商品主图", "video": "短视频", "poster": "宣传海报"}
    kind_name = kind_names.get(req.kind, "商品素材")
    prompt = f"""你是资深电商运营专家 + 视觉营销总监。用户想为「{req.theme}」生成{kind_name}。

当前时间：{now.strftime('%Y年%m月%d日')}（{season_hint}，临近节日：{festival_hint}）
目标平台：{req.platform}
生成类型：{kind_name}

请从**电商转化**角度深度分析，给出：

【1. 目标用户画像】
【2. 当下市场趋势】
【3. 电商转化建议】
【4. 推荐视觉方案】
【5. 可直接使用的生成 Prompt】

严格返回以下 JSON（不要 markdown 代码块，不要任何解释）：
{{
  "user_profile": "一句话用户画像（30字内）",
  "purchase_scenario": "主要购买场景（15字内）",
  "season_tag": "{season_hint}",
  "festival_tag": "{festival_hint}",
  "trend_style": "当下最流行的视觉风格（20字内）",
  "color_scheme": "推荐配色（20字内）",
  "composition": "推荐构图（20字内）",
  "lighting": "推荐光影（15字内）",
  "selling_points": ["核心卖点1", "核心卖点2", "核心卖点3"],
  "conversion_tips": ["转化建议1", "转化建议2", "转化建议3"],
  "avoid": ["过时元素1", "过时元素2"],
  "prompt_enhancement": "100-150字的完整生成 prompt（中英混排）",
  "ai_comment": "一句给用户的核心建议（40字内）"
}}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        result = json.loads(raw)
        add_log("market_analysis", {"theme": req.theme, "kind": req.kind}, "深度分析成功", True)
        return {"success": True, "analysis": result}
    except Exception as e:
        print(f"[market_analysis] 失败: {e}")
        return {"success": True, "analysis": {"user_profile": "25-45岁品质生活人群", "purchase_scenario": "自用/送礼", "season_tag": season_hint, "festival_tag": festival_hint, "trend_style": "现代简约，暖色调", "color_scheme": "奶油白 + 焦糖棕", "composition": "中心对称，留白30%", "lighting": "柔光箱打光，暖色温", "selling_points": ["品质感", "设计感", "性价比"], "conversion_tips": ["第一眼突出商品", "叠加使用场景", "弱化复杂背景"], "avoid": ["过时的浓重滤镜", "杂乱背景"], "prompt_enhancement": "modern minimalist style, warm beige tone, high-end commercial photography, soft studio lighting, clean background, focus on product texture, ecommerce main image, high detail, professional", "ai_comment": "建议突出品质感与使用场景，避免过度装饰"}}


# ============================================================
# ★ AI 智能推荐商品（新增）
# ============================================================
class RecommendProductsRequest(BaseModel):
    text: str
    max_results: int = 3


@app.post("/api/ai/recommend-products")
async def recommend_products(req: RecommendProductsRequest):
    """根据客户对话，AI 推荐最匹配的商品"""
    if not products:
        return {"success": True, "products": []}

    products_summary = [
        {"id": p["id"], "name": p["name"], "category": p.get("category", ""),
         "subcat": p.get("subcat", ""), "tags": p.get("tags", []),
         "price": p.get("price", 0), "spec": p.get("spec", "")}
        for p in products
    ]

    prompt = f"""你是电商客服助手。客户说了以下话，请从商品库中挑选最匹配的商品推荐给客户。

客户说：{req.text}

商品库：
{json.dumps(products_summary, ensure_ascii=False, indent=2)}

请严格返回 JSON（不要 markdown 代码块）：
{{
  "product_ids": [商品ID数组，按匹配度排序，最多 {req.max_results} 个],
  "reason": "简短说明推荐理由"
}}

规则：
- 优先按商品名、分类、标签的语义匹配
- 如果客户没有明确指向，返回销量最高的前 {req.max_results} 个
- 如果完全没有相关商品，返回空数组 []
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        result = json.loads(raw)
        ids = result.get("product_ids", [])
        matched = [p for p in products if p["id"] in ids]
        for p in matched:
            if not p.get("url"):
                p["url"] = _build_product_url(p)
        add_log("recommend_products", {"text": req.text[:50]}, f"推荐 {len(matched)} 个商品", True)
        return {"success": True, "products": matched, "reason": result.get("reason", "")}
    except Exception as e:
        print(f"[recommend_products] 失败: {e}")
        sorted_products = sorted(products, key=lambda p: p.get("sales", 0), reverse=True)
        top = sorted_products[:req.max_results]
        for p in top:
            if not p.get("url"):
                p["url"] = _build_product_url(p)
        return {"success": True, "products": top, "fallback": True}


# ============================================================
# ★ 卡片点击追踪（新增）
# ============================================================
class TrackClickRequest(BaseModel):
    product_id: Optional[int] = None
    product_name: Optional[str] = ""
    action: Optional[str] = "view"
    conv_id: Optional[str] = ""
    time: Optional[str] = ""


@app.post("/api/ai/track-card-click")
async def track_card_click(req: TrackClickRequest):
    data = load_card_clicks()
    data["total_clicked"] = data.get("total_clicked", 0) + 1
    pid = str(req.product_id) if req.product_id else "unknown"
    if pid not in data["by_product"]:
        data["by_product"][pid] = {"name": req.product_name, "shown": 0, "clicked": 0}
    data["by_product"][pid]["clicked"] = data["by_product"][pid].get("clicked", 0) + 1
    save_card_clicks(data)
    add_log("track_card_click", {"product_id": req.product_id, "action": req.action}, "已记录点击", True)
    return {"success": True}


# ============================================================
# ★ 优惠券（新增）
# ============================================================
@app.get("/api/coupons")
async def get_coupons():
    return {"coupons": load_coupons()}


# ============================================================
# 商品数据接口
# ============================================================
@app.get("/api/products")
async def get_products():
    for p in products:
        if not p.get("url"):
            p["url"] = _build_product_url(p)
    return {"products": products}


@app.get("/api/categories/tree")
async def get_category_tree():
    return {"categories": CATEGORY_TREE}


@app.get("/api/categories/map")
async def get_category_map():
    return {"map": CATEGORY_MAP}


@app.get("/api/tags")
async def get_tags():
    return {"tags": tags_store}


class TagCreateRequest(BaseModel):
    name: str
    color: Optional[str] = "#4dabf7"


@app.post("/api/tags/create")
async def create_tag(req: TagCreateRequest):
    global tags_store
    name = req.name.strip()
    if not name:
        return {"success": False, "message": "标签名称不能为空"}
    if any(t["name"] == name for t in tags_store):
        return {"success": False, "message": f"标签「{name}」已存在"}
    new_id = f"tag_{len(tags_store) + 1}_{int(datetime.now().timestamp())}"
    new_tag = {"id": new_id, "name": name, "color": req.color}
    tags_store.append(new_tag)
    save_tags(tags_store)
    add_log("create_tag", {"name": name}, f"已新建标签「{name}」", True)
    return {"success": True, "message": f"已新建标签「{name}」", "tag": new_tag}


class TagDeleteRequest(BaseModel):
    tag_id: str


@app.post("/api/tags/delete")
async def delete_tag(req: TagDeleteRequest):
    global tags_store
    tag = next((t for t in tags_store if t["id"] == req.tag_id), None)
    if not tag:
        return {"success": False, "message": "标签不存在"}
    tags_store = [t for t in tags_store if t["id"] != req.tag_id]
    save_tags(tags_store)
    add_log("delete_tag", {"tag_id": req.tag_id}, f"已删除标签「{tag['name']}」", True)
    return {"success": True, "message": f"已删除标签「{tag['name']}」"}


@app.get("/api/logs")
async def get_logs():
    return {"logs": load_logs()}


class ExecuteRequest(BaseModel):
    type: str
    args: dict


@app.post("/api/ai/execute")
async def execute_action(req: ExecuteRequest):
    global products
    try:
        result = do_action(req.type, req.args)
        add_log(req.type, req.args, result.get("message", ""), result.get("success", False))
        return result
    except Exception as e:
        msg = f"执行失败：{str(e)}"
        add_log(req.type, req.args, msg, False)
        return {"success": False, "message": msg}


def do_action(t, args):
    if t == "update_price":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        old = p["price"]; p["price"] = args["new_price"]
        save_products(products)
        return {"success": True, "message": f"已将「{p['name']}」的价格从 ¥{old} 改为 ¥{p['price']}"}
    if t == "update_stock":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        old = p["stock"]; action = args.get("action"); amount = args.get("amount", 0)
        if action == "increase": p["stock"] = old + amount
        elif action == "decrease": p["stock"] = max(0, old - amount)
        else: p["stock"] = amount
        save_products(products)
        action_text = {"increase": "增加", "decrease": "减少", "set": "设置为"}.get(action, "调整")
        return {"success": True, "message": f"已将「{p['name']}」的库存{action_text} {amount} 件，当前库存 {p['stock']} 件"}
    if t == "query_low_stock":
        threshold = args.get("threshold", 50)
        low = [p for p in products if p["stock"] < threshold]
        if not low: return {"success": True, "message": f"没有库存低于 {threshold} 件的商品"}
        lines = [f"· {p['name']}：库存 {p['stock']} 件" for p in low]
        return {"success": True, "message": f"库存低于 {threshold} 件的商品共 {len(low)} 个：\n" + "\n".join(lines)}
    if t == "publish_to_platforms":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        platforms = args.get("platforms", [])
        p["platform"] = platforms[0] if platforms else p["platform"]
        p["status"] = "在售"
        save_products(products)
        return {"success": True, "message": f"已将「{p['name']}」上架到 {', '.join(platforms)}"}
    if t == "update_title":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        old = p["name"]; p["name"] = args["new_title"]
        save_products(products)
        return {"success": True, "message": f"已将「{old}」的标题改为「{p['name']}」"}
    if t == "take_off_shelf":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        p["status"] = "下架"; save_products(products)
        return {"success": True, "message": f"已下架「{p['name']}」"}
    if t == "put_on_shelf":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        p["status"] = "在售"; save_products(products)
        return {"success": True, "message": f"已上架「{p['name']}」"}
    if t == "query_sales":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        return {"success": True, "message": f"「{p['name']}」累计销量 {p['sales']} 件，当前价格 ¥{p['price']}，库存 {p['stock']} 件"}
    if t == "create_coupon":
        return {"success": True, "message": f"已创建优惠券：满 {args.get('threshold')} 减 {args.get('discount')}，有效期 {args.get('days', 7)} 天"}
    if t == "batch_take_off":
        category = args.get("category")
        matched = [p for p in products if p["category"] == category]
        if not matched: return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
        for p in matched: p["status"] = "下架"
        save_products(products)
        names = "、".join([p["name"] for p in matched])
        return {"success": True, "message": f"已批量下架「{category}」分类共 {len(matched)} 个商品：\n{names}"}
    if t == "batch_update_price":
        category = args.get("category"); adjust_type = args.get("adjust_type", "percent"); adjust_value = args.get("adjust_value", 0)
        matched = [p for p in products if p["category"] == category]
        if not matched: return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
        lines = []
        for p in matched:
            old = p["price"]
            if adjust_type == "percent": p["price"] = round(old * (1 + adjust_value / 100), 2)
            else: p["price"] = round(old + adjust_value, 2)
            lines.append(f"· {p['name']}：¥{old} → ¥{p['price']}")
        save_products(products)
        action_text = f"{'+' if adjust_value > 0 else ''}{adjust_value}%" if adjust_type == "percent" else f"{'+' if adjust_value > 0 else ''}{adjust_value}元"
        return {"success": True, "message": f"已将「{category}」分类商品价格统一调整（{action_text}）：\n" + "\n".join(lines)}
    if t == "query_by_price_range":
        min_p = args.get("min_price", 0); max_p = args.get("max_price", 99999)
        matched = [p for p in products if min_p <= p["price"] <= max_p]
        if not matched: return {"success": True, "message": f"没有价格在 ¥{min_p} - ¥{max_p} 之间的商品"}
        lines = [f"· {p['name']}：¥{p['price']}（库存 {p['stock']}）" for p in matched]
        return {"success": True, "message": f"价格在 ¥{min_p} - ¥{max_p} 之间的商品共 {len(matched)} 个：\n" + "\n".join(lines)}
    if t == "batch_take_off_zero_stock":
        matched = [p for p in products if p["stock"] == 0]
        if not matched: return {"success": True, "message": "没有库存为 0 的商品"}
        for p in matched: p["status"] = "下架"
        save_products(products)
        names = "、".join([p["name"] for p in matched])
        return {"success": True, "message": f"已下架库存为 0 的商品共 {len(matched)} 个：\n{names}"}
    if t == "batch_markup_all":
        percent = args.get("percent", 10); lines = []
        for p in products:
            old = p["price"]; p["price"] = round(old * (1 + percent / 100), 2)
            lines.append(f"· {p['name']}：¥{old} → ¥{p['price']}")
        save_products(products)
        return {"success": True, "message": f"已给所有商品加价 {percent}%：\n" + "\n".join(lines)}
    if t == "query_sort_by_sales":
        order = args.get("order", "desc"); limit = args.get("limit", 10)
        sorted_products = sorted(products, key=lambda p: p["sales"], reverse=(order == "desc"))[:limit]
        lines = [f"{i+1}. {p['name']}：销量 {p['sales']} 件" for i, p in enumerate(sorted_products)]
        order_text = "从高到低" if order == "desc" else "从低到高"
        return {"success": True, "message": f"按销量{order_text}排序（前 {len(sorted_products)} 个）：\n" + "\n".join(lines)}
    if t == "add_product":
        name = args.get("name"); price = args.get("price")
        stock = args.get("stock", 0); category = args.get("category", "文创"); platform = args.get("platform", "淘宝")
        spec = args.get("spec", ""); sku = args.get("sku", "")
        if not name or price is None: return {"success": False, "message": "缺少商品名称或价格"}
        if any(p["name"] == name for p in products): return {"success": False, "message": f"商品「{name}」已存在"}
        new_id = max([p["id"] for p in products], default=0) + 1
        icon_map = {"茶叶": "fa-leaf", "手工艺": "fa-bag-shopping", "食品": "fa-seedling", "文创": "fa-palette"}
        new_product = {"id": new_id, "name": name, "price": price, "stock": stock, "category": category, "platform": platform, "status": "在售", "sales": 0, "icon": icon_map.get(category, "fa-box"), "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": spec, "sku": sku, "rating": "", "subcat": "", "third": "", "tags": [], "category_path": [], "url": "", "original_price": None}
        products.append(new_product)
        save_products(products)
        return {"success": True, "message": f"已新增商品「{name}」：价格 ¥{price}，库存 {stock} 件，分类 {category}，平台 {platform}"}
    if t == "delete_product":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        name = p["name"]; products.remove(p); save_products(products)
        return {"success": True, "message": f"已删除商品「{name}」"}
    if t == "batch_update_stock":
        category = args.get("category"); action = args.get("action", "increase"); amount = args.get("amount", 0)
        matched = [p for p in products if p["category"] == category]
        if not matched: return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
        lines = []
        for p in matched:
            old = p["stock"]
            if action == "increase": p["stock"] = old + amount
            elif action == "decrease": p["stock"] = max(0, old - amount)
            else: p["stock"] = amount
            lines.append(f"· {p['name']}：{old} → {p['stock']}")
        save_products(products)
        action_text = {"increase": "增加", "decrease": "减少", "set": "设置为"}.get(action, "调整")
        return {"success": True, "message": f"已将「{category}」分类商品库存统一{action_text} {amount} 件：\n" + "\n".join(lines)}
    if t == "query_by_platform":
        platform = args.get("platform"); matched = [p for p in products if p["platform"] == platform]
        if not matched: return {"success": True, "message": f"「{platform}」平台上没有商品"}
        lines = [f"· {p['name']}（{p['status']}）：¥{p['price']}，库存 {p['stock']}" for p in matched]
        return {"success": True, "message": f"「{platform}」平台共有 {len(matched)} 个商品：\n" + "\n".join(lines)}
    if t == "query_by_status":
        status = args.get("status"); matched = [p for p in products if p["status"] == status]
        if not matched: return {"success": True, "message": f"没有「{status}」状态的商品"}
        lines = [f"· {p['name']}（{p['category']}）：¥{p['price']}，库存 {p['stock']}" for p in matched]
        return {"success": True, "message": f"「{status}」状态的商品共 {len(matched)} 个：\n" + "\n".join(lines)}
    if t == "batch_put_on_shelf":
        category = args.get("category"); matched = [p for p in products if p["category"] == category]
        if not matched: return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
        for p in matched: p["status"] = "在售"
        save_products(products)
        names = "、".join([p["name"] for p in matched])
        return {"success": True, "message": f"已批量上架「{category}」分类共 {len(matched)} 个商品：\n{names}"}
    if t == "duplicate_product":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        new_name = args.get("new_name")
        if not new_name: return {"success": False, "message": "缺少新商品名称"}
        if any(prod["name"] == new_name for prod in products): return {"success": False, "message": f"商品「{new_name}」已存在"}
        new_id = max([p["id"] for p in products], default=0) + 1
        new_product = dict(p); new_product["id"] = new_id; new_product["name"] = new_name; new_product["sales"] = 0
        products.append(new_product); save_products(products)
        return {"success": True, "message": f"已复制「{p['name']}」为「{new_name}」"}
    if t == "query_by_category":
        category = args.get("category"); matched = [p for p in products if p["category"] == category]
        if not matched: return {"success": True, "message": f"没有「{category}」分类的商品"}
        lines = [f"· {p['name']}（{p['status']}）：¥{p['price']}，库存 {p['stock']}，销量 {p['sales']}" for p in matched]
        return {"success": True, "message": f"「{category}」分类共 {len(matched)} 个商品：\n" + "\n".join(lines)}
    if t == "export_products":
        return {"success": True, "message": f"共 {len(products)} 个商品，数据已可从前端 /api/products 接口获取"}
    if t == "search_product":
        keyword = args.get("keyword", "").strip()
        if not keyword: return {"success": False, "message": "搜索关键词不能为空"}
        matched = [p for p in products if keyword in p["name"] or keyword in p["category"] or keyword in p.get("platform", "")]
        if not matched: return {"success": True, "message": f"没有找到包含「{keyword}」的商品"}
        lines = [f"· {p['name']}（{p['category']}）：¥{p['price']}，库存 {p['stock']}" for p in matched]
        return {"success": True, "message": f"搜索「{keyword}」共找到 {len(matched)} 个商品：\n" + "\n".join(lines)}
    if t == "batch_delete_by_category":
        category = args.get("category"); matched = [p for p in products if p["category"] == category]
        if not matched: return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
        names = "、".join([p["name"] for p in matched])
        for p in matched:
            _delete_file_by_url(p.get("main_image", ""))
            _delete_file_by_url(p.get("video", ""))
            for sub in p.get("sub_images", []):
                _delete_file_by_url(sub)
            products.remove(p)
        save_products(products)
        return {"success": True, "message": f"已删除「{category}」分类共 {len(matched)} 个商品：\n{names}"}
    if t == "query_restock_alert":
        threshold = args.get("threshold", 50)
        matched = [p for p in products if p["stock"] < threshold]
        if not matched: return {"success": True, "message": f"没有需要补货的商品（库存都 ≥ {threshold} 件）"}
        lines = [f"· {p['name']}：库存仅 {p['stock']} 件（分类 {p['category']}）" for p in matched]
        return {"success": True, "message": f"⚠️ 需要补货的商品共 {len(matched)} 个（库存 < {threshold} 件）：\n" + "\n".join(lines)}
    if t == "export_to_excel":
        return {"success": True, "message": f"数据已准备好（共 {len(products)} 个商品）", "csv": True}
    if t == "query_images":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        main_status = "有主图" if p.get("main_image") else "无主图"
        sub_count = len(p.get("sub_images", []))
        video_status = "有视频" if p.get("video") else "无视频"
        return {"success": True, "message": f"「{p['name']}」媒体信息：\n· 主图：{main_status}\n· 副图：{sub_count} 张\n· 视频：{video_status}"}
    if t == "delete_main_image":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        if not p.get("main_image"): return {"success": False, "message": f"「{p['name']}」没有主图"}
        old_url = p["main_image"]
        p["main_image"] = ""
        save_products(products)
        _delete_file_by_url(old_url)
        return {"success": True, "message": f"已删除「{p['name']}」的主图"}
    if t == "remove_sub_image":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        index = args.get("index", 1)
        subs = p.get("sub_images", [])
        if index < 1 or index > len(subs): return {"success": False, "message": f"副图序号 {index} 不存在，当前共 {len(subs)} 张"}
        old_url = subs.pop(index - 1)
        p["sub_images"] = subs
        save_products(products)
        _delete_file_by_url(old_url)
        return {"success": True, "message": f"已删除「{p['name']}」的第 {index} 张副图，剩余 {len(subs)} 张"}
    if t == "clear_sub_images":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        subs = p.get("sub_images", [])
        if len(subs) == 0: return {"success": False, "message": f"「{p['name']}」没有副图"}
        for url in subs:
            _delete_file_by_url(url)
        p["sub_images"] = []
        save_products(products)
        return {"success": True, "message": f"已清空「{p['name']}」的所有副图（共 {len(subs)} 张）"}
    if t == "clear_all_images":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        main_exist = bool(p.get("main_image")); sub_count = len(p.get("sub_images", []))
        if not main_exist and sub_count == 0: return {"success": False, "message": f"「{p['name']}」没有任何图片"}
        _delete_file_by_url(p.get("main_image", ""))
        for url in p.get("sub_images", []):
            _delete_file_by_url(url)
        p["main_image"] = ""; p["sub_images"] = []
        save_products(products)
        return {"success": True, "message": f"已清空「{p['name']}」的所有图片"}
    return {"success": False, "message": f"未知操作类型：{t}"}


# ============================================================
# 图片上传 / 删除
# ============================================================
@app.post("/api/images/upload")
async def upload_image(
    request: Request,
    product_name: str = Form(...),
    image_type: str = Form(...),
    file: UploadFile = File(...)
):
    p = None
    if image_type != "detail":
        p = find_product(product_name)
        if not p:
            return {"success": False, "message": f"未找到商品：{product_name}"}
    content = await file.read()
    max_size = 50 * 1024 * 1024 if image_type == "video" else 2 * 1024 * 1024
    if len(content) > max_size:
        limit_text = "50MB" if image_type == "video" else "2MB"
        return {"success": False, "message": f"文件不能超过 {limit_text}"}
    if image_type == "video":
        if not file.content_type or not file.content_type.startswith("video/"):
            return {"success": False, "message": "只支持视频文件"}
    else:
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"success": False, "message": "只支持图片文件"}
    if image_type == "sub":
        safe_name = "".join(c for c in product_name if c.isalnum() or c in "-_") or "unknown"
        sub_dir = UPLOAD_DIR / "sub" / safe_name
        sub_dir.mkdir(parents=True, exist_ok=True)
    else:
        sub_dir = UPLOAD_DIR / image_type
    file_id = uuid.uuid4().hex[:16]
    ext = Path(file.filename or "").suffix.lower()
    if not ext:
        ext = ".mp4" if image_type == "video" else ".jpg"
    stored_name = f"{file_id}{ext}"
    file_path = sub_dir / stored_name
    with open(file_path, "wb") as f:
        f.write(content)
    base_url = get_public_base_url(request)
    if image_type == "sub":
        safe_name = "".join(c for c in product_name if c.isalnum() or c in "-_") or "unknown"
        public_url = f"{base_url}/uploads/sub/{safe_name}/{stored_name}"
    else:
        public_url = f"{base_url}/uploads/{image_type}/{stored_name}"
    if image_type == "main":
        old_url = p.get("main_image", "")
        p["main_image"] = public_url
        save_products(products)
        if old_url and old_url != public_url:
            _delete_file_by_url(old_url)
        add_log("set_main_image", {"product_name": product_name}, f"已设置「{p['name']}」的主图", True)
        return {"success": True, "message": f"已设置「{p['name']}」的主图", "url": public_url}
    elif image_type == "sub":
        subs = p.get("sub_images", [])
        if len(subs) >= 5:
            file_path.unlink(missing_ok=True)
            return {"success": False, "message": f"「{p['name']}」副图已达上限（5 张）"}
        subs.append(public_url)
        p["sub_images"] = subs
        save_products(products)
        add_log("add_sub_image", {"product_name": product_name}, f"已为「{p['name']}」添加第 {len(subs)} 张副图", True)
        return {"success": True, "message": f"已为「{p['name']}」添加第 {len(subs)} 张副图", "url": public_url}
    elif image_type == "video":
        old_url = p.get("video", "")
        p["video"] = public_url
        save_products(products)
        if old_url and old_url != public_url:
            _delete_file_by_url(old_url)
        add_log("set_video", {"product_name": product_name}, f"已设置「{p['name']}」的视频", True)
        return {"success": True, "message": f"已设置「{p['name']}」的视频", "url": public_url}
    elif image_type == "detail":
        return {"success": True, "url": public_url, "message": "图片已上传"}
    file_path.unlink(missing_ok=True)
    return {"success": False, "message": "image_type 必须是 main / sub / video / detail"}


@app.post("/api/images/delete")
async def delete_image(payload: dict):
    product_name = payload.get("product_name")
    image_type = payload.get("image_type")
    index = payload.get("index")
    p = find_product(product_name)
    if not p:
        return {"success": False, "message": f"未找到商品：{product_name}"}
    if image_type == "main":
        if not p.get("main_image"):
            return {"success": False, "message": "没有主图可删除"}
        old_url = p["main_image"]
        p["main_image"] = ""
        save_products(products)
        _delete_file_by_url(old_url)
        add_log("delete_main_image", {"product_name": product_name}, f"已删除「{p['name']}」的主图", True)
        return {"success": True, "message": f"已删除「{p['name']}」的主图"}
    elif image_type == "sub":
        subs = p.get("sub_images", [])
        if index is None or index < 0 or index >= len(subs):
            return {"success": False, "message": "副图序号无效"}
        old_url = subs.pop(index)
        p["sub_images"] = subs
        save_products(products)
        _delete_file_by_url(old_url)
        add_log("remove_sub_image", {"product_name": product_name, "index": index + 1}, f"已删除「{p['name']}」的第 {index + 1} 张副图", True)
        return {"success": True, "message": f"已删除「{p['name']}」的第 {index + 1} 张副图"}
    elif image_type == "video":
        if not p.get("video"):
            return {"success": False, "message": "没有视频可删除"}
        old_url = p["video"]
        p["video"] = ""
        save_products(products)
        _delete_file_by_url(old_url)
        add_log("delete_video", {"product_name": product_name}, f"已删除「{p['name']}」的视频", True)
        return {"success": True, "message": f"已删除「{p['name']}」的视频"}
    return {"success": False, "message": "image_type 必须是 main / sub / video"}


class DetailRequest(BaseModel):
    product_name: str
    detail_html: str


@app.post("/api/products/detail")
async def save_detail(req: DetailRequest):
    p = find_product(req.product_name)
    if not p:
        return {"success": False, "message": f"未找到商品：{req.product_name}"}
    p["detail_html"] = req.detail_html
    save_products(products)
    add_log("update_detail", {"product_name": req.product_name}, f"已保存「{p['name']}」的图文详情", True)
    return {"success": True, "message": "详情已保存"}


class ProductUpdateRequest(BaseModel):
    product_name: str
    fields: dict


@app.post("/api/products/update")
async def update_product(req: ProductUpdateRequest):
    p = find_product(req.product_name)
    if not p:
        return {"success": False, "message": f"未找到商品：{req.product_name}"}
    for key, val in req.fields.items():
        if key in ("id",): continue
        p[key] = val
    save_products(products)
    add_log("update_product", {"product_name": req.product_name}, f"已更新「{p['name']}」信息", True)
    return {"success": True, "message": f"已更新「{p['name']}」"}


class ProductCreateRequest(BaseModel):
    fields: dict


@app.post("/api/products/create")
async def create_product(req: ProductCreateRequest):
    global products
    f = req.fields
    name = f.get("name", "").strip()
    if not name:
        return {"success": False, "message": "缺少商品名称"}
    if any(p["name"] == name for p in products):
        return {"success": False, "message": f"商品「{name}」已存在"}
    new_id = max([p["id"] for p in products], default=0) + 1
    icon_map = {"茶叶": "fa-leaf", "手工艺": "fa-bag-shopping", "食品": "fa-seedling", "文创": "fa-palette"}
    new_product = {
        "id": new_id, "name": name,
        "price": f.get("price", 0), "stock": f.get("stock", 0),
        "category": f.get("category", "文创"), "platform": f.get("platform", "淘宝"),
        "status": f.get("status", "在售"), "sales": 0,
        "icon": icon_map.get(f.get("category", "文创"), "fa-box"),
        "main_image": f.get("main_image", ""), "sub_images": f.get("sub_images", []),
        "video": f.get("video", ""), "detail_html": f.get("detail_html", ""),
        "spec": f.get("spec", ""), "sku": f.get("sku", ""),
        "rating": f.get("rating", ""), "subcat": f.get("subcat", ""),
        "third": f.get("third", ""), "tags": f.get("tags", []),
        "category_path": f.get("category_path", []),
        "url": f.get("url", ""), "original_price": f.get("original_price", None),
    }
    products.append(new_product)
    save_products(products)
    add_log("create_product", {"name": name}, f"已新增「{name}」", True)
    return {"success": True, "message": f"已新增「{name}」", "product": new_product}


class ProductDeleteRequest(BaseModel):
    product_name: str


@app.post("/api/products/delete")
async def delete_product_api(req: ProductDeleteRequest):
    global products
    p = find_product(req.product_name)
    if not p:
        return {"success": False, "message": f"未找到商品：{req.product_name}"}
    name = p["name"]
    _delete_file_by_url(p.get("main_image", ""))
    _delete_file_by_url(p.get("video", ""))
    for sub in p.get("sub_images", []):
        _delete_file_by_url(sub)
    products.remove(p)
    save_products(products)
    add_log("delete_product", {"product_name": name}, f"已删除「{name}」", True)
    return {"success": True, "message": f"已删除「{name}」"}


# ============================================================
# 同步素材
# ============================================================
class SyncMaterialRequest(BaseModel):
    product_names: List[str]
    main_image: Optional[str] = None
    sub_images: Optional[List[str]] = None
    video: Optional[str] = None
    detail_html: Optional[str] = None
    mode: str = "merge"


@app.post("/api/materials/sync")
async def sync_materials(req: SyncMaterialRequest):
    global products
    if not req.product_names:
        return {"success": False, "all_ok": False, "message": "请至少选择一个商品"}
    results = []
    for name in req.product_names:
        p = find_product(name)
        if not p:
            results.append({"name": name, "ok": False, "msg": "未找到商品"})
            continue
        try:
            if req.main_image:
                old = p.get("main_image", "")
                p["main_image"] = req.main_image
                if old and old != req.main_image:
                    _delete_file_by_url(old)
            if req.sub_images:
                if req.mode == "replace":
                    for old in p.get("sub_images", []):
                        _delete_file_by_url(old)
                    p["sub_images"] = req.sub_images[:5]
                else:
                    existing = p.get("sub_images", [])
                    p["sub_images"] = (existing + req.sub_images)[:5]
            if req.video:
                old = p.get("video", "")
                p["video"] = req.video
                if old and old != req.video:
                    _delete_file_by_url(old)
            if req.detail_html:
                p["detail_html"] = req.detail_html
            results.append({"name": p["name"], "ok": True, "msg": "已同步"})
        except Exception as e:
            results.append({"name": name, "ok": False, "msg": str(e)})
    save_products(products)
    ok_count = len([r for r in results if r["ok"]])
    fail_count = len(results) - ok_count
    all_ok = (fail_count == 0)
    add_log("sync_materials", {"products": req.product_names, "mode": req.mode}, f"已同步素材到 {ok_count} 个商品（失败 {fail_count}）", all_ok)
    return {"success": True, "all_ok": all_ok, "ok_count": ok_count, "fail_count": fail_count, "message": f"同步完成：成功 {ok_count} 个，失败 {fail_count} 个", "results": results}


# ============================================================
# 知识库 API
# ============================================================
@app.get("/api/kb/list")
async def kb_list(include_disabled: bool = True):
    meta = _load_kb_meta()
    items = meta.get("items", [])
    if not include_disabled:
        items = [x for x in items if x.get("enabled", True)]
    categories = meta.get("categories", DEFAULT_KB_CATEGORIES)
    for item in items:
        item["category_name"] = _find_category_name(categories, item.get("category", ""))
    return {"success": True, "items": items, "categories": categories, "total": len(items), "stats": meta.get("stats", {})}


@app.get("/api/kb/categories")
async def kb_categories():
    meta = _load_kb_meta()
    return {"success": True, "categories": meta.get("categories", DEFAULT_KB_CATEGORIES)}


class KBCategoryCreateRequest(BaseModel):
    name: str
    parent_id: Optional[str] = None
    icon: Optional[str] = "📄"


@app.post("/api/kb/categories/create")
async def kb_category_create(req: KBCategoryCreateRequest):
    meta = _load_kb_meta()
    categories = meta.get("categories", DEFAULT_KB_CATEGORIES)
    new_id = f"cat_{uuid.uuid4().hex[:8]}"
    new_node = {"id": new_id, "name": req.name, "icon": req.icon or "📄", "parent_id": req.parent_id, "children": []}
    if not req.parent_id:
        categories.append(new_node)
    else:
        def find_and_add(tree):
            for node in tree:
                if node["id"] == req.parent_id:
                    node.setdefault("children", []).append(new_node)
                    return True
                if node.get("children") and find_and_add(node["children"]):
                    return True
            return False
        if not find_and_add(categories):
            return {"success": False, "message": "父分类不存在"}
    meta["categories"] = categories
    _save_kb_meta(meta)
    return {"success": True, "category": new_node}


class KBCategoryDeleteRequest(BaseModel):
    id: str


@app.post("/api/kb/categories/delete")
async def kb_category_delete(req: KBCategoryDeleteRequest):
    meta = _load_kb_meta()
    categories = meta.get("categories", DEFAULT_KB_CATEGORIES)
    def remove_node(tree):
        new_tree = []
        for node in tree:
            if node["id"] == req.id: continue
            if node.get("children"):
                node["children"] = remove_node(node["children"])
            new_tree.append(node)
        return new_tree
    meta["categories"] = remove_node(categories)
    _save_kb_meta(meta)
    return {"success": True}


@app.post("/api/kb/upload")
async def kb_upload(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form("cat_other"),
    tags: str = Form(""),
    content_override: str = Form("")
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in KB_ALLOWED_EXT:
        return {"success": False, "message": f"不支持的文件类型：{ext}，仅支持 txt/md/json/csv/pdf/docx"}
    content_bytes = await file.read()
    if len(content_bytes) > KB_MAX_FILE_SIZE:
        return {"success": False, "message": f"文件超过 {KB_MAX_FILE_SIZE // (1024*1024)}MB"}
    meta = _load_kb_meta()
    item_id = f"kb_{meta['next_id']:04d}"
    meta["next_id"] += 1
    safe_name = f"{item_id}{ext}"
    file_path = KB_DIR / "files" / safe_name
    with open(file_path, "wb") as f:
        f.write(content_bytes)
    if content_override.strip():
        text_content = content_override
    else:
        text_content = _extract_text_from_file(file_path, ext)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    categories = meta.get("categories", DEFAULT_KB_CATEGORIES)
    cat_name = _find_category_name(categories, category)
    item = {
        "id": item_id, "title": title, "category": category, "category_name": cat_name,
        "tags": tag_list, "filename": file.filename, "stored_name": safe_name,
        "file_size": len(content_bytes), "content_length": len(text_content),
        "content": text_content[:20000], "enabled": True, "pinned": False,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uploaded_by": "admin"
    }
    meta["items"].insert(0, item)
    _save_kb_meta(meta)
    _rebuild_kb_vectors()
    add_log("kb_upload", {"title": title}, f"已上传知识：{title}", True)
    return {"success": True, "message": f"已上传知识「{title}」", "item": item}


@app.post("/api/kb/batch-upload")
async def kb_batch_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    category: str = Form("cat_other"),
    tags: str = Form("")
):
    if len(files) > 20:
        return {"success": False, "message": "单次最多 20 个文件"}
    results = []
    for f in files:
        try:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in KB_ALLOWED_EXT:
                results.append({"filename": f.filename, "success": False, "message": f"不支持的类型 {ext}"})
                continue
            title = Path(f.filename or "未命名").stem
            single_result = await kb_upload(request, f, title, category, tags, "")
            results.append({"filename": f.filename, **single_result})
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "message": str(e)})
    ok_count = len([r for r in results if r.get("success")])
    add_log("kb_batch_upload", {"count": len(files)}, f"批量上传 {ok_count}/{len(files)} 成功", True)
    return {"success": True, "message": f"批量上传完成：成功 {ok_count}/{len(files)}", "results": results}


class KBUpdateRequest(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    enabled: Optional[bool] = None
    pinned: Optional[bool] = None


@app.post("/api/kb/update")
async def kb_update(req: KBUpdateRequest):
    meta = _load_kb_meta()
    item = next((x for x in meta["items"] if x["id"] == req.id), None)
    if not item:
        return {"success": False, "message": "知识不存在"}
    if req.title is not None: item["title"] = req.title
    if req.content is not None:
        item["content"] = req.content[:20000]
        item["content_length"] = len(req.content)
    if req.category is not None:
        item["category"] = req.category
        item["category_name"] = _find_category_name(meta.get("categories", []), req.category)
    if req.tags is not None: item["tags"] = req.tags
    if req.enabled is not None: item["enabled"] = req.enabled
    if req.pinned is not None: item["pinned"] = req.pinned
    item["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_kb_meta(meta)
    _rebuild_kb_vectors()
    add_log("kb_update", {"id": req.id}, f"已更新知识", True)
    return {"success": True, "message": "已更新", "item": item}


class KBToggleRequest(BaseModel):
    id: str
    enabled: bool


@app.post("/api/kb/toggle")
async def kb_toggle(req: KBToggleRequest):
    meta = _load_kb_meta()
    item = next((x for x in meta["items"] if x["id"] == req.id), None)
    if not item:
        return {"success": False, "message": "知识不存在"}
    item["enabled"] = req.enabled
    item["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_kb_meta(meta)
    _rebuild_kb_vectors()
    status = "已启用" if req.enabled else "已停用"
    return {"success": True, "message": f"「{item['title']}」{status}"}


class KBPinRequest(BaseModel):
    id: str
    pinned: bool


@app.post("/api/kb/pin")
async def kb_pin(req: KBPinRequest):
    meta = _load_kb_meta()
    item = next((x for x in meta["items"] if x["id"] == req.id), None)
    if not item:
        return {"success": False, "message": "知识不存在"}
    item["pinned"] = req.pinned
    _save_kb_meta(meta)
    status = "已置顶" if req.pinned else "已取消置顶"
    return {"success": True, "message": status}


@app.post("/api/kb/search")
async def kb_search(req: dict):
    keyword = req.get("keyword", "")
    top_k = int(req.get("top_k", 5))
    results = search_knowledge_hybrid(keyword, top_k)
    slim = []
    for r in results:
        item = r["item"]
        slim.append({
            "id": item["id"], "title": item["title"],
            "category": item.get("category", ""), "category_name": item.get("category_name", ""),
            "tags": item.get("tags", []), "summary": item.get("content", "")[:300],
            "score": round(r["score"], 3), "match_type": r["match_type"]
        })
    return {"success": True, "results": slim, "mode": "hybrid"}


class KBDeleteRequest(BaseModel):
    id: str


@app.post("/api/kb/delete")
async def kb_delete(req: KBDeleteRequest):
    meta = _load_kb_meta()
    item = next((x for x in meta["items"] if x["id"] == req.id), None)
    if not item:
        return {"success": False, "message": "知识不存在"}
    try:
        (KB_DIR / "files" / item["stored_name"]).unlink(missing_ok=True)
    except Exception:
        pass
    meta["items"] = [x for x in meta["items"] if x["id"] != req.id]
    if "stats" in meta and req.id in meta["stats"]:
        del meta["stats"][req.id]
    _save_kb_meta(meta)
    _rebuild_kb_vectors()
    add_log("kb_delete", {"id": req.id}, f"已删除知识：{item['title']}", True)
    return {"success": True, "message": f"已删除「{item['title']}」"}


class KBBatchDeleteRequest(BaseModel):
    ids: List[str]


@app.post("/api/kb/batch-delete")
async def kb_batch_delete(req: KBBatchDeleteRequest):
    meta = _load_kb_meta()
    deleted = 0
    for item_id in req.ids:
        item = next((x for x in meta["items"] if x["id"] == item_id), None)
        if item:
            try:
                (KB_DIR / "files" / item["stored_name"]).unlink(missing_ok=True)
            except Exception:
                pass
            deleted += 1
    meta["items"] = [x for x in meta["items"] if x["id"] not in req.ids]
    _save_kb_meta(meta)
    _rebuild_kb_vectors()
    return {"success": True, "message": f"已删除 {deleted} 条知识"}


@app.get("/api/kb/stats")
async def kb_stats():
    meta = _load_kb_meta()
    stats = meta.get("stats", {})
    items = meta.get("items", [])
    result = []
    for item in items:
        s = stats.get(item["id"], {"hit_count": 0, "last_hit": "", "satisfied": 0, "unsatisfied": 0})
        total_feedback = s.get("satisfied", 0) + s.get("unsatisfied", 0)
        satisfaction_rate = round(s.get("satisfied", 0) / total_feedback * 100, 1) if total_feedback > 0 else None
        result.append({
            "id": item["id"], "title": item["title"], "category_name": item.get("category_name", ""),
            "hit_count": s.get("hit_count", 0), "last_hit": s.get("last_hit", ""),
            "satisfied": s.get("satisfied", 0), "unsatisfied": s.get("unsatisfied", 0),
            "satisfaction_rate": satisfaction_rate
        })
    result.sort(key=lambda x: -x["hit_count"])
    return {"success": True, "stats": result}


class KBFeedbackRequest(BaseModel):
    id: str
    satisfied: bool


@app.post("/api/kb/feedback")
async def kb_feedback(req: KBFeedbackRequest):
    record_kb_feedback(req.id, req.satisfied)
    add_log("kb_feedback", {"id": req.id, "satisfied": req.satisfied}, "已记录反馈", True)
    return {"success": True}


class KBAutoGenerateRequest(BaseModel):
    session_id: Optional[str] = None
    history: Optional[List[dict]] = None
    category: Optional[str] = "cat_faq"


@app.post("/api/kb/auto-generate")
async def kb_auto_generate(req: KBAutoGenerateRequest):
    if not req.history and not req.session_id:
        return {"success": False, "message": "请提供对话历史"}
    history = req.history or []
    if req.session_id and req.session_id in sessions_store:
        s = sessions_store[req.session_id]
        history = [{"role": t.get("role"), "text": t.get("text", "")} for t in s.get("turns", [])]
    if not history:
        return {"success": False, "message": "对话历史为空"}

    history_text = "\n".join([f"{'用户' if h.get('role') == 'user' else '客服'}：{h.get('text', '')}" for h in history[-20:]])

    prompt = f"""你是知识库整理助手。以下是客服与客户的对话记录，请从中提炼出 FAQ。

对话记录：
{history_text}

请返回 JSON 数组（不要 markdown 代码块），每个元素：
{{
  "question": "客户可能问的问题",
  "answer": "标准答案（从对话中的客服回复整理）",
  "tags": ["标签1", "标签2"]
}}

要求：
- 提炼 1-5 条最有价值的 FAQ
- 问题要简洁明确，答案要完整
- 如果对话里没有可提炼的内容，返回 []
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        faqs = json.loads(raw)
        meta = _load_kb_meta()
        added = 0
        for faq in faqs:
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()
            if not q or not a: continue
            item_id = f"kb_{meta['next_id']:04d}"
            meta["next_id"] += 1
            content = f"Q: {q}\nA: {a}"
            item = {
                "id": item_id, "title": q[:60], "category": req.category or "cat_faq",
                "category_name": _find_category_name(meta.get("categories", []), req.category or "cat_faq"),
                "tags": faq.get("tags", []) + ["AI自动生成"],
                "filename": "ai_generated.txt", "stored_name": "",
                "file_size": len(content.encode("utf-8")), "content_length": len(content),
                "content": content, "enabled": True, "pinned": False,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "uploaded_by": "ai_auto"
            }
            meta["items"].insert(0, item)
            added += 1
        _save_kb_meta(meta)
        _rebuild_kb_vectors()
        add_log("kb_auto_generate", {"count": added}, f"AI 提炼 {added} 条 FAQ", True)
        return {"success": True, "message": f"已从对话中提炼 {added} 条 FAQ", "faqs": faqs, "added": added}
    except Exception as e:
        print(f"[kb_auto_generate] 失败: {e}")
        return {"success": False, "message": str(e)}


class KBRecommendRequest(BaseModel):
    text: str
    top_k: int = 3


@app.post("/api/kb/recommend")
async def kb_recommend(req: KBRecommendRequest):
    if not req.text or len(req.text) < 2:
        return {"success": True, "results": []}
    results = search_knowledge_hybrid(req.text, req.top_k)
    slim = [{
        "id": r["item"]["id"], "title": r["item"]["title"],
        "category_name": r["item"].get("category_name", ""),
        "summary": r["item"].get("content", "")[:150],
        "score": round(r["score"], 3)
    } for r in results]
    return {"success": True, "results": slim}


# ============================================================
# 资产库 API
# ============================================================
@app.get("/api/assets/list")
async def assets_list(asset_type: str = "all", tag: str = ""):
    meta = _load_assets_meta()
    items = meta.get("items", [])
    if asset_type != "all":
        items = [x for x in items if x.get("type") == asset_type]
    if tag:
        items = [x for x in items if tag in (x.get("tags") or [])]
    return {"success": True, "items": items, "total": len(items)}


@app.post("/api/assets/upload")
async def assets_upload(
    request: Request,
    file: UploadFile = File(...),
    tags: str = Form(""),
    product_name: str = Form(""),
    category: str = Form("产品图")
):
    ext = Path(file.filename or "").suffix.lower()
    if ext in ASSET_IMAGE_EXT:
        asset_type = "image"; max_size = ASSET_IMAGE_MAX; sub_dir = "images"
    elif ext in ASSET_VIDEO_EXT:
        asset_type = "video"; max_size = ASSET_VIDEO_MAX; sub_dir = "videos"
    else:
        return {"success": False, "message": f"不支持的文件类型：{ext}"}
    content = await file.read()
    if len(content) > max_size:
        return {"success": False, "message": f"文件超过 {max_size // (1024*1024)}MB"}
    meta = _load_assets_meta()
    asset_id = f"asset_{meta['next_id']:04d}"
    meta["next_id"] += 1
    safe_name = f"{asset_id}{ext}"
    file_path = ASSETS_DIR / sub_dir / safe_name
    with open(file_path, "wb") as f:
        f.write(content)
    base_url = get_public_base_url(request)
    public_url = f"{base_url}/assets/{sub_dir}/{safe_name}"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    item = {
        "id": asset_id, "type": asset_type, "name": file.filename or safe_name,
        "url": public_url, "tags": tag_list, "product_name": product_name,
        "category": category, "file_size": len(content), "stored_name": safe_name,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uploaded_by": "admin"
    }
    meta["items"].insert(0, item)
    _save_assets_meta(meta)
    add_log("asset_upload", {"name": item["name"], "type": asset_type}, f"已上传资产", True)
    return {"success": True, "message": f"已上传「{item['name']}」", "item": item}


@app.post("/api/assets/search")
async def assets_search(req: dict):
    keyword = req.get("keyword", "")
    asset_type = req.get("asset_type", "all")
    top_k = int(req.get("top_k", 6))
    results = search_assets(keyword, asset_type, top_k)
    return {"success": True, "results": results}


class AssetDeleteRequest(BaseModel):
    id: str


@app.post("/api/assets/delete")
async def assets_delete(req: AssetDeleteRequest):
    meta = _load_assets_meta()
    item = next((x for x in meta["items"] if x["id"] == req.id), None)
    if not item:
        return {"success": False, "message": "资产不存在"}
    sub_dir = "images" if item["type"] == "image" else "videos"
    try:
        (ASSETS_DIR / sub_dir / item["stored_name"]).unlink(missing_ok=True)
    except Exception:
        pass
    meta["items"] = [x for x in meta["items"] if x["id"] != req.id]
    _save_assets_meta(meta)
    add_log("asset_delete", {"id": req.id}, f"已删除资产：{item['name']}", True)
    return {"success": True, "message": f"已删除「{item['name']}」"}


class AssetUpdateRequest(BaseModel):
    id: str
    name: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


@app.post("/api/assets/update")
async def assets_update(req: AssetUpdateRequest):
    meta = _load_assets_meta()
    item = next((x for x in meta["items"] if x["id"] == req.id), None)
    if not item:
        return {"success": False, "message": "素材不存在"}
    if req.name is not None: item["name"] = req.name
    if req.product_name is not None: item["product_name"] = req.product_name
    if req.category is not None: item["category"] = req.category
    if req.tags is not None: item["tags"] = req.tags
    item["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_assets_meta(meta)
    add_log("asset_update", {"id": req.id}, f"已更新素材", True)
    return {"success": True, "message": "已保存", "item": item}


# ============================================================
# 临时图床
# ============================================================
@app.get("/temp/{file_id}")
async def serve_temp_file(file_id: str):
    _cleanup_expired()
    info = temp_meta.get(file_id)
    if not info:
        return JSONResponse({"error": "file not found or expired"}, status_code=404)
    file_path = Path(info["path"])
    if not file_path.exists():
        return JSONResponse({"error": "file missing"}, status_code=404)
    return FileResponse(path=str(file_path), media_type=info.get("content_type", "application/octet-stream"), filename=info.get("filename", file_id))


@app.post("/api/upload/temp")
async def upload_temp_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form("reference")
):
    _cleanup_expired()
    content = await file.read()
    if len(content) > TEMP_MAX_FILE_SIZE:
        return {"success": False, "message": f"文件不能超过 {TEMP_MAX_FILE_SIZE // (1024*1024)}MB"}
    ctype = file.content_type or ""
    is_image = ctype in TEMP_ALLOWED_IMAGE
    is_video = ctype in TEMP_ALLOWED_VIDEO
    if not (is_image or is_video):
        return {"success": False, "message": f"不支持的文件类型：{ctype}"}
    file_id = uuid.uuid4().hex[:16]
    ext = Path(file.filename or "").suffix.lower()
    if not ext:
        ext = ".jpg" if is_image else ".mp4"
    stored_name = f"{file_id}{ext}"
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = TEMP_UPLOAD_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)
    file_path = day_dir / stored_name
    with open(file_path, "wb") as f:
        f.write(content)
    temp_meta[file_id] = {
        "id": file_id, "path": str(file_path),
        "filename": file.filename or stored_name, "content_type": ctype,
        "size": len(content), "purpose": purpose, "created_at": time.time(),
        "is_image": is_image, "is_video": is_video
    }
    _save_temp_meta(temp_meta)
    base_url = get_public_base_url(request)
    public_url = f"{base_url}/temp/{file_id}"
    add_log("upload_temp", {"file_id": file_id, "purpose": purpose}, f"已上传临时文件 {file_id}", True)
    return {"success": True, "file_id": file_id, "url": public_url, "content_type": ctype, "size": len(content), "expires_in": TEMP_FILE_TTL}


@app.post("/api/upload/temp/batch")
async def upload_temp_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    purpose: str = Form("reference")
):
    if len(files) > 10:
        return {"success": False, "message": "单次最多上传 10 个文件"}
    results = []
    for f in files:
        single = await upload_temp_file(request, f, purpose)
        results.append(single)
    ok_count = len([r for r in results if r.get("success")])
    return {"success": True, "message": f"上传完成：成功 {ok_count}/{len(results)}", "results": results}


@app.post("/api/upload/temp/cleanup")
async def cleanup_temp_manual():
    before = len(temp_meta)
    _cleanup_expired()
    after = len(temp_meta)
    return {"success": True, "message": f"已清理 {before - after} 个过期文件"}


# ============================================================
# 视频抽帧 / 反解
# ============================================================
@app.post("/api/video/extract-frames")
async def extract_frames(
    request: Request,
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    count: int = Form(3)
):
    video_bytes = None
    if file is not None:
        video_bytes = await file.read()
    elif video_url:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.get(video_url)
                if r.status_code != 200:
                    return {"success": False, "message": f"下载视频失败：HTTP {r.status_code}"}
                video_bytes = r.content
        except Exception as e:
            return {"success": False, "message": f"下载视频失败：{e}"}
    else:
        return {"success": False, "message": "请上传视频文件或提供 video_url"}
    if len(video_bytes) > TEMP_MAX_FILE_SIZE * 2:
        return {"success": False, "message": f"视频不能超过 {TEMP_MAX_FILE_SIZE * 2 // (1024*1024)}MB"}
    frames_data = None
    try:
        import cv2
        import numpy as np
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frames_data = []
            for i in range(count):
                pos = int(total * i / max(count - 1, 1)) if count > 1 else total // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, total - 1))
                ret, frame = cap.read()
                if not ret: continue
                h, w = frame.shape[:2]
                if w > 720:
                    new_w = 720
                    new_h = int(h * 720 / w)
                    frame = cv2.resize(frame, (new_w, new_h))
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames_data.append(buf.tobytes())
            cap.release()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except ImportError:
        return {"success": False, "message": "服务端未安装 OpenCV，请在前端抽帧后再上传", "fallback": "client_side"}
    if not frames_data:
        return {"success": False, "message": "抽帧失败，未获取到有效帧"}
    base_url = get_public_base_url(request)
    frame_urls = []
    for idx, data in enumerate(frames_data):
        file_id = uuid.uuid4().hex[:16]
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = TEMP_UPLOAD_DIR / today
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / f"{file_id}.jpg"
        with open(file_path, "wb") as f:
            f.write(data)
        temp_meta[file_id] = {"id": file_id, "path": str(file_path), "filename": f"keyframe_{idx+1}.jpg", "content_type": "image/jpeg", "size": len(data), "purpose": "keyframe", "created_at": time.time(), "is_image": True, "is_video": False}
        frame_urls.append(f"{base_url}/temp/{file_id}")
    _save_temp_meta(temp_meta)
    add_log("extract_frames", {"count": len(frame_urls)}, f"已抽取 {len(frame_urls)} 帧", True)
    return {"success": True, "count": len(frame_urls), "frames": frame_urls}


class DecomposeRequest(BaseModel):
    keyframe_urls: List[str]
    product_name: Optional[str] = ""


@app.post("/api/video/decompose")
async def decompose_video(req: DecomposeRequest):
    if not req.keyframe_urls:
        return {"success": False, "message": "请至少提供 1 张关键帧"}
    system_prompt = """你是专业的电商短视频分析师。
用户会提供几帧参考视频的画面，请你反解出：
1. **画面内容**：商品是什么、场景、构图、色调、光影
2. **运镜方式**：镜头运动、景别
3. **节奏与转场**：快慢、是否有慢动作、转场方式
4. **生成 prompt**：把以上内容整合成一段 80-150 字的中文 prompt

严格输出以下 JSON 格式（不要 markdown 代码块）：
{
  "content": "画面内容描述",
  "camera": "运镜方式描述",
  "rhythm": "节奏与转场描述",
  "prompt": "可直接用于视频生成的完整 prompt",
  "style_tags": ["标签1", "标签2"]
}"""
    user_content = [{"type": "text", "text": f"目标商品：{req.product_name or '未指定'}\n请分析以下 {len(req.keyframe_urls)} 张关键帧："}]
    for url in req.keyframe_urls[:5]:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    multimodal_models = [os.getenv("AGNES_VL_MODEL", "agnes-vl-2.0"), "agnes-3.0-flash"]
    last_err = None
    for model_name in multimodal_models:
        try:
            response = agnes_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                temperature=0.3, max_tokens=800
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"content": raw, "camera": "", "rhythm": "", "prompt": raw, "style_tags": []}
            add_log("decompose_video", {"count": len(req.keyframe_urls), "model": model_name}, f"反解成功", True)
            return {"success": True, "analysis": result, "model": model_name}
        except Exception as e:
            last_err = str(e)
            continue
    add_log("decompose_video", {"count": len(req.keyframe_urls)}, f"反解失败：{last_err}", False)
    return {"success": False, "message": f"反解失败：{last_err}"}


# ============================================================
# CSV & 根路径
# ============================================================
@app.get("/api/export/csv")
async def export_csv():
    lines = ["ID,商品名称,规格,SKU,价格,库存,分类,平台,状态,销量,评价"]
    for p in products:
        lines.append(f"{p['id']},{p.get('name','')},{p.get('spec','')},{p.get('sku','')},{p['price']},{p['stock']},{p['category']},{p['platform']},{p['status']},{p['sales']},{p.get('rating','')}")
    csv_content = "\ufeff" + "\n".join(lines)
    return PlainTextResponse(content=csv_content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=products.csv"})


@app.get("/")
async def root():
    agnes_key_len = len(os.getenv("AGNES_API_KEY", "").strip())
    base_url_info = os.getenv("PUBLIC_BASE_URL", "(未设置，使用 request.base_url)")
    kb_meta = _load_kb_meta()
    asset_meta = _load_assets_meta()
    coupons = load_coupons()
    clicks = load_card_clicks()
    return {
        "status": "ok",
        "message": "AI 助手后端服务运行中（含知识库 + 资产库 + 商品卡片 + 优惠券 + 点击追踪）",
        "public_base_url": base_url_info,
        "AGNES_API_KEY_length": agnes_key_len,
        "total_products": len(products),
        "total_sessions": len(sessions_store),
        "total_knowledge": len(kb_meta.get("items", [])),
        "total_assets": len(asset_meta.get("items", [])),
        "total_coupons": len(coupons),
        "card_clicks": {
            "total_shown": clicks.get("total_shown", 0),
            "total_clicked": clicks.get("total_clicked", 0)
        },
        "features": [
            "Agnes 后端代理",
            "5 级智能商品匹配",
            "视频队列自动重试",
            "AI 意图识别",
            "AI 电商策略分析",
            "临时图床",
            "视频反解",
            "素材持久化",
            "多轮对话上下文",
            "上次生成结果记忆",
            "企业知识库",
            "向量检索",
            "引用溯源",
            "访问统计",
            "AI 自动提炼 FAQ",
            "图片/视频资产库",
            "★ 商品链接卡片",
            "★ 卡片点击追踪",
            "★ 卡片样式自定义",
            "★ 多商品卡片轮播",
            "★ 优惠券卡片",
            "★ AI 智能选品（DeepSeek）"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

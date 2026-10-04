# backend/sales_agent.py
"""
销售型 AI 客服 · 单文件实现
包含：状态机 + 上下文 + 意图识别 + KB 检索 + 商品推荐 + 成交学习
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum
import os, json, math, re, httpx
from collections import Counter

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")
LEARN_DIR = os.path.join(BASE_DIR, "data", "conversations", "successful")
KB_META = os.path.join(BASE_DIR, "knowledge_base", "_meta.json")
PRODUCTS_FILE = os.path.join(BASE_DIR, "products.json")

for d in (SESSIONS_DIR, LEARN_DIR):
    os.makedirs(d, exist_ok=True)

router = APIRouter(prefix="/api/sales", tags=["sales"])

# ============================================================
# 一、状态机
# ============================================================
class DialogStage(str, Enum):
    GREETING     = "greeting"
    EXPLORING    = "exploring"
    ANSWERING    = "answering"
    RECOMMENDING = "recommending"
    OBJECTION    = "objection"
    CLOSING      = "closing"
    CONFIRMED    = "confirmed"
    FAREWELL     = "farewell"
    NUDGING      = "nudging"

def _session_path(session_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return os.path.join(SESSIONS_DIR, f"{safe}.json")

def load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "session_id": session_id,
        "stage": DialogStage.GREETING.value,
        "history": [],
        "user_profile": {
            "budget": None,
            "preferences": [],
            "pain_points": [],
            "shown_products": [],
            "clicked_products": [],
        },
        "turn_count": 0,
        "last_user_msg_ts": None,
        "nudge_count": 0,
        "objection_count": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

def save_session(session_id: str, sess: dict):
    sess["updated_at"] = datetime.now().isoformat()
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(sess, f, ensure_ascii=False, indent=2)

def transition(sess: dict, user_text: str, intent: str) -> str:
    """状态机：根据当前状态 + 意图 → 下一状态"""
    cur = sess.get("stage", DialogStage.GREETING.value)
    t = (user_text or "").lower()

    if cur == DialogStage.FAREWELL.value:
        return cur

    # 成交信号
    if any(k in t for k in ["下单", "买了", "拍下", "付款", "怎么买", "链接",
                             "支付宝", "微信支付", "要了", "来一个", "来一件"]):
        return DialogStage.CLOSING.value

    # 已下单确认
    if cur == DialogStage.CLOSING.value and any(k in t for k in ["已下单", "已付款", "付了", "买好了", "搞定"]):
        return DialogStage.CONFIRMED.value

    # 异议
    if any(k in t for k in ["太贵", "便宜点", "优惠", "折扣", "别家", "再看看", "对比", "有点贵", "能少点"]):
        sess["objection_count"] = sess.get("objection_count", 0) + 1
        if sess["objection_count"] <= 3:
            return DialogStage.OBJECTION.value
        return DialogStage.RECOMMENDING.value

    # 购买意向
    if any(k in t for k in ["想要", "想买", "推荐", "有什么", "哪款", "多少钱",
                             "价格", "看看", "帮我选", "适合", "有没有"]):
        return DialogStage.RECOMMENDING.value

    # 咨询
    if intent == "question":
        return DialogStage.ANSWERING.value

    # 感谢 → 结束
    if any(k in t for k in ["谢谢", "感谢", "好的", "知道了", "收到"]):
        if cur in (DialogStage.CLOSING.value, DialogStage.CONFIRMED.value):
            return DialogStage.FAREWELL.value

    # 打招呼后 → 探索
    if cur == DialogStage.GREETING.value and sess.get("turn_count", 0) >= 1:
        return DialogStage.EXPLORING.value

    return cur

# ============================================================
# 二、上下文 + 画像
# ============================================================
MAX_HISTORY = 20

def build_context(sess: dict, current_user_text: str) -> List[dict]:
    history = sess.get("history", [])[-MAX_HISTORY:]
    messages = []
    for h in history:
        role = "user" if h.get("role") == "user" else "assistant"
        c = h.get("content", "")
        if c:
            messages.append({"role": role, "content": c})
    if current_user_text:
        messages.append({"role": "user", "content": current_user_text})
    return messages

def extract_profile(sess: dict, user_text: str) -> dict:
    p = sess.setdefault("user_profile", {
        "budget": None, "preferences": [], "pain_points": [],
        "shown_products": [], "clicked_products": []
    })
    t = (user_text or "").lower()

    m = re.search(r"(\d{2,6})\s*(元|块|rmb|￥|¥|块钱)", t)
    if m:
        p["budget"] = int(m.group(1))

    for kw in ["送礼", "自用", "家用", "办公室", "户外", "旅行", "生日", "节日",
               "送女友", "送男友", "送长辈", "送父母", "送朋友", "高端", "平价",
               "实惠", "颜值", "实用", "定制", "轻便"]:
        if kw in t and kw not in p["preferences"]:
            p["preferences"].append(kw)

    for kw in ["太贵", "质量差", "怕假", "不会用", "不合适", "退换", "物流慢",
               "怕坏", "怕摔", "怕过敏", "怕褪色", "怕变形"]:
        if kw in t and kw not in p["pain_points"]:
            p["pain_points"].append(kw)

    return p

# ============================================================
# 三、意图识别
# ============================================================
def classify(text: str) -> str:
    t = (text or "").lower()
    rules = [
        ("farewell", ["再见", "拜拜", "不聊了", "谢谢", "感谢"]),
        ("closing",  ["下单", "买了", "拍下", "付款", "怎么买", "链接", "要了"]),
        ("objection",["太贵", "便宜点", "优惠", "折扣", "别家", "再看看", "对比"]),
        ("buying",   ["想要", "想买", "推荐", "有什么", "哪款", "多少钱", "价格", "帮我选", "适合"]),
        ("question", ["怎么", "什么", "多少", "多久", "能不能", "有没有", "是否", "?", "？"]),
        ("smalltalk",["你好", "在吗", "hi", "hello", "在么"]),
    ]
    for intent, kws in rules:
        if any(k in t for k in kws):
            return intent
    return "smalltalk"

# ============================================================
# 四、知识库检索（TF-IDF）
# ============================================================
def _load_kb():
    if not os.path.exists(KB_META):
        return []
    try:
        with open(KB_META, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("items", [])
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _tokenize(text: str):
    text = (text or "").lower()
    tokens = list(text)
    tokens += [text[i:i+2] for i in range(len(text)-1)]
    return tokens

def _tfidf_score(query: str, doc: str) -> float:
    q = Counter(_tokenize(query))
    d = Counter(_tokenize(doc))
    if not q or not d:
        return 0.0
    common = set(q) & set(d)
    score = sum(q[t] * d[t] for t in common)
    return score / math.sqrt(len(d) + 1)

def search_kb(query: str, top_k: int = 3, min_score: float = 0.3):
    items = _load_kb()
    if not items:
        return []
    scored = []
    for it in items:
        if not it.get("enabled", True):
            continue
        text = (it.get("title","") + " " + it.get("content","") + " " +
                " ".join(it.get("tags", [])))
        sc = _tfidf_score(query, text)
        if sc >= min_score:
            scored.append((sc, it))
    scored.sort(key=lambda x: -x[0])
    return [
        {"id": it.get("id"), "title": it.get("title",""),
         "content": (it.get("content","") or "")[:300],
         "score": round(sc, 2)}
        for sc, it in scored[:top_k]
    ]

# ============================================================
# 五、商品推荐
# ============================================================
def _load_products():
    if not os.path.exists(PRODUCTS_FILE):
        return []
    try:
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("products", [])
        return data if isinstance(data, list) else []
    except Exception:
        return []

def recommend(sess: dict, user_text: str, top_k: int = 3):
    products = _load_products()
    if not products:
        return []

    p = sess.get("user_profile", {})
    shown = set(p.get("shown_products", []))
    t = (user_text or "").lower()

    candidates = []
    for prod in products:
        pid = prod.get("id")
        if pid in shown:
            continue
        if prod.get("stock", 1) <= 0:
            continue

        score = 0.0
        name = (str(prod.get("name", "")) + " " +
                " ".join(prod.get("tags", []) or [])).lower()

        for kw in t.replace("，", " ").replace(",", " ").split():
            if len(kw) >= 2 and kw in name:
                score += 2.0
        for pref in p.get("preferences", []):
            if pref in name:
                score += 3.0

        budget = p.get("budget")
        if budget:
            price = float(prod.get("price", 0) or 0)
            if price <= budget:
                score += 1.0
            if price > budget * 1.2:
                score -= 2.0

        score += min(float(prod.get("sales", 0) or 0) / 1000.0, 2.0)
        if score > 0:
            candidates.append((score, prod))

    candidates.sort(key=lambda x: -x[0])
    picked = [prod for _, prod in candidates[:top_k]]

    if not picked:
        sorted_by_sales = sorted(products, key=lambda x: -float(x.get("sales", 0) or 0))
        picked = [x for x in sorted_by_sales if x.get("id") not in shown][:top_k]

    for prod in picked:
        if prod.get("id") is not None:
            shown.add(prod.get("id"))
    p["shown_products"] = list(shown)
    sess["user_profile"] = p

    return [
        {"id": x.get("id"), "name": x.get("name",""),
         "price": x.get("price", 0),
         "original_price": x.get("original_price"),
         "main_image": x.get("main_image",""),
         "sales": x.get("sales", 0), "stock": x.get("stock", 0),
         "platform": x.get("platform",""), "spec": x.get("spec",""),
         "rating": x.get("rating",""), "tags": x.get("tags", []),
         "url": x.get("url",""), "sku": x.get("sku","")}
        for x in picked
    ]

# ============================================================
# 六、成交学习
# ============================================================
def get_few_shot_examples(stage: str, limit: int = 2):
    if not os.path.exists(LEARN_DIR):
        return []
    files = [f for f in os.listdir(LEARN_DIR) if f.endswith(".json")]
    files.sort()
    files = files[-30:]
    picked, seen = [], set()
    for fn in files:
        try:
            with open(os.path.join(LEARN_DIR, fn), "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        for turn in s.get("turns", []):
            if turn.get("stage") != stage:
                continue
            u = (turn.get("user_text") or "").strip()
            a = (turn.get("agent_reply") or "").strip()
            if not u or not a or (u, a) in seen:
                continue
            seen.add((u, a))
            picked.append({"user": u, "agent": a})
            if len(picked) >= limit:
                return picked
    return picked

def save_successful_conversation(session_id: str, sess: dict):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    sample = {
        "session_id": session_id,
        "outcome": "success",
        "user_profile": sess.get("user_profile", {}),
        "created_at": sess.get("created_at"),
        "turns": [],
    }
    history = sess.get("history", [])
    for i in range(len(history) - 1):
        u, a = history[i], history[i+1]
        if u.get("role") == "user" and a.get("role") == "assistant":
            sample["turns"].append({
                "user_text": u.get("content",""),
                "agent_reply": a.get("content",""),
                "stage": u.get("stage", "unknown"),
            })
    path = os.path.join(LEARN_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

# ============================================================
# 七、Prompt 模板
# ============================================================
BASE_ROLE = """你是「云山优选」的金牌销售客服小云。
你的唯一目标：在真诚帮助客户的前提下，引导客户完成下单。

【铁律】
1. 永远先解决客户疑虑，再推荐商品。不要强推。
2. 每次回复控制在 3 句话以内，口语化，不要长篇大论。
3. 已经推过的商品不要重复推（见"已推商品"列表）。
4. 只有当客户明确表达需求/预算/场景时，才推商品。
5. 客户问政策/参数时，只答问题，不推销。
6. 当客户表达购买意向，立刻给商品卡片（系统会自动附）。
7. 客户下单后，确认订单 + 感谢 + 结束对话，不再推销。
8. 不说"亲亲""宝贝"等过度亲昵词，用"您"。
9. 不要自己写商品价格、链接，系统会自动附加卡片。
"""

STAGE_PROMPTS = {
    "greeting": BASE_ROLE + "\n【阶段：打招呼】友好简短问候，主动询问客户想找什么。",
    "exploring": BASE_ROLE + "\n【阶段：探索需求】不急着推商品，先问清用途/预算/送人还是自用，一次只问 1 个问题。",
    "answering": BASE_ROLE + "\n【阶段：回答问题】只答客户问的问题，参考知识库作答。回答完可轻带一句'有需要可帮您推荐'，但不强推。",
    "recommending": BASE_ROLE + "\n【阶段：推荐商品】结合画像（预算/偏好）推荐 1-3 款，一句话点出'为什么适合您'。系统会自动附卡片。",
    "objection": BASE_ROLE + "\n【阶段：处理异议】先共情，再给价值，再给方案。可提'满减'，但不要立刻降价。",
    "closing": BASE_ROLE + "\n【阶段：引导下单】立刻给明确行动指令：'点击购物车或立即购买就可以啦'。系统会自动附卡片。",
    "confirmed": BASE_ROLE + "\n【阶段：订单确认】感谢 + 告知 24h 发货 + 7 天无理由退换。",
    "farewell": BASE_ROLE + "\n【阶段：礼貌结束】客户已下单/已感谢。简短收尾，不要再提任何商品，回复不超过 1 句话。",
    "nudging": BASE_ROLE + "\n【阶段：沉默跟进】客户 3 分钟没回。主动轻推一下，要带价值（如'库存不多了'），不要问'还在吗'。",
}

def get_system_prompt(stage: str, sess: dict, kb_hits: list = None) -> str:
    base = STAGE_PROMPTS.get(stage, BASE_ROLE)
    p = sess.get("user_profile", {})
    lines = []
    if p.get("budget"):
        lines.append(f"- 预算：约 {p['budget']} 元")
    if p.get("preferences"):
        lines.append(f"- 偏好：{'、'.join(p['preferences'])}")
    if p.get("pain_points"):
        lines.append(f"- 顾虑：{'、'.join(p['pain_points'])}")
    if p.get("shown_products"):
        lines.append(f"- 已推商品 id：{p['shown_products']}（不要重复推）")
    if lines:
        base += "\n【用户画像】\n" + "\n".join(lines)
    if kb_hits:
        base += "\n【知识库参考】\n"
        for i, k in enumerate(kb_hits[:3], 1):
            base += f"{i}. {k.get('title','')}：{k.get('content','')}\n"
    return base

# ============================================================
# 八、DeepSeek 调用
# ============================================================
async def _call_deepseek(messages) -> str:
    if not DEEPSEEK_API_KEY:
        return "（未配置 DEEPSEEK_API_KEY）我先帮您看看，稍等～"
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 300,
                },
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[sales] DeepSeek 调用失败:", e)
        return "抱歉，我这边网络有点问题，稍后再为您服务～"

# ============================================================
# 九、对外 API
# ============================================================
class ChatIn(BaseModel):
    session_id: str
    text: str
    conv_id: Optional[str] = None

class ProductOut(BaseModel):
    id: Optional[object] = None
    name: str = ""
    price: float = 0
    original_price: Optional[float] = None
    main_image: str = ""
    sales: int = 0
    stock: int = 0
    platform: str = ""
    spec: str = ""
    rating: str = ""
    tags: List[str] = []
    url: str = ""
    sku: str = ""

class ChatOut(BaseModel):
    type: str = "chat"
    text: str = ""
    stage: str = "greeting"
    products: List[ProductOut] = []
    kb_hits: List[dict] = []
    farewell: bool = False

@router.post("/chat", response_model=ChatOut)
async def sales_chat(inp: ChatIn):
    sess = load_session(inp.session_id)
    sess["turn_count"] = sess.get("turn_count", 0) + 1
    sess["last_user_msg_ts"] = datetime.now().isoformat()

    extract_profile(sess, inp.text)
    intent = classify(inp.text)
    new_stage = transition(sess, inp.text, intent)
    old_stage = sess.get("stage")
    sess["stage"] = new_stage

    kb_hits = []
    if intent in ("question", "objection"):
        kb_hits = search_kb(inp.text, top_k=3)

    products_raw = []
    if new_stage in (DialogStage.RECOMMENDING.value, DialogStage.CLOSING.value):
        products_raw = recommend(sess, inp.text, top_k=3)

    system_prompt = get_system_prompt(new_stage, sess, kb_hits)

    try:
        for ex in get_few_shot_examples(new_stage, limit=2):
            system_prompt += f"\n【优秀示例】\n客户：{ex['user']}\n客服：{ex['agent']}\n"
    except Exception as e:
        print("[sales] few-shot 加载失败:", e)

    messages = [{"role": "system", "content": system_prompt}]
    messages += build_context(sess, inp.text)

    reply_text = await _call_deepseek(messages)

    sess["history"].append({
        "role": "user", "content": inp.text,
        "ts": datetime.now().isoformat(), "stage": new_stage,
    })
    sess["history"].append({
        "role": "assistant", "content": reply_text,
        "ts": datetime.now().isoformat(), "stage": new_stage,
    })

    if new_stage == DialogStage.FAREWELL.value and old_stage != DialogStage.FAREWELL.value:
        try:
            save_successful_conversation(inp.session_id, sess)
        except Exception as e:
            print("[sales] 保存样本失败:", e)

    save_session(inp.session_id, sess)

    products_out = []
    for p in products_raw:
        try:
            products_out.append(ProductOut(**{k: v for k, v in p.items() if k in ProductOut.__fields__}))
        except Exception:
            pass

    return ChatOut(
        type="chat", text=reply_text, stage=new_stage,
        products=products_out, kb_hits=kb_hits,
        farewell=(new_stage == DialogStage.FAREWELL.value),
    )

@router.post("/nudge")
async def nudge(session_id: str):
    """沉默跟进"""
    sess = load_session(session_id)
    if sess.get("stage") in (DialogStage.FAREWELL.value, DialogStage.CONFIRMED.value):
        return {"skip": True, "reason": "会话已结束"}
    if sess.get("nudge_count", 0) >= 2:
        return {"skip": True, "reason": "已跟进多次"}

    sess["nudge_count"] = sess.get("nudge_count", 0) + 1
    sess["stage"] = DialogStage.NUDGING.value

    system_prompt = get_system_prompt(DialogStage.NUDGING.value, sess, None)
    messages = [{"role": "system", "content": system_prompt}]
    messages += build_context(sess, "")

    reply_text = await _call_deepseek(messages)
    sess["history"].append({
        "role": "assistant", "content": reply_text,
        "ts": datetime.now().isoformat(), "stage": DialogStage.NUDGING.value,
    })
    save_session(session_id, sess)
    return {"skip": False, "text": reply_text}

# ============================================================
# 十、可选：用户行为追踪（点击商品）
# ============================================================
class TrackIn(BaseModel):
    session_id: str
    product_id: object
    action: str = "view"

@router.post("/track")
async def track(inp: TrackIn):
    sess = load_session(inp.session_id)
    p = sess.setdefault("user_profile", {})
    clicked = p.setdefault("clicked_products", [])
    if inp.product_id not in clicked:
        clicked.append(inp.product_id)
    save_session(inp.session_id, sess)
    return {"ok": True}

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import base64
import uuid
import time
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List
from contextlib import asynccontextmanager

load_dotenv()

# ============================================================
# 临时图床配置
# ============================================================
TEMP_UPLOAD_DIR = Path(os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")).resolve()
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_FILE_TTL = int(os.getenv("TEMP_FILE_TTL", 3600))
TEMP_MAX_FILE_SIZE = 20 * 1024 * 1024
TEMP_ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp", "image/gif"}
TEMP_ALLOWED_VIDEO = {"video/mp4", "video/webm", "video/quicktime"}

TEMP_META_FILE = TEMP_UPLOAD_DIR / "_meta.json"

# ============================================================
# 商品素材持久化目录
# ============================================================
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

for _sub in ("main", "sub", "video", "detail"):
    (UPLOAD_DIR / _sub).mkdir(parents=True, exist_ok=True)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载商品素材静态目录
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# DeepSeek（文本）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Agnes（多模态 + 视频/图片生成）
agnes_client = OpenAI(
    api_key=os.getenv("AGNES_API_KEY", "sk-XmLjN9e3Mhh8wf"),
    base_url="https://apihub.agnes-ai.com/v1"
)

DATA_FILE = "products.json"
LOG_FILE = "logs.json"
TAG_FILE = "tags.json"
LOG_MAX = 1000

# ============================================================
# 平台级类目树
# ============================================================
CATEGORY_TREE = [
    {
        "id": "cat_food", "name": "食品", "icon": "🍜",
        "children": [
            {"id": "cat_food_nuts", "name": "坚果", "children": [
                {"id": "cat_food_nuts_pistachio", "name": "开心果"},
                {"id": "cat_food_nuts_walnut", "name": "核桃"},
                {"id": "cat_food_nuts_almond", "name": "巴旦木"},
            ]},
            {"id": "cat_food_tea", "name": "茶饮", "children": [
                {"id": "cat_food_tea_green", "name": "绿茶"},
                {"id": "cat_food_tea_black", "name": "红茶"},
                {"id": "cat_food_tea_oolong", "name": "乌龙茶"},
            ]},
            {"id": "cat_food_snack", "name": "零食", "children": [
                {"id": "cat_food_snack_candy", "name": "糖果"},
                {"id": "cat_food_snack_dried", "name": "果干"},
                {"id": "cat_food_snack_meat", "name": "肉干"},
            ]},
            {"id": "cat_food_health", "name": "滋补", "children": [
                {"id": "cat_food_health_tea", "name": "养生茶"},
                {"id": "cat_food_health_soup", "name": "汤料"},
            ]},
        ]
    },
    {
        "id": "cat_handicraft", "name": "手工艺", "icon": "🎨",
        "children": [
            {"id": "cat_handicraft_weave", "name": "编织", "children": [
                {"id": "cat_handicraft_weave_bamboo", "name": "竹编"},
                {"id": "cat_handicraft_weave_rattan", "name": "藤编"},
                {"id": "cat_handicraft_weave_cloth", "name": "布艺"},
            ]},
            {"id": "cat_handicraft_ceramic", "name": "陶瓷", "children": [
                {"id": "cat_handicraft_ceramic_cup", "name": "陶瓷杯"},
                {"id": "cat_handicraft_ceramic_plate", "name": "陶瓷盘"},
                {"id": "cat_handicraft_ceramic_teapot", "name": "陶瓷壶"},
            ]},
            {"id": "cat_handicraft_wood", "name": "木艺", "children": [
                {"id": "cat_handicraft_wood_box", "name": "木盒"},
                {"id": "cat_handicraft_wood_toy", "name": "木制玩具"},
            ]},
        ]
    },
    {
        "id": "cat_tea", "name": "茶叶", "icon": "🍃",
        "children": [
            {"id": "cat_tea_green", "name": "绿茶", "children": [
                {"id": "cat_tea_green_longjing", "name": "龙井"},
                {"id": "cat_tea_green_biluochun", "name": "碧螺春"},
            ]},
            {"id": "cat_tea_black", "name": "红茶", "children": [
                {"id": "cat_tea_black_junshan", "name": "君山银针"},
                {"id": "cat_tea_black_keemun", "name": "祁门红茶"},
            ]},
            {"id": "cat_tea_oolong", "name": "乌龙茶", "children": [
                {"id": "cat_tea_oolong_tieguanyin", "name": "铁观音"},
                {"id": "cat_tea_oolong_dahongpao", "name": "大红袍"},
            ]},
        ]
    },
    {
        "id": "cat_cultural", "name": "文创", "icon": "📚",
        "children": [
            {"id": "cat_cultural_paper", "name": "纸艺", "children": [
                {"id": "cat_cultural_paper_fan", "name": "折扇"},
                {"id": "cat_cultural_paper_card", "name": "明信片"},
                {"id": "cat_cultural_paper_bookmark", "name": "书签"},
            ]},
            {"id": "cat_cultural_fabric", "name": "布艺", "children": [
                {"id": "cat_cultural_fabric_scarf", "name": "丝巾"},
                {"id": "cat_cultural_fabric_bag", "name": "布包"},
            ]},
            {"id": "cat_cultural_soap", "name": "香道", "children": [
                {"id": "cat_cultural_soap_handmade", "name": "手工皂"},
                {"id": "cat_cultural_soap_scent", "name": "香薰"},
            ]},
        ]
    },
]

CATEGORY_MAP = {}


def build_category_map(tree, parent_path=None):
    if parent_path is None:
        parent_path = []
    for node in tree:
        node_id = node["id"]
        path = parent_path + [{"id": node_id, "name": node["name"]}]
        CATEGORY_MAP[node_id] = {
            "name": node["name"],
            "path": path,
            "level": len(path),
            "parent_id": parent_path[-1]["id"] if parent_path else None,
        }
        if "children" in node:
            build_category_map(node["children"], path)


build_category_map(CATEGORY_TREE)


def find_category_node(node_id, tree=None):
    if tree is None:
        tree = CATEGORY_TREE
    for node in tree:
        if node["id"] == node_id:
            return node
        if "children" in node:
            found = find_category_node(node_id, node["children"])
            if found:
                return found
    return None


all_categories_flat = list(CATEGORY_MAP.keys())

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
                return data
        except Exception:
            return DEFAULT_PRODUCTS
    return DEFAULT_PRODUCTS


def save_products(products):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)


products = load_products()

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
        "market_analysis": "市场趋势分析",
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


# ============================================================
# 根据 URL 反查并删除文件
# ============================================================
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
# AI 工具（33 个）
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


class ChatRequest(BaseModel):
    text: str


@app.post("/api/ai/parse")
async def parse_intent(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {"role": "system", "content": (
                    "你是电商运营助手。用户会用自然语言下达指令，"
                    "你需要判断应该调用哪个工具，并提取参数。"
                    "如果用户只是闲聊或问问题，不要调用工具，直接回复。"
                )},
                {"role": "user", "content": req.text}
            ],
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return {"type": call.function.name, "args": json.loads(call.function.arguments)}
        else:
            return {"type": "chat", "text": msg.content}
    except Exception as e:
        return {"type": "error", "text": str(e)}


# ============================================================
# AI 生成意图识别
# ============================================================
@app.post("/api/ai/parse-generation")
async def parse_generation(req: dict):
    text = req.get('text', '')
    available_products = req.get('available_products', [])
    has_reference = req.get('has_reference', False)
    ref_types = req.get('ref_types', [])

    if not text:
        return {"kind": None, "is_bound": False, "product_name": ""}

    products_str = '、'.join(available_products) if available_products else '（暂无）'
    ref_str = f'有（{"/".join(ref_types)}）' if has_reference else '无'

    prompt = f"""你是电商素材生成助手。请分析用户输入，判断生成意图。

用户输入：{text}
有无参考素材：{ref_str}
可用商品列表：{products_str}

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

示例：
- "图片+文字参考图生成视频" → {{"kind": "video", "is_bound": false, "product_name": "", "seconds": "5", "free_theme": "", "ai_suggestion": "基于参考图生成视频"}}
- "生成一张茶叶促销海报" → {{"kind": "poster", "is_bound": false, "product_name": "", "poster_type": "促销海报", "free_theme": "茶叶"}}
- "生成3张主图" → {{"kind": "images", "is_bound": false, "product_name": "", "count": 3}}
- "给云山茶叶礼盒生成3张主图" → {{"kind": "images", "is_bound": true, "product_name": "云山茶叶礼盒", "count": 3}}
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
        return {
            "kind": None,
            "is_bound": False,
            "product_name": "",
            "error": str(e)
        }


# ============================================================
# ★ 新增 1：商品智能匹配（三级匹配）
# ============================================================
class MatchRequest(BaseModel):
    keywords: List[str]
    generation_context: Optional[str] = ""


@app.post("/api/products/match")
async def match_products(req: MatchRequest):
    """
    三级匹配：
    L1 精确匹配 → score 1.0
    L2 语义匹配（AI）→ score 0.5~0.99
    L3 分类/标签匹配 → score 0.5~0.7
    """
    if not products or not req.keywords:
        return {"success": True, "matches": []}

    matches = []
    seen = set()

    # ---------- L1: 精确匹配 ----------
    for kw in req.keywords:
        clean = kw.replace(" ", "").strip().lower()
        for p in products:
            pn = p["name"].replace(" ", "").lower()
            if pn == clean:
                key = (p["name"], "exact")
                if key not in seen:
                    seen.add(key)
                    matches.append({
                        "product_name": p["name"],
                        "match_level": "exact",
                        "score": 1.0,
                        "reason": f"精确匹配「{kw}」"
                    })

    if matches:
        add_log("match_products", {"keywords": req.keywords}, f"L1 精确匹配 {len(matches)} 个", True)
        return {"success": True, "matches": matches}

    # ---------- L2: 语义匹配（AI） ----------
    products_summary = [
        {
            "name": p["name"],
            "category": p.get("category", ""),
            "subcat": p.get("subcat", ""),
            "tags": p.get("tags", []),
            "spec": p.get("spec", ""),
            "price": p.get("price", 0)
        }
        for p in products
    ]

    try:
        prompt = f"""你是电商商品匹配助手。

用户输入：{', '.join(req.keywords)}
生成内容类型：{req.generation_context or '图片'}

可选商品列表：
{json.dumps(products_summary, ensure_ascii=False, indent=2)}

请判断用户输入最可能对应哪个（或哪些）商品。考虑：
1. 商品名称的语义相关性
2. 商品的分类（如"茶叶"类商品与用户说的"茶叶"相关）
3. 商品的子分类（如"龙井"与子类"龙井"相关）
4. 商品的标签（如标签"送礼"与用户说的"送礼"相关）
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
- score 1.0 = 完全等价，0.8 = 高度相关，0.6 = 中等相关，0.5 = 弱相关
- 如果都不相关，返回 {{"matches": []}}
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
                    matches.append({
                        "product_name": m["product_name"],
                        "match_level": "semantic",
                        "score": float(m.get("score", 0.6)),
                        "reason": m.get("reason", "语义相关")
                    })
    except Exception as e:
        print(f"[match_products] AI 语义匹配失败: {e}")

    # ---------- L3: 分类/标签匹配（兜底） ----------
    if not matches:
        for kw in req.keywords:
            clean = kw.replace(" ", "").lower()
            for p in products:
                p_category = p.get("category", "").lower()
                p_subcat = p.get("subcat", "").lower()
                p_tags = [t.lower() for t in p.get("tags", [])]

                hit = False
                if clean in p_category or clean in p_subcat:
                    hit = True
                if any(clean in t for t in p_tags):
                    hit = True
                if clean in p["name"].lower():
                    hit = True

                if hit:
                    key = (p["name"], "category")
                    if key not in seen:
                        seen.add(key)
                        matches.append({
                            "product_name": p["name"],
                            "match_level": "category",
                            "score": 0.6,
                            "reason": f"命中分类/标签"
                        })

    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:5]

    add_log("match_products", {"keywords": req.keywords}, f"共匹配 {len(matches)} 个商品", True)
    return {"success": True, "matches": matches}


# ============================================================
# ★ 新增 2：AI 分析市场环境（自由生成时用）
# ============================================================
class MarketAnalysisRequest(BaseModel):
    theme: str
    kind: str
    platform: Optional[str] = "全平台"


@app.post("/api/ai/market-analysis")
async def market_analysis(req: MarketAnalysisRequest):
    """
    分析当下市场环境，为自由生成提供 prompt 增强建议。
    """
    if not req.theme:
        return {"success": False, "message": "缺少主题"}

    now = datetime.now()
    month = now.month
    if month in (3, 4, 5):
        season_hint = "春季"
    elif month in (6, 7, 8):
        season_hint = "夏季"
    elif month in (9, 10, 11):
        season_hint = "秋季"
    else:
        season_hint = "冬季"

    kind_names = {"images": "商品主图", "video": "短视频", "poster": "宣传海报"}
    kind_name = kind_names.get(req.kind, "商品素材")

    prompt = f"""你是资深电商视觉营销专家。用户想为「{req.theme}」生成{kind_name}。
当前时间：{now.strftime('%Y年%m月%d日')}（{season_hint}）
目标平台：{req.platform}

请分析当下市场环境，给出：
1. 这个主题在**当前季节/时间**最受欢迎的**视觉风格**（如国潮、极简、暖色调、新中式等）
2. 该主题在**当下电商平台**的**主流消费场景**（如送礼、自用、囤货、换季）
3. 应该突出的**核心卖点**（3个以内）
4. 应该避免的**过时元素**（1-2个）

严格返回 JSON（不要 markdown 代码块）：
{{
  "trend_style": "当下最流行的视觉风格关键词（20字内）",
  "scenario": "主流消费场景（15字内）",
  "selling_points": ["卖点1", "卖点2", "卖点3"],
  "avoid": ["避免1", "避免2"],
  "prompt_enhancement": "一段 60-100 字的 prompt 增强描述",
  "season_tag": "{season_hint}"
}}
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

        result = json.loads(raw)
        add_log("market_analysis", {"theme": req.theme, "kind": req.kind}, "分析成功", True)
        return {"success": True, "analysis": result}
    except Exception as e:
        print(f"[market_analysis] 失败: {e}")
        return {
            "success": True,
            "analysis": {
                "trend_style": "现代简约，暖色调",
                "scenario": "自用/送礼",
                "selling_points": ["品质", "设计", "性价比"],
                "avoid": ["过时的浓重滤镜"],
                "prompt_enhancement": "modern minimalist style, warm tone, high-end commercial photography",
                "season_tag": season_hint
            }
        }


# ============================================================
# 商品数据接口
# ============================================================
@app.get("/api/products")
async def get_products():
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
        new_product = {"id": new_id, "name": name, "price": price, "stock": stock, "category": category, "platform": platform, "status": "在售", "sales": 0, "icon": icon_map.get(category, "fa-box"), "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": spec, "sku": sku, "rating": "", "subcat": "", "third": "", "tags": [], "category_path": []}
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

    base_url = str(request.base_url).rstrip("/")
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
        if key in ("id",):
            continue
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
        "price": f.get("price", 0),
        "stock": f.get("stock", 0),
        "category": f.get("category", "文创"),
        "platform": f.get("platform", "淘宝"),
        "status": f.get("status", "在售"),
        "sales": 0,
        "icon": icon_map.get(f.get("category", "文创"), "fa-box"),
        "main_image": f.get("main_image", ""),
        "sub_images": f.get("sub_images", []),
        "video": f.get("video", ""),
        "detail_html": f.get("detail_html", ""),
        "spec": f.get("spec", ""),
        "sku": f.get("sku", ""),
        "rating": f.get("rating", ""),
        "subcat": f.get("subcat", ""),
        "third": f.get("third", ""),
        "tags": f.get("tags", []),
        "category_path": f.get("category_path", []),
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
# 批量同步素材到商品
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

    add_log(
        "sync_materials",
        {"products": req.product_names, "mode": req.mode},
        f"已同步素材到 {ok_count} 个商品（失败 {fail_count}）",
        all_ok
    )
    return {
        "success": True,
        "all_ok": all_ok,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "message": f"同步完成：成功 {ok_count} 个，失败 {fail_count} 个",
        "results": results
    }


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
    return FileResponse(
        path=str(file_path),
        media_type=info.get("content_type", "application/octet-stream"),
        filename=info.get("filename", file_id)
    )


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
        "id": file_id,
        "path": str(file_path),
        "filename": file.filename or stored_name,
        "content_type": ctype,
        "size": len(content),
        "purpose": purpose,
        "created_at": time.time(),
        "is_image": is_image,
        "is_video": is_video
    }
    _save_temp_meta(temp_meta)

    base_url = str(request.base_url).rstrip("/")
    public_url = f"{base_url}/temp/{file_id}"

    add_log("upload_temp", {"file_id": file_id, "purpose": purpose}, f"已上传临时文件 {file_id}", True)

    return {
        "success": True,
        "file_id": file_id,
        "url": public_url,
        "content_type": ctype,
        "size": len(content),
        "expires_in": TEMP_FILE_TTL
    }


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
    return {
        "success": True,
        "message": f"上传完成：成功 {ok_count}/{len(results)}",
        "results": results
    }


@app.post("/api/upload/temp/cleanup")
async def cleanup_temp_manual():
    before = len(temp_meta)
    _cleanup_expired()
    after = len(temp_meta)
    return {"success": True, "message": f"已清理 {before - after} 个过期文件"}


# ============================================================
# 视频抽帧（服务端，OpenCV 可选）
# ============================================================
class ExtractFramesRequest(BaseModel):
    video_url: Optional[str] = None
    count: int = 3


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
        import httpx
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
                if not ret:
                    continue
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
        return {
            "success": False,
            "message": "服务端未安装 OpenCV，请在前端抽帧后再上传",
            "fallback": "client_side"
        }

    if not frames_data:
        return {"success": False, "message": "抽帧失败，未获取到有效帧"}

    base_url = str(request.base_url).rstrip("/")
    frame_urls = []
    for idx, data in enumerate(frames_data):
        file_id = uuid.uuid4().hex[:16]
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = TEMP_UPLOAD_DIR / today
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / f"{file_id}.jpg"
        with open(file_path, "wb") as f:
            f.write(data)

        temp_meta[file_id] = {
            "id": file_id,
            "path": str(file_path),
            "filename": f"keyframe_{idx+1}.jpg",
            "content_type": "image/jpeg",
            "size": len(data),
            "purpose": "keyframe",
            "created_at": time.time(),
            "is_image": True,
            "is_video": False
        }
        frame_urls.append(f"{base_url}/temp/{file_id}")
    _save_temp_meta(temp_meta)

    add_log("extract_frames", {"count": len(frame_urls)}, f"已抽取 {len(frame_urls)} 帧", True)

    return {
        "success": True,
        "count": len(frame_urls),
        "frames": frame_urls
    }


# ============================================================
# 多模态反解
# ============================================================
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
2. **运镜方式**：镜头运动（推/拉/摇/移/跟/升降/环绕）、景别（特写/中景/全景）
3. **节奏与转场**：快慢、是否有慢动作、转场方式
4. **生成 prompt**：把以上内容整合成一段 80-150 字的中文 prompt，可直接用于 AI 视频生成

严格输出以下 JSON 格式（不要 markdown 代码块）：
{
  "content": "画面内容描述",
  "camera": "运镜方式描述",
  "rhythm": "节奏与转场描述",
  "prompt": "可直接用于视频生成的完整 prompt",
  "style_tags": ["标签1", "标签2"]
}"""

    user_content = [
        {"type": "text", "text": f"目标商品：{req.product_name or '未指定'}\n请分析以下 {len(req.keyframe_urls)} 张关键帧："}
    ]
    for url in req.keyframe_urls[:5]:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    multimodal_models = [
        os.getenv("AGNES_VL_MODEL", "agnes-vl-2.0"),
        "agnes-3.0-flash",
    ]
    last_err = None
    for model_name in multimodal_models:
        try:
            response = agnes_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=800
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
                result = {
                    "content": raw,
                    "camera": "",
                    "rhythm": "",
                    "prompt": raw,
                    "style_tags": []
                }

            add_log("decompose_video", {"count": len(req.keyframe_urls), "model": model_name},
                    f"反解成功", True)
            return {"success": True, "analysis": result, "model": model_name}

        except Exception as e:
            last_err = str(e)
            print(f"[decompose] model {model_name} 失败: {e}")
            continue

    add_log("decompose_video", {"count": len(req.keyframe_urls)},
            f"反解失败：{last_err}", False)
    return {"success": False, "message": f"反解失败：{last_err}"}


# ============================================================
# CSV 导出 & 根路径
# ============================================================
@app.get("/api/export/csv")
async def export_csv():
    lines = ["ID,商品名称,规格,SKU,价格,库存,分类,平台,状态,销量,评价"]
    for p in products:
        lines.append(f"{p['id']},{p.get('name','')},{p.get('spec','')},{p.get('sku','')},{p['price']},{p['stock']},{p['category']},{p['platform']},{p['status']},{p['sales']},{p.get('rating','')}")
    csv_content = "\ufeff" + "\n".join(lines)
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=products.csv"}
    )


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI 助手后端服务运行中（含 AI 意图识别 + 智能商品匹配 + 市场分析 + 临时图床 + 视频反解 + 素材持久化）"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

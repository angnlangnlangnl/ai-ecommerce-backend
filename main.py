from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

DATA_FILE = "products.json"
LOG_FILE = "logs.json"
LOG_MAX = 1000

DEFAULT_PRODUCTS = [
    {"id": 1, "name": "云山茶叶礼盒", "price": 128, "stock": 234, "category": "茶叶", "platform": "抖音", "status": "在售", "sales": 1247, "icon": "fa-leaf", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "500g×2", "sku": "YS-2026-001", "rating": "4.9星", "subcat": "茶饮", "third": "礼盒装"},
    {"id": 2, "name": "手工竹编包", "price": 89, "stock": 247, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 856, "icon": "fa-bag-shopping", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "35×28cm", "sku": "ZB-2026-002", "rating": "4.8星", "subcat": "编织", "third": "手工"},
    {"id": 3, "name": "山核桃仁 250g", "price": 45, "stock": 156, "category": "食品", "platform": "拼多多", "status": "在售", "sales": 2345, "icon": "fa-seedling", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "250g/袋", "sku": "HT-2026-003", "rating": "4.7星", "subcat": "坚果", "third": "散装"},
    {"id": 4, "name": "云山手工皂套装", "price": 79, "stock": 89, "category": "文创", "platform": "京东", "status": "在售", "sales": 567, "icon": "fa-soap", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "3块装", "sku": "SZ-2026-004", "rating": "4.5星", "subcat": "纸艺", "third": "定制"},
    {"id": 5, "name": "云山陶瓷杯", "price": 58, "stock": 143, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 1876, "icon": "fa-mug-saucer", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "350ml", "sku": "TC-2026-005", "rating": "4.9星", "subcat": "陶瓷", "third": "手工"},
    {"id": 6, "name": "手写书法折扇", "price": 35, "stock": 0, "category": "文创", "platform": "抖音", "status": "下架", "sales": 234, "icon": "fa-scroll", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "10寸", "sku": "FS-2026-006", "rating": "4.6星", "subcat": "纸艺", "third": "定制"},
    {"id": 7, "name": "手工红糖姜茶", "price": 29.9, "stock": 210, "category": "食品", "platform": "淘宝", "status": "在售", "sales": 3456, "icon": "fa-candy-cane", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "300g/盒", "sku": "JC-2026-007", "rating": "4.9星", "subcat": "茶饮", "third": "礼盒装"},
    {"id": 8, "name": "云山国风丝巾", "price": 68, "stock": 76, "category": "文创", "platform": "拼多多", "status": "在售", "sales": 789, "icon": "fa-palette", "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": "90×90cm", "sku": "SJ-2026-008", "rating": "4.8星", "subcat": "编织", "third": "手工"},
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
                return data
        except Exception:
            return DEFAULT_PRODUCTS
    return DEFAULT_PRODUCTS

def save_products(products):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

products = load_products()

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
                    "分类只支持：茶叶、手工艺、食品、文创。"
                    "平台只支持：淘宝、抖音、拼多多、京东。"
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

@app.get("/api/products")
async def get_products():
    return {"products": products}

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
        new_product = {"id": new_id, "name": name, "price": price, "stock": stock, "category": category, "platform": platform, "status": "在售", "sales": 0, "icon": icon_map.get(category, "fa-box"), "main_image": "", "sub_images": [], "video": "", "detail_html": "", "spec": spec, "sku": sku, "rating": "", "subcat": "", "third": ""}
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
        for p in matched: products.remove(p)
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
        return {"success": True, "message": f"「{p['name']}」媒体信息：\n· 主图：{main_status}\n· 副图：{sub_count} 张\n· 视频：{video_status}\n（上传/修改请打开 images.html 页面）"}

    if t == "delete_main_image":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        if not p.get("main_image"): return {"success": False, "message": f"「{p['name']}」没有主图"}
        p["main_image"] = ""; save_products(products)
        return {"success": True, "message": f"已删除「{p['name']}」的主图"}

    if t == "remove_sub_image":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        index = args.get("index", 1)
        subs = p.get("sub_images", [])
        if index < 1 or index > len(subs): return {"success": False, "message": f"副图序号 {index} 不存在，当前共 {len(subs)} 张"}
        subs.pop(index - 1); p["sub_images"] = subs; save_products(products)
        return {"success": True, "message": f"已删除「{p['name']}」的第 {index} 张副图，剩余 {len(subs)} 张"}

    if t == "clear_sub_images":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        count = len(p.get("sub_images", []))
        if count == 0: return {"success": False, "message": f"「{p['name']}」没有副图"}
        p["sub_images"] = []; save_products(products)
        return {"success": True, "message": f"已清空「{p['name']}」的所有副图（共 {count} 张）"}

    if t == "clear_all_images":
        p = find_product(args.get("product_name"))
        if not p: return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
        main_exist = bool(p.get("main_image")); sub_count = len(p.get("sub_images", []))
        if not main_exist and sub_count == 0: return {"success": False, "message": f"「{p['name']}」没有任何图片"}
        p["main_image"] = ""; p["sub_images"] = []; save_products(products)
        return {"success": True, "message": f"已清空「{p['name']}」的所有图片"}

    return {"success": False, "message": f"未知操作类型：{t}"}

@app.post("/api/images/upload")
async def upload_image(
    product_name: str = Form(...),
    image_type: str = Form(...),
    file: UploadFile = File(...)
):
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

    b64 = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{file.content_type};base64,{b64}"

    if image_type == "main":
        p["main_image"] = data_url
        save_products(products)
        add_log("set_main_image", {"product_name": product_name}, f"已设置「{p['name']}」的主图", True)
        return {"success": True, "message": f"已设置「{p['name']}」的主图"}

    elif image_type == "sub":
        subs = p.get("sub_images", [])
        if len(subs) >= 5:
            return {"success": False, "message": f"「{p['name']}」副图已达上限（5 张）"}
        subs.append(data_url)
        p["sub_images"] = subs
        save_products(products)
        add_log("add_sub_image", {"product_name": product_name}, f"已为「{p['name']}」添加第 {len(subs)} 张副图", True)
        return {"success": True, "message": f"已为「{p['name']}」添加第 {len(subs)} 张副图"}

    elif image_type == "video":
        p["video"] = data_url
        save_products(products)
        add_log("set_video", {"product_name": product_name}, f"已设置「{p['name']}」的视频", True)
        return {"success": True, "message": f"已设置「{p['name']}」的视频"}

    elif image_type == "detail":
        return {"success": True, "url": data_url, "message": "图片已上传"}

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
        p["main_image"] = ""; save_products(products)
        add_log("delete_main_image", {"product_name": product_name}, f"已删除「{p['name']}」的主图", True)
        return {"success": True, "message": f"已删除「{p['name']}」的主图"}

    elif image_type == "sub":
        subs = p.get("sub_images", [])
        if index is None or index < 0 or index >= len(subs):
            return {"success": False, "message": "副图序号无效"}
        subs.pop(index); p["sub_images"] = subs; save_products(products)
        add_log("remove_sub_image", {"product_name": product_name, "index": index + 1}, f"已删除「{p['name']}」的第 {index + 1} 张副图", True)
        return {"success": True, "message": f"已删除「{p['name']}」的第 {index + 1} 张副图"}

    elif image_type == "video":
        if not p.get("video"):
            return {"success": False, "message": "没有视频可删除"}
        p["video"] = ""; save_products(products)
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

# 前端直接编辑商品全量字段（用于商品管理页的编辑/新增）
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
    }
    products.append(new_product)
    save_products(products)
    add_log("add_product", {"name": name}, f"已新增商品「{name}」", True)
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
    products.remove(p)
    save_products(products)
    add_log("delete_product", {"product_name": name}, f"已删除「{name}」", True)
    return {"success": True, "message": f"已删除「{name}」"}

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
    return {"status": "ok", "message": "AI 助手后端服务运行中"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

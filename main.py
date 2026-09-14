from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
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

# ============================================================
# 商品数据存储（JSON 文件持久化）
# ============================================================
DATA_FILE = "products.json"

DEFAULT_PRODUCTS = [
    {"id": 1, "name": "云山茶叶礼盒", "price": 118, "stock": 234, "category": "茶叶", "platform": "抖音", "status": "在售", "sales": 1247, "icon": "fa-leaf"},
    {"id": 2, "name": "手工竹编包", "price": 89, "stock": 237, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 856, "icon": "fa-bag-shopping"},
    {"id": 3, "name": "山核桃仁 250g", "price": 45, "stock": 156, "category": "食品", "platform": "拼多多", "status": "在售", "sales": 2345, "icon": "fa-seedling"},
    {"id": 4, "name": "云山手工皂套装", "price": 79, "stock": 89, "category": "文创", "platform": "京东", "status": "待审核", "sales": 567, "icon": "fa-soap"},
    {"id": 5, "name": "云山手作陶瓷杯", "price": 58, "stock": 143, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 1876, "icon": "fa-mug-saucer"},
    {"id": 6, "name": "手写书法折扇", "price": 35, "stock": 0, "category": "文创", "platform": "抖音", "status": "下架", "sales": 234, "icon": "fa-scroll"},
    {"id": 7, "name": "手工红糖姜茶", "price": 29.9, "stock": 210, "category": "食品", "platform": "淘宝", "status": "在售", "sales": 3456, "icon": "fa-candy-cane"},
    {"id": 8, "name": "云山国风丝巾", "price": 68, "stock": 76, "category": "文创", "platform": "拼多多", "status": "在售", "sales": 789, "icon": "fa-palette"},
]

def load_products():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_PRODUCTS
    return DEFAULT_PRODUCTS

def save_products(products):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

products = load_products()

def find_product(name: str):
    """模糊匹配商品名"""
    if not name:
        return None
    clean = name.replace(" ", "")
    for p in products:
        if p["name"].replace(" ", "") in clean or clean in p["name"].replace(" ", ""):
            return p
    if "茶叶" in clean or "礼盒" in clean:
        for p in products:
            if "茶叶" in p["name"]:
                return p
    if "竹编" in clean:
        for p in products:
            if "竹编" in p["name"]:
                return p
    if "核桃" in clean:
        for p in products:
            if "核桃" in p["name"]:
                return p
    if "皂" in clean:
        for p in products:
            if "皂" in p["name"]:
                return p
    if "陶瓷" in clean or "杯" in clean:
        for p in products:
            if "陶瓷" in p["name"]:
                return p
    if "折扇" in clean or "扇" in clean:
        for p in products:
            if "扇" in p["name"]:
                return p
    if "姜茶" in clean or "红糖" in clean:
        for p in products:
            if "姜茶" in p["name"]:
                return p
    if "丝巾" in clean:
        for p in products:
            if "丝巾" in p["name"]:
                return p
    return None

# ============================================================
# AI 工具定义（15个）
# ============================================================
tools = [
    # ===== 原有 9 个 =====
    {
        "type": "function",
        "function": {
            "name": "update_price",
            "description": "修改指定商品的价格。当用户说'把XX的价格改成YY元'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "商品名称"},
                    "new_price": {"type": "number", "description": "新的价格"}
                },
                "required": ["product_name", "new_price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": "修改指定商品的库存。action 表示操作类型：increase=增加，decrease=减少，set=设置为具体数值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "action": {"type": "string", "enum": ["increase", "decrease", "set"]},
                    "amount": {"type": "integer"}
                },
                "required": ["product_name", "action", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_low_stock",
            "description": "查询库存低于指定阈值的商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "integer"}
                },
                "required": ["threshold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "publish_to_platforms",
            "description": "把商品上架到指定平台。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "platforms": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["product_name", "platforms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_title",
            "description": "修改商品标题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "new_title": {"type": "string"}
                },
                "required": ["product_name", "new_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_off_shelf",
            "description": "下架单个商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "put_on_shelf",
            "description": "上架单个商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sales",
            "description": "查询指定商品的销售数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_coupon",
            "description": "创建优惠券。",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "integer", "description": "满减门槛金额"},
                    "discount": {"type": "integer", "description": "减免金额"},
                    "days": {"type": "integer", "description": "有效天数，默认7"}
                },
                "required": ["threshold", "discount"]
            }
        }
    },
    # ===== 新增 6 个 =====
    {
        "type": "function",
        "function": {
            "name": "batch_take_off",
            "description": "批量下架某个分类的所有商品。当用户说'把某分类的商品全部下架'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "商品分类，例如：茶叶、手工艺、食品、文创"}
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_update_price",
            "description": "批量调整某个分类的商品价格。adjust_type 表示调整方式：percent=按百分比，fixed=固定金额。正数表示上调，负数表示下调。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "商品分类"},
                    "adjust_type": {"type": "string", "enum": ["percent", "fixed"], "description": "percent=百分比，fixed=固定金额"},
                    "adjust_value": {"type": "number", "description": "调整值，例如10表示+10%，-5表示-5%（percent）；或5表示+5元，-3表示-3元（fixed）"}
                },
                "required": ["category", "adjust_type", "adjust_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_by_price_range",
            "description": "查询价格在指定区间内的商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_price": {"type": "number", "description": "最低价格"},
                    "max_price": {"type": "number", "description": "最高价格"}
                },
                "required": ["min_price", "max_price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_take_off_zero_stock",
            "description": "把所有库存为0的商品下架。",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_markup_all",
            "description": "给所有商品统一加价（按百分比）。当用户说'给所有商品加X%的利润'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {"type": "number", "description": "加价百分比，例如10表示加价10%"}
                },
                "required": ["percent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sort_by_sales",
            "description": "按销量排序查看商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order": {"type": "string", "enum": ["desc", "asc"], "description": "desc=从高到低，asc=从低到高，默认desc"},
                    "limit": {"type": "integer", "description": "显示前几个，默认10"}
                },
                "required": []
            }
        }
    },
]

class ChatRequest(BaseModel):
    text: str

@app.post("/api/ai/parse")
async def parse_intent(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是电商运营助手。用户会用自然语言下达指令，"
                        "你需要判断应该调用哪个工具，并提取参数。"
                        "如果用户只是闲聊或问问题，不要调用工具，直接回复。"
                        "注意分类只支持：茶叶、手工艺、食品、文创。"
                    )
                },
                {"role": "user", "content": req.text}
            ],
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return {
                "type": call.function.name,
                "args": json.loads(call.function.arguments)
            }
        else:
            return {"type": "chat", "text": msg.content}
    except Exception as e:
        return {"type": "error", "text": str(e)}

# ============================================================
# 商品数据接口
# ============================================================
@app.get("/api/products")
async def get_products():
    return {"products": products}

# ============================================================
# AI 指令执行接口
# ============================================================
class ExecuteRequest(BaseModel):
    type: str
    args: dict

@app.post("/api/ai/execute")
async def execute_action(req: ExecuteRequest):
    global products
    t = req.type
    args = req.args

    try:
        # ===== 原有 9 个 =====
        if t == "update_price":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            old = p["price"]
            p["price"] = args["new_price"]
            save_products(products)
            return {"success": True, "message": f"已将「{p['name']}」的价格从 ¥{old} 改为 ¥{p['price']}"}

        if t == "update_stock":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            old = p["stock"]
            action = args.get("action")
            amount = args.get("amount", 0)
            if action == "increase":
                p["stock"] = old + amount
            elif action == "decrease":
                p["stock"] = max(0, old - amount)
            else:
                p["stock"] = amount
            save_products(products)
            action_text = {"increase": "增加", "decrease": "减少", "set": "设置为"}.get(action, "调整")
            return {"success": True, "message": f"已将「{p['name']}」的库存{action_text} {amount} 件，当前库存 {p['stock']} 件"}

        if t == "query_low_stock":
            threshold = args.get("threshold", 50)
            low = [p for p in products if p["stock"] < threshold]
            if not low:
                return {"success": True, "message": f"没有库存低于 {threshold} 件的商品"}
            lines = [f"· {p['name']}：库存 {p['stock']} 件" for p in low]
            return {"success": True, "message": f"库存低于 {threshold} 件的商品共 {len(low)} 个：\n" + "\n".join(lines)}

        if t == "publish_to_platforms":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            platforms = args.get("platforms", [])
            p["platform"] = platforms[0] if platforms else p["platform"]
            p["status"] = "在售"
            save_products(products)
            return {"success": True, "message": f"已将「{p['name']}」上架到 {', '.join(platforms)}"}

        if t == "update_title":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            old = p["name"]
            p["name"] = args["new_title"]
            save_products(products)
            return {"success": True, "message": f"已将「{old}」的标题改为「{p['name']}」"}

        if t == "take_off_shelf":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            p["status"] = "下架"
            save_products(products)
            return {"success": True, "message": f"已下架「{p['name']}」"}

        if t == "put_on_shelf":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            p["status"] = "在售"
            save_products(products)
            return {"success": True, "message": f"已上架「{p['name']}」"}

        if t == "query_sales":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            return {"success": True, "message": f"「{p['name']}」累计销量 {p['sales']} 件，当前价格 ¥{p['price']}，库存 {p['stock']} 件"}

        if t == "create_coupon":
            threshold = args.get("threshold")
            discount = args.get("discount")
            days = args.get("days", 7)
            return {"success": True, "message": f"已创建优惠券：满 {threshold} 减 {discount}，有效期 {days} 天"}

        # ===== 新增 6 个 =====
        if t == "batch_take_off":
            category = args.get("category")
            matched = [p for p in products if p["category"] == category]
            if not matched:
                return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
            for p in matched:
                p["status"] = "下架"
            save_products(products)
            names = "、".join([p["name"] for p in matched])
            return {"success": True, "message": f"已批量下架「{category}」分类共 {len(matched)} 个商品：\n{names}"}

        if t == "batch_update_price":
            category = args.get("category")
            adjust_type = args.get("adjust_type", "percent")
            adjust_value = args.get("adjust_value", 0)
            matched = [p for p in products if p["category"] == category]
            if not matched:
                return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
            lines = []
            for p in matched:
                old = p["price"]
                if adjust_type == "percent":
                    p["price"] = round(old * (1 + adjust_value / 100), 2)
                else:
                    p["price"] = round(old + adjust_value, 2)
                lines.append(f"· {p['name']}：¥{old} → ¥{p['price']}")
            save_products(products)
            action_text = f"{'+' if adjust_value > 0 else ''}{adjust_value}%" if adjust_type == "percent" else f"{'+' if adjust_value > 0 else ''}{adjust_value}元"
            return {"success": True, "message": f"已将「{category}」分类商品价格统一调整（{action_text}）：\n" + "\n".join(lines)}

        if t == "query_by_price_range":
            min_p = args.get("min_price", 0)
            max_p = args.get("max_price", 99999)
            matched = [p for p in products if min_p <= p["price"] <= max_p]
            if not matched:
                return {"success": True, "message": f"没有价格在 ¥{min_p} - ¥{max_p} 之间的商品"}
            lines = [f"· {p['name']}：¥{p['price']}（库存 {p['stock']}）" for p in matched]
            return {"success": True, "message": f"价格在 ¥{min_p} - ¥{max_p} 之间的商品共 {len(matched)} 个：\n" + "\n".join(lines)}

        if t == "batch_take_off_zero_stock":
            matched = [p for p in products if p["stock"] == 0]
            if not matched:
                return {"success": True, "message": "没有库存为 0 的商品"}
            for p in matched:
                p["status"] = "下架"
            save_products(products)
            names = "、".join([p["name"] for p in matched])
            return {"success": True, "message": f"已下架库存为 0 的商品共 {len(matched)} 个：\n{names}"}

        if t == "batch_markup_all":
            percent = args.get("percent", 10)
            lines = []
            for p in products:
                old = p["price"]
                p["price"] = round(old * (1 + percent / 100), 2)
                lines.append(f"· {p['name']}：¥{old} → ¥{p['price']}")
            save_products(products)
            return {"success": True, "message": f"已给所有商品加价 {percent}%：\n" + "\n".join(lines)}

        if t == "query_sort_by_sales":
            order = args.get("order", "desc")
            limit = args.get("limit", 10)
            sorted_products = sorted(products, key=lambda p: p["sales"], reverse=(order == "desc"))[:limit]
            lines = [f"{i+1}. {p['name']}：销量 {p['sales']} 件" for i, p in enumerate(sorted_products)]
            order_text = "从高到低" if order == "desc" else "从低到高"
            return {"success": True, "message": f"按销量{order_text}排序（前 {len(sorted_products)} 个）：\n" + "\n".join(lines)}

        return {"success": False, "message": f"未知操作类型：{t}"}

    except Exception as e:
        return {"success": False, "message": f"执行失败：{str(e)}"}

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI 助手后端服务运行中"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

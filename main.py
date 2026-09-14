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
# 商品数据存储
# ============================================================
DATA_FILE = "products.json"

DEFAULT_PRODUCTS = [
    {"id": 1, "name": "云山茶叶礼盒", "price": 128, "stock": 234, "category": "茶叶", "platform": "抖音", "status": "在售", "sales": 1247, "icon": "fa-leaf"},
    {"id": 2, "name": "手工竹编包", "price": 89, "stock": 247, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 856, "icon": "fa-bag-shopping"},
    {"id": 3, "name": "山核桃仁 250g", "price": 45, "stock": 156, "category": "食品", "platform": "拼多多", "status": "在售", "sales": 2345, "icon": "fa-seedling"},
    {"id": 4, "name": "云山手工皂套装", "price": 79, "stock": 89, "category": "文创", "platform": "京东", "status": "待审核", "sales": 567, "icon": "fa-soap"},
    {"id": 5, "name": "云山陶瓷杯", "price": 58, "stock": 143, "category": "手工艺", "platform": "淘宝", "status": "在售", "sales": 1876, "icon": "fa-mug-saucer"},
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
    """查找商品：精确匹配优先，其次模糊匹配，最后别名匹配"""
    if not name:
        return None
    clean = name.replace(" ", "").strip()

    # 1. 精确匹配
    for p in products:
        if p["name"].replace(" ", "") == clean:
            return p

    # 2. 模糊匹配（包含关系）
    for p in products:
        pn = p["name"].replace(" ", "")
        if pn in clean or clean in pn:
            return p

    # 3. 别名匹配
    aliases = {
        "茶叶": "茶叶", "礼盒": "茶叶",
        "竹编": "竹编",
        "核桃": "核桃",
        "皂": "皂",
        "陶瓷": "陶瓷", "杯": "陶瓷",
        "折扇": "扇", "扇": "扇",
        "姜茶": "姜茶", "红糖": "姜茶",
        "丝巾": "丝巾",
        "红茶": "红茶",
        "茶": "茶"
    }
    for key, val in aliases.items():
        if key in clean:
            for p in products:
                if val in p["name"] or val in p["category"]:
                    return p
    return None

# ============================================================
# AI 工具定义（25个）
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "update_price",
            "description": "修改指定商品的价格。当用户说'把XX的价格改成YY元'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "new_price": {"type": "number"}
                },
                "required": ["product_name", "new_price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": "修改指定商品的库存。action: increase=增加，decrease=减少，set=设置为。",
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
                "properties": {"threshold": {"type": "integer"}},
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
                "properties": {"product_name": {"type": "string"}},
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
                "properties": {"product_name": {"type": "string"}},
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
                "properties": {"product_name": {"type": "string"}},
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
                    "threshold": {"type": "integer"},
                    "discount": {"type": "integer"},
                    "days": {"type": "integer"}
                },
                "required": ["threshold", "discount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_take_off",
            "description": "批量下架某个分类的所有商品。",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_update_price",
            "description": "批量调整某个分类的商品价格。adjust_type: percent=百分比，fixed=固定金额。正数上调，负数下调。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "adjust_type": {"type": "string", "enum": ["percent", "fixed"]},
                    "adjust_value": {"type": "number"}
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
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"}
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
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_markup_all",
            "description": "给所有商品统一加价（按百分比）。",
            "parameters": {
                "type": "object",
                "properties": {"percent": {"type": "number"}},
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
                    "order": {"type": "string", "enum": ["desc", "asc"]},
                    "limit": {"type": "integer"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_product",
            "description": "新增一个商品。当用户说'新增一个商品叫XX，价格YY元'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "商品名称"},
                    "price": {"type": "number", "description": "价格"},
                    "stock": {"type": "integer", "description": "库存，默认0"},
                    "category": {"type": "string", "description": "分类：茶叶、手工艺、食品、文创"},
                    "platform": {"type": "string", "description": "平台：淘宝、抖音、拼多多、京东"}
                },
                "required": ["name", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_product",
            "description": "删除指定商品。",
            "parameters": {
                "type": "object",
                "properties": {"product_name": {"type": "string"}},
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_update_stock",
            "description": "批量修改某个分类的商品库存。action: increase=增加，decrease=减少，set=设置为。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "action": {"type": "string", "enum": ["increase", "decrease", "set"]},
                    "amount": {"type": "integer"}
                },
                "required": ["category", "action", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_by_platform",
            "description": "查询某个平台上架的商品。",
            "parameters": {
                "type": "object",
                "properties": {"platform": {"type": "string"}},
                "required": ["platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_by_status",
            "description": "查询指定状态的商品。",
            "parameters": {
                "type": "object",
                "properties": {"status": {"type": "string", "enum": ["在售", "下架", "待审核"]}},
                "required": ["status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_put_on_shelf",
            "description": "批量上架某个分类的所有商品。",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "duplicate_product",
            "description": "复制一个商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "new_name": {"type": "string"}
                },
                "required": ["product_name", "new_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_by_category",
            "description": "查询某个分类的所有商品。",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_products",
            "description": "导出所有商品数据。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "按关键词搜索商品，支持名字、分类、平台模糊匹配。",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"]
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
                        "分类只支持：茶叶、手工艺、食品、文创。"
                        "平台只支持：淘宝、抖音、拼多多、京东。"
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

@app.get("/api/products")
async def get_products():
    return {"products": products}

class ExecuteRequest(BaseModel):
    type: str
    args: dict

@app.post("/api/ai/execute")
async def execute_action(req: ExecuteRequest):
    global products
    t = req.type
    args = req.args

    try:
        # ===== 单商品操作 =====
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

        # ===== 批量操作 =====
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

        # ===== 新增商品 / 删除 / 批量库存 =====
        if t == "add_product":
            name = args.get("name")
            price = args.get("price")
            stock = args.get("stock", 0)
            category = args.get("category", "文创")
            platform = args.get("platform", "淘宝")
            if not name or price is None:
                return {"success": False, "message": "缺少商品名称或价格"}
            # 精确检查重名
            if any(p["name"] == name for p in products):
                return {"success": False, "message": f"商品「{name}」已存在"}
            new_id = max([p["id"] for p in products], default=0) + 1
            icon_map = {"茶叶": "fa-leaf", "手工艺": "fa-bag-shopping", "食品": "fa-seedling", "文创": "fa-palette"}
            new_product = {
                "id": new_id, "name": name, "price": price, "stock": stock,
                "category": category, "platform": platform, "status": "在售",
                "sales": 0, "icon": icon_map.get(category, "fa-box")
            }
            products.append(new_product)
            save_products(products)
            return {"success": True, "message": f"已新增商品「{name}」：价格 ¥{price}，库存 {stock} 件，分类 {category}，平台 {platform}"}

        if t == "delete_product":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            name = p["name"]
            products.remove(p)
            save_products(products)
            return {"success": True, "message": f"已删除商品「{name}」"}

        if t == "batch_update_stock":
            category = args.get("category")
            action = args.get("action", "increase")
            amount = args.get("amount", 0)
            matched = [p for p in products if p["category"] == category]
            if not matched:
                return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
            lines = []
            for p in matched:
                old = p["stock"]
                if action == "increase":
                    p["stock"] = old + amount
                elif action == "decrease":
                    p["stock"] = max(0, old - amount)
                else:
                    p["stock"] = amount
                lines.append(f"· {p['name']}：{old} → {p['stock']}")
            save_products(products)
            action_text = {"increase": "增加", "decrease": "减少", "set": "设置为"}.get(action, "调整")
            return {"success": True, "message": f"已将「{category}」分类商品库存统一{action_text} {amount} 件：\n" + "\n".join(lines)}

        # ===== 查询 =====
        if t == "query_by_platform":
            platform = args.get("platform")
            matched = [p for p in products if p["platform"] == platform]
            if not matched:
                return {"success": True, "message": f"「{platform}」平台上没有商品"}
            lines = [f"· {p['name']}（{p['status']}）：¥{p['price']}，库存 {p['stock']}" for p in matched]
            return {"success": True, "message": f"「{platform}」平台共有 {len(matched)} 个商品：\n" + "\n".join(lines)}

        if t == "query_by_status":
            status = args.get("status")
            matched = [p for p in products if p["status"] == status]
            if not matched:
                return {"success": True, "message": f"没有「{status}」状态的商品"}
            lines = [f"· {p['name']}（{p['category']}）：¥{p['price']}，库存 {p['stock']}" for p in matched]
            return {"success": True, "message": f"「{status}」状态的商品共 {len(matched)} 个：\n" + "\n".join(lines)}

        if t == "batch_put_on_shelf":
            category = args.get("category")
            matched = [p for p in products if p["category"] == category]
            if not matched:
                return {"success": False, "message": f"没有找到分类为「{category}」的商品"}
            for p in matched:
                p["status"] = "在售"
            save_products(products)
            names = "、".join([p["name"] for p in matched])
            return {"success": True, "message": f"已批量上架「{category}」分类共 {len(matched)} 个商品：\n{names}"}

        if t == "duplicate_product":
            p = find_product(args.get("product_name"))
            if not p:
                return {"success": False, "message": f"未找到商品：{args.get('product_name')}"}
            new_name = args.get("new_name")
            if not new_name:
                return {"success": False, "message": "缺少新商品名称"}
            # 精确检查重名
            if any(prod["name"] == new_name for prod in products):
                return {"success": False, "message": f"商品「{new_name}」已存在"}
            new_id = max([p["id"] for p in products], default=0) + 1
            new_product = dict(p)
            new_product["id"] = new_id
            new_product["name"] = new_name
            new_product["sales"] = 0
            products.append(new_product)
            save_products(products)
            return {"success": True, "message": f"已复制「{p['name']}」为「{new_name}」：价格 ¥{new_product['price']}，库存 {new_product['stock']} 件"}

        if t == "query_by_category":
            category = args.get("category")
            matched = [p for p in products if p["category"] == category]
            if not matched:
                return {"success": True, "message": f"没有「{category}」分类的商品"}
            lines = [f"· {p['name']}（{p['status']}）：¥{p['price']}，库存 {p['stock']}，销量 {p['sales']}" for p in matched]
            return {"success": True, "message": f"「{category}」分类共 {len(matched)} 个商品：\n" + "\n".join(lines)}

        if t == "export_products":
            return {"success": True, "message": f"共 {len(products)} 个商品，数据已可从前端 /api/products 接口获取"}

        if t == "search_product":
            keyword = args.get("keyword", "").strip()
            if not keyword:
                return {"success": False, "message": "搜索关键词不能为空"}
            # 多字段模糊匹配：名字、分类、平台
            matched = [
                p for p in products
                if keyword in p["name"] or keyword in p["category"] or keyword in p.get("platform", "")
            ]
            if not matched:
                return {"success": True, "message": f"没有找到包含「{keyword}」的商品"}
            lines = [f"· {p['name']}（{p['category']}）：¥{p['price']}，库存 {p['stock']}" for p in matched]
            return {"success": True, "message": f"搜索「{keyword}」共找到 {len(matched)} 个商品：\n" + "\n".join(lines)}

        return {"success": False, "message": f"未知操作类型：{t}"}

    except Exception as e:
        return {"success": False, "message": f"执行失败：{str(e)}"}

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI 助手后端服务运行中"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

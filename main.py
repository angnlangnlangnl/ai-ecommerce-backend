from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

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

tools = [
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
            "description": "修改指定商品的库存。当用户说'把XX的库存增加/减少/改成YY件'时调用。action 表示操作类型：increase=增加，decrease=减少，set=设置为具体数值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "商品名称"},
                    "action": {
                        "type": "string",
                        "enum": ["increase", "decrease", "set"],
                        "description": "操作类型：increase=增加，decrease=减少，set=设置为"
                    },
                    "amount": {"type": "integer", "description": "数量"}
                },
                "required": ["product_name", "action", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_low_stock",
            "description": "查询库存低于指定阈值的商品。当用户说'查一下库存低于XX件的商品'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "integer", "description": "库存阈值，例如50"}
                },
                "required": ["threshold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "publish_to_platforms",
            "description": "把商品上架到指定平台。当用户说'把XX上架到YY平台'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "商品名称"},
                    "platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "目标平台列表，例如 ['抖音', '淘宝']"
                    }
                },
                "required": ["product_name", "platforms"]
            }
        }
    }
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

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI 助手后端服务运行中"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

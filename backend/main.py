"""
航班智能助手后端
- 聊天 + Function Call + RAG + 记忆 + 历史记录
"""

import json
import re
import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="航班助手")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# ========== 数据存储（内存） ==========
chat_histories = {}   # session_id -> [messages]
user_memories = {}     # session_id -> {key: value}
subscriptions = {      # flight_id -> [session_id]
    "CA1234": ["default"],
    "MU2567": ["default"],
    "CZ3388": ["default"],
}

# ========== RAG 知识库 ==========
KB_DOCS = [
    {"id": 1, "title": "退订航班通知 / Unsubscribe", "content": "unsubscribe / 退订 / 取消订阅: 用户可以通过取消订阅功能停止接收特定航班的动态通知。只需提供航班号即可完成退订。"},
    {"id": 2, "title": "航班延误赔偿 / Delay Compensation", "content": "delay / 延误 / compensation / 赔偿: 国内航班延误超过4小时，航空公司应提供餐食或住宿。延误超过8小时，可申请经济补偿。"},
    {"id": 3, "title": "退改签政策 / Refund Policy", "content": "refund / cancel / change / 退改签 / 退票: 一般提前24小时以上退票免费，24小时内收取票价5-20%手续费。特价机票不可退改。"},
    {"id": 4, "title": "行李规定 / Baggage Allowance", "content": "baggage / luggage / 行李 / allowance: 国内经济舱免费托运20kg，商务舱30kg，头等舱40kg。随身行李不超过7kg。"},
    {"id": 5, "title": "选座服务 / Seat Selection", "content": "seat / 选座 / check-in / 值机: 网上值机通常提前24小时开放。紧急出口排座位需到柜台办理。"},
    {"id": 6, "title": "特殊旅客服务 / Special Assistance", "content": "special / elderly / child / 特殊 / 老人 / 儿童: 老人、儿童、孕妇等特殊旅客可申请陪伴服务，需提前48小时申请。"},
    {"id": 7, "title": "航班取消保护 / Cancellation Protection", "content": "cancel / 取消: 航班取消后，旅客可免费改签至当日同航线其他航班，或申请全额退款。"},
    {"id": 8, "title": "会员积分 / Loyalty Points", "content": "points / miles / 积分 / 会员: 每消费1元积1分，积分可兑换机票或升舱。金银卡会员享受额外行李额。"},
]


def rag_retrieve(query: str, top_k: int = 2) -> list:
    """简单的关键词匹配 RAG 检索"""
    query_words = set(re.sub(r'[^\w]', ' ', query).lower().split())
    scored = []
    for doc in KB_DOCS:
        text = (doc["title"] + " " + doc["content"]).lower()
        text_words = set(re.sub(r'[^\w]', ' ', text).lower().split())
        overlap = len(query_words & text_words)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_k]]


# ========== 系统提示词 ==========
SYSTEM_PROMPT = """你是一个航班智能助手，负责帮助用户处理与航班相关的事务。

## 你的能力
1. 查询和解释航班信息
2. 管理航班通知订阅（订阅/退订）
3. 记住用户偏好（座位偏好、常飞航线等）
4. 回答航班相关政策和规则

## 工具
当用户要求取消航班通知时，你必须调用 unsubscribe_flight 工具。
当用户表达偏好或习惯时，记住这些信息。

## 行为准则
- 回答简洁准确，使用中文
- 涉及退订时必须调用工具
- 不确定的政策引导用户联系航空公司确认"""


# ========== 工具定义 ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "unsubscribe_flight",
            "description": "取消指定航班的通知订阅",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {
                        "type": "string",
                        "description": "要退订的航班号，如 CA1234"
                    },
                    "reason": {
                        "type": "string",
                        "description": "退订原因"
                    }
                },
                "required": ["flight_id"]
            }
        }
    }
]


# ========== API 模型 ==========
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    flight_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    tool_calls: list = []
    session_id: str


# ========== 核心逻辑 ==========

def execute_unsubscribe(flight_id: str, reason: str = "") -> str:
    """执行退订操作"""
    global subscriptions
    removed = 0
    for fid in list(subscriptions.keys()):
        if fid == flight_id or not flight_id:
            removed += len(subscriptions[fid])
            del subscriptions[fid]
    if removed > 0:
        return f"已成功取消航班 {flight_id} 的通知订阅"
    return f"未找到航班 {flight_id} 的订阅记录"


def save_memory(session_id: str, key: str, value: str):
    """保存用户记忆"""
    if session_id not in user_memories:
        user_memories[session_id] = {}
    user_memories[session_id][key] = value


def get_memory_context(session_id: str) -> str:
    """获取记忆上下文"""
    mems = user_memories.get(session_id, {})
    if not mems:
        return ""
    parts = [f"{k}: {v}" for k, v in mems.items()]
    return "用户偏好:\n" + "\n".join(parts)


def local_chat(messages: list, tools: list, memory_context: str, rag_context: str) -> dict:
    """无 API key 时的本地简易对话"""
    last = messages[-1]["content"] if messages else ""
    last_lower = last.lower()

    # 检测退订意图 — 只返回 tool call，不执行，由主 handler 统一处理
    if any(w in last for w in ["退订", "取消订阅", "unsubscribe", "不想接收", "取消通知"]):
        ids = re.findall(r'[A-Z]{2}\d{3,4}', last)
        if ids:
            return {
                "role": "assistant",
                "content": f"正在处理航班 {ids[0]} 的退订请求...",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "unsubscribe_flight",
                        "arguments": json.dumps({"flight_id": ids[0], "reason": "用户主动退订"})
                    }
                }]
            }
        return {"role": "assistant", "content": "请问你要退订哪个航班的通知？请提供航班号。"}

    # 检测记忆意图
    if any(w in last for w in ["记住", "我喜欢", "偏好", "习惯", "记得"]):
        save_memory("default", "偏好", last)
        return {"role": "assistant", "content": "好的，我记住了。"}

    # RAG 相关问题
    if any(w in last_lower for w in ["政策", "规定", "行李", "退改签", "赔偿", "积分", "选座", "延误", "policy", "luggage", "baggage", "delay", "cancel", "refund", "regulation"]):
        docs = rag_retrieve(last) if not rag_context else json.loads(rag_context)
        if docs:
            ctx = "\n".join([f"- {d['title']}: {d['content']}" for d in docs])
            return {"role": "assistant", "content": f"根据相关规定：\n{ctx}"}

    return {"role": "assistant", "content": f"收到你的消息。如有航班相关问题，请随时问我。"}


async def call_deepseek(messages: list, tools: list) -> dict:
    """调用 DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return None

    import httpx
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools if tools else None,
        "stream": False
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            return None
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        result = {"role": "assistant", "content": msg.get("content", "")}
        if "tool_calls" in msg:
            result["tool_calls"] = msg["tool_calls"]
        return result


# ========== 路由 ==========

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sid = req.session_id

    # 1. 初始化历史
    if sid not in chat_histories:
        chat_histories[sid] = []

    # 2. RAG 检索
    rag_docs = rag_retrieve(req.message)
    rag_context = json.dumps(rag_docs, ensure_ascii=False) if rag_docs else ""

    # 3. 记忆上下文
    memory_ctx = get_memory_context(sid)

    # 4. 构建 system prompt
    system_parts = [SYSTEM_PROMPT]
    if memory_ctx:
        system_parts.append(f"\n{memory_ctx}")
    if rag_context:
        system_parts.append(f"\n相关知识:\n" + "\n".join([f"- {d['title']}: {d['content']}" for d in rag_docs]))

    system_msg = {"role": "system", "content": "\n\n".join(system_parts)}

    # 5. 构建消息列表
    user_msg = {"role": "user", "content": req.message}
    if req.flight_id:
        user_msg["content"] = f"[航班 {req.flight_id}] {req.message}"

    messages = [system_msg] + chat_histories[sid][-10:] + [user_msg]

    # 6. 调用模型
    result = await call_deepseek(messages, TOOLS)

    if result is None:
        # 无 API key，使用本地逻辑
        result = local_chat(
            [{"role": "user", "content": req.message}],
            TOOLS,
            memory_ctx,
            rag_context if rag_docs else ""
        )

    # 7. 处理 tool_calls
    tool_results = []
    if "tool_calls" in result and result["tool_calls"]:
        for tc in result["tool_calls"]:
            fn = tc["function"]
            if fn["name"] == "unsubscribe_flight":
                args = json.loads(fn["arguments"])
                output = execute_unsubscribe(args.get("flight_id", ""), args.get("reason", ""))
                tool_results.append({
                    "tool_name": fn["name"],
                    "arguments": fn["arguments"],
                    "output": output
                })

    # 8. 保存历史
    chat_histories[sid].append(user_msg)
    chat_histories[sid].append({
        "role": "assistant",
        "content": result.get("content", "")
    })

    return ChatResponse(
        reply=result.get("content", ""),
        tool_calls=tool_results,
        session_id=sid
    )


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    msgs = chat_histories.get(session_id, [])
    return {"session_id": session_id, "messages": msgs}


@app.get("/api/memories/{session_id}")
def get_memories(session_id: str):
    mems = user_memories.get(session_id, {})
    return {"session_id": session_id, "memories": mems}


@app.get("/api/flights")
def get_flights():
    """返回航班列表及其订阅状态"""
    result = []
    import copy
    flights_data = [
        {"id": "CA1234", "from": "北京", "to": "上海", "departure": "08:00", "arrival": "10:15", "status": "on-time", "date": "2026-06-20"},
        {"id": "MU2567", "from": "上海", "to": "广州", "departure": "09:30", "arrival": "12:00", "status": "on-time", "date": "2026-06-20"},
        {"id": "CZ3388", "from": "广州", "to": "成都", "departure": "11:00", "arrival": "13:50", "status": "delayed", "date": "2026-06-20"},
        {"id": "HU4411", "from": "深圳", "to": "北京", "departure": "14:00", "arrival": "17:00", "status": "on-time", "date": "2026-06-20"},
        {"id": "3U8899", "from": "成都", "to": "杭州", "departure": "15:30", "arrival": "18:10", "status": "cancelled", "date": "2026-06-20"},
        {"id": "MF6622", "from": "厦门", "to": "西安", "departure": "16:45", "arrival": "19:20", "status": "on-time", "date": "2026-06-20"},
        {"id": "ZH9090", "from": "武汉", "to": "三亚", "departure": "18:00", "arrival": "20:35", "status": "delayed", "date": "2026-06-20"},
    ]
    for f in flights_data:
        f["subscribed"] = f["id"] in subscriptions
        result.append(f)
    return {"flights": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

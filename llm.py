from dataclasses import dataclass

@dataclass
class Response:
    """LLM调用的结构化响应"""
    content: str=""
    reasoning: str | None =None
    tool_call: dict | None =None
    # 后端可能提供的元数据，例如"model","prompt_tokens"
    metadata: dict | None =None

import json
import urllib.request
class LLM:
    """查询 OpenAI 兼容端点"""
    def __init__(
        self,
        model: str,
        base_url: str ="http://localhost:11434/v1",
        api_key: str ="no_key",
        think: bool =False,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think

    def generate(
        self,
        messages: list[dict],
        tools: list | None =None
    ) -> Response:
        """根据消息列表从LLM生成响应"""
        # 请求体
        body = {
            "model": self.model,
            "messages": messages,
        }

        # 工具和推理
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"

        # 向OpenAI兼容的/chat/completions 端点发起POST请求
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        response = urllib.request.urlopen(request)
        data = json.loads(response.read())

        # 提取message、tool_call and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }

        # 组装为Response数据类
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata,
        )


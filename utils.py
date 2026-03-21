import os
import datetime
from typing import Any

def get_current_hour_minute() -> tuple[int, int]:
    """현재 시스템 시각의 (시, 분)을 반환합니다."""
    now = datetime.datetime.now()
    return now.hour, now.minute

def _load_docs() -> str:
    """Docs/*.txt 파일들의 내용을 읽어 하나의 문자열로 합칩니다."""
    docs_dir = os.path.join(os.path.dirname(__file__), "Docs")
    if not os.path.exists(docs_dir):
        return ""
    
    combined_docs = []
    filenames = sorted(os.listdir(docs_dir))
    
    if "shortcuts.txt" in filenames:
        filenames.remove("shortcuts.txt")
        filenames.insert(0, "shortcuts.txt")

    for filename in filenames:
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    header = "### [필독/최우선] " if filename == "shortcuts.txt" else "### "
                    combined_docs.append(f"{header}문서: {filename}\n{content}\n")
            except Exception as e:
                combined_docs.append(f"### 문서: {filename}\n[오류 발생: {e}]\n")
    
    return "\n".join(combined_docs)

def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

def _history_to_messages(chat_history: list[Any] | None) -> list[Any]:
    from langchain_core.messages import HumanMessage, AIMessage
    messages: list[Any] = []
    if not chat_history:
        return messages

    for msg in chat_history:
        if isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            continue

        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))

    return messages

"""会话管理器 — 管理多轮对话历史"""

from typing import Dict, List
import time
from collections import defaultdict
from datetime import datetime


class SessionManager:
    def __init__(self, max_history_length: int = 10):
        self.sessions: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history_length = max_history_length
        self.last_activity: Dict[str, float] = {} #session_id => 时间

    def get_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        self._update_activity(session_id)
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, user_message: str, bot_reply: str):
        """添加对话记录"""
        self._update_activity(session_id)

        message_record = {
            "user_message": user_message,
            "bot_reply": bot_reply,
            "timestamp": datetime.now().isoformat(),
        }

        self.sessions[session_id].append(message_record)

        # 限制历史长度
        if len(self.sessions[session_id]) > self.max_history_length:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_length:]

    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.last_activity:
            del self.last_activity[session_id]

    def get_session_stats(self) -> Dict:
        """获取会话统计信息"""
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len([
                s for s, t in self.last_activity.items()
                if time.time() - t < 3600  # 1小时内活跃
            ]),
            "total_messages": sum(len(h) for h in self.sessions.values()),
        }

    def _update_activity(self, session_id: str):
        """更新会话活跃时间"""
        self.last_activity[session_id] = time.time()

    def cleanup_inactive_sessions(self, timeout_hours: int = 24) -> int:
        """清理不活跃的会话，返回清理数量"""
        current_time = time.time()
        timeout_seconds = timeout_hours * 3600

        inactive = [
            sid for sid, t in self.last_activity.items()
            if current_time - t > timeout_seconds
        ]

        for sid in inactive:
            self.clear_session(sid)

        return len(inactive)

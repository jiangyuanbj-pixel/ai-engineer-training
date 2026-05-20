"""测试客户端 — 覆盖阶段一、阶段二的典型场景"""

import requests
import json
import time


class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_user_{int(time.time())}"

    def send_message(self, message: str) -> dict:
        """发送消息"""
        url = f"{self.base_url}/chat"
        payload = {"session_id": self.session_id, "message": message}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_history(self) -> dict:
        """获取历史记录"""
        url = f"{self.base_url}/session/{self.session_id}/history"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def clear_session(self) -> dict:
        """清除会话"""
        url = f"{self.base_url}/session/{self.session_id}"
        try:
            response = requests.delete(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def health_check(self) -> dict:
        """健康检查"""
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def test_stage1_time_awareness(client: ChatClient):
    """阶段一测试：相对时间理解"""
    print("\n" + "=" * 50)
    print("阶段一测试：基础对话 + 时间理解")
    print("=" * 50)

    test_messages = [
        "你好，请问你是谁？",
        "我昨天下的单，请告诉我下单日期是什么，帮我看看什么时候能到？",
        "上周五我下了一个订单，现在什么状态？",
    ]

    for message in test_messages:
        print(f"\n👤 用户: {message}")
        result = client.send_message(message)
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
        else:
            print(f"🤖 助手: {result['reply']}")
        time.sleep(1)


def test_stage2_tool_calling(client: ChatClient):
    """阶段二测试：多轮对话 + 工具调用"""
    print("\n" + "=" * 50)
    print("阶段二测试：多轮对话 + 工具调用")
    print("=" * 50)

    # 场景1：查订单（追问订单号 → 提供订单号 → 查询结果）
    print("\n--- 场景1：查订单流程 ---")
    test_messages_1 = [
        "我想查一下订单",
        "订单号是1000000",
    ]

    for message in test_messages_1:
        print(f"\n👤 用户: {message}")
        result = client.send_message(message)
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
        else:
            print(f"🤖 助手: {result['reply']}")
        time.sleep(1)

    # 场景2：直接带订单号查询
    print("\n--- 场景2：直接查询订单 ---")
    result = client.send_message("帮我查一下订单1000002的详情")
    print(f"\n👤 用户: 帮我查一下订单1000002的详情")
    if "error" not in result:
        print(f"🤖 助手: {result['reply']}")
    time.sleep(1)

    # 场景3：查询不存在的订单
    print("\n--- 场景3：查询不存在的订单 ---")
    result = client.send_message("帮我查一下订单9999999")
    if "error" not in result:
        print(f"🤖 助手: {result['reply']}")
    time.sleep(1)

    # 场景4：申请退款
    print("\n--- 场景4：申请退款 ---")
    test_messages_4 = [
        "我想退款，订单号是1000001，原因是商品有问题",
    ]
    for message in test_messages_4:
        print(f"\n👤 用户: {message}")
        result = client.send_message(message)
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
        else:
            print(f"🤖 助手: {result['reply']}")
        time.sleep(1)

    # 场景5：重复退款
    print("\n--- 场景5：重复退款 ---")
    result = client.send_message("订单1000001我再退一次")
    if "error" not in result:
        print(f"🤖 助手: {result['reply']}")


def main():
    """测试主函数"""
    client = ChatClient()

    # 健康检查
    print("🏥 健康检查:")
    health = client.health_check()
    print(f"   {json.dumps(health, ensure_ascii=False)}")

    # 阶段一测试
    test_stage1_time_awareness(client)

    # 清除会话，用新会话测试阶段二
    client.clear_session()
    client.session_id = f"test_stage2_{int(time.time())}"

    # 阶段二测试
    test_stage2_tool_calling(client)

    # 查看历史
    print("\n" + "=" * 50)
    print("📋 对话历史:")
    print("=" * 50)
    history = client.get_history()
    if "error" not in history:
        for i, record in enumerate(history.get("history", []), 1):
            print(f"{i}. 👤 {record['user_message']}")
            print(f"   🤖 {record['bot_reply'][:100]}...")
            print(f"   🕐 {record['timestamp']}")


if __name__ == "__main__":
    main()

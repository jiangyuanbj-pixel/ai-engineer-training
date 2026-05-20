from langchain_core.tools import tool
from datetime import datetime
import json


# ============ 模拟数据 ============

order_list = {
    "1000000": {
        "order_id": "1000000",
        "order_time": "2025-06-01 12:00:00",
        "order_amount": 250.00,
        "order_status": "已完成",
        "order_items": [
            {"item_id": "123456", "item_name": "商品1", "item_price": 50.00, "item_quantity": 1},
            {"item_id": "123457", "item_name": "商品2", "item_price": 100.00, "item_quantity": 2},
        ],
        "order_address": "上海市徐汇区漕河泾开发区",
        "order_remarks": "无",
    },
    "1000001": {
        "order_id": "1000001",
        "order_time": "2025-06-02 10:30:00",
        "order_amount": 100.00,
        "order_status": "已完成",
        "order_items": [
            {"item_id": "123458", "item_name": "商品3", "item_price": 20.00, "item_quantity": 5},
        ],
        "order_address": "上海市浦东新区世纪大道",
        "order_remarks": "无",
    },
    "1000002": {
        "order_id": "1000002",
        "order_time": "2025-06-05 09:15:00",
        "order_amount": 380.00,
        "order_status": "配送中",
        "order_items": [
            {"item_id": "123459", "item_name": "商品4", "item_price": 190.00, "item_quantity": 2},
        ],
        "order_address": "北京市朝阳区望京SOHO",
        "order_remarks": "请尽快发货",
    },
    "1000003": {
        "order_id": "1000003",
        "order_time": "2025-06-06 14:20:00",
        "order_amount": 56.00,
        "order_status": "待发货",
        "order_items": [
            {"item_id": "123460", "item_name": "商品5", "item_price": 28.00, "item_quantity": 2},
        ],
        "order_address": "广州市天河区珠江新城",
        "order_remarks": "无",
    },
}

# 退款记录
refund_records = {}


# ============ 工具定义 ============

@tool
def get_current_time() -> str:
    """获取当前日期和时间，用于推断用户提到的相对时间（如'昨天'、'上周'等）"""
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


@tool
def query_order(order_id: str) -> str:
    """查询指定订单详细信息

    Args:
        order_id: 订单id，例如 1000000
    """
    result = order_list.get(order_id)
    if result is None:
        return f"订单 {order_id} 不存在，请确认订单号是否正确。"
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def apply_refund(order_id: str, reason: str) -> str:
    """申请退款

    Args:
        order_id: 订单id，例如 1000000
        reason: 退款原因
    """
    order = order_list.get(order_id)
    if order is None:
        return f"订单 {order_id} 不存在，请确认订单号是否正确。"

    if order["order_status"] == "已退款":
        return f"订单 {order_id} 已退过款，请勿重复申请。"

    if order["order_status"] == "已完成":
        # 模拟退款
        order["order_status"] = "已退款"
        refund_records[order_id] = {
            "order_id": order_id,
            "refund_amount": order["order_amount"],
            "reason": reason,
            "refund_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "退款中",
        }
        return f"订单 {order_id} 退款申请已提交，退款金额 {order['order_amount']} 元，原因：{reason}。预计3-5个工作日到账。"

    return f"订单 {order_id} 当前状态为「{order['order_status']}」，该状态暂不支持退款。"


# 所有工具列表
all_tools = [get_current_time, query_order, apply_refund]

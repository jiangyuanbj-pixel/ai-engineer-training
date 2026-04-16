from langchain_core.tools import tool
import json



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
}

#订单查询工具
@tool
def query_order(order_id: str) -> str:
    """查询指定订单详细信息
    
    Args:
        order_id: 订单id，例如1000011
    """
    # 实现你的逻辑
    result = order_list.get(order_id, {})
    if len(result) == 0:
        return "订单不存在"
    
    return f"订单id={order_id}的详细信息如下：\n" + json.dumps(result, indent=2)
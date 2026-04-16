import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import datetime

# 加载环境变量
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')

# ========== 1. 初始化 LLM（大语言模型） ==========
# LangChain 特点：统一的 LLM 接口，支持多种模型提供商
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="deepseek-chat",
    temperature=0.7  # LangChain 特点：统一的参数配置
)

# ========== 2. LLMChain：Prompt → LLM → 输出链的基本流程封装 ==========
def demo_llm_chain():
    """
    演示 LLMChain：支持变量注入与模板复用的核心组件
    LangChain 特点：模板化提示词管理，支持变量替换
    """
    print("=" * 50)
    print("🔗 LLMChain 演示：Prompt → LLM → 输出链")
    print("=" * 50)
    
    # 创建提示词模板 - LangChain 特点：模板复用
    prompt_template = PromptTemplate(
        input_variables=["topic", "style"],
        template="""
        请以{style}的风格，写一段关于{topic}的介绍。
        要求：简洁明了，不超过100字。
        """
    )
    
    # LangChain 0.3 推荐使用 LCEL (LangChain Expression Language)
    # 这是新的链式组合方式：prompt | llm
    chain = prompt_template | llm
    
    # 执行链 - 变量注入
    result = chain.invoke({"topic": "人工智能", "style": "科普"})
    print(f"📝 LLMChain 输出：\n{result.content}\n")
    
    return result.content

#以下是输出
"""
==================================================
🔗 LLMChain 演示：Prompt → LLM → 输出链
==================================================
📝 LLMChain 输出：
人工智能（AI）是让机器模拟人类智能的技术。它通过学习数据中的规律，能完成识别图像、理解语言、辅助决策等任务。如今，AI已融入生活，从手机助手到医疗诊断，正悄然改变我们的世界。
"""

# ========== 3. Tools：工具系统 ==========
def get_current_time(query: str) -> str:
    """获取当前时间的工具函数"""
    return f"当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def calculate_simple(expression: str) -> str:
    """简单计算器工具"""
    try:
        # 安全的数学表达式计算
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        else:
            return "错误：包含不允许的字符"
    except Exception as e:
        return f"计算错误：{str(e)}"

# LangChain 特点：统一的工具接口定义
tools = [
    Tool(
        name="get_time",
        func=get_current_time,
        description="获取当前的日期和时间信息"
    ),
    Tool(
        name="calculator",
        func=calculate_simple,
        description="执行简单的数学计算，如加减乘除运算"
    )
]

def demo_tools():
    """演示 Tools 工具系统"""
    print("=" * 50)
    print("🛠️ Tools 演示：工具系统")
    print("=" * 50)
    
    for tool in tools:
        print(f"工具名称：{tool.name}")
        print(f"工具描述：{tool.description}")
        
        # 测试工具
        if tool.name == "get_time":
            result = tool.run("现在几点了？")
        else:
            result = tool.run("10 + 5 * 2")
        
        print(f"工具输出：{result}\n")
#以下是输出
"""
==================================================
🛠️ Tools 演示：工具系统
==================================================
工具名称：get_time
工具描述：获取当前的日期和时间信息
工具输出：当前时间是：2026-04-14 19:18:56

工具名称：calculator
工具描述：执行简单的数学计算，如加减乘除运算
工具输出：计算结果：10 + 5 * 2 = 20
"""        

# ========== 4. 简化版 Agents：手动工具选择演示 ==========
def demo_simple_agents():
    """
    演示简化版 Agents：手动工具选择和执行
    LangChain 特点：工具集成和智能选择（这里用简化版演示概念）
    """
    print("=" * 50)
    print("🤖 简化版 Agents 演示：工具选择与执行")
    print("=" * 50)
    
    # 创建工具选择提示词
    tool_selection_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，可以使用以下工具：
        1. get_time - 获取当前时间
        2. calculator - 执行数学计算
        
        请分析用户问题，选择合适的工具并说明原因。
        只回答工具名称和原因，格式：工具名称|原因"""),
        ("human", "{question}")
    ])
    
    tool_chain = tool_selection_prompt | llm
    
    test_questions = [
        "现在几点了？",
        "帮我计算 15 * 8 + 20",
        "今天是什么日期？"
    ]
    
    for question in test_questions:
        print(f"👤 用户问题：{question}")
        
        # 1. 工具选择
        selection_result = tool_chain.invoke({"question": question})
        print(f"🧠 工具选择：{selection_result.content}")
        
        # 2. 执行工具（简化版手动执行）
        if "get_time" in selection_result.content.lower():
            result = get_current_time(question)
        elif "calculator" in selection_result.content.lower():
            # 提取数学表达式（简化处理）
            if "15 * 8 + 20" in question:
                result = calculate_simple("15 * 8 + 20")
            else:
                result = "需要具体的数学表达式"
        else:
            result = "未找到合适的工具"
        
        print(f"🛠️ 工具执行结果：{result}\n")

#输出
"""
==================================================
🤖 简化版 Agents 演示：工具选择与执行
==================================================
👤 用户问题：现在几点了？
🧠 工具选择：get_time|用户询问当前时间，需要获取实时时间信息。
🛠️ 工具执行结果：当前时间是：2026-04-14 19:26:43

👤 用户问题：帮我计算 15 * 8 + 20
🧠 工具选择：calculator|用户需要进行数学计算，涉及乘法和加法运算
🛠️ 工具执行结果：计算结果：15 * 8 + 20 = 140

👤 用户问题：今天是什么日期？
🧠 工具选择：get_time|用户询问当前日期，需要获取当前时间信息。
🛠️ 工具执行结果：当前时间是：2026-04-14 19:26:47

"""

# ========== 5. Memory：记忆系统 ==========
def demo_memory():
    """
    演示 Memory：对话记忆管理
    LangChain 特点：自动管理对话历史
    """
    print("=" * 50)
    print("🧠 Memory 演示：记忆系统")
    print("=" * 50)
    
    # 使用简化的记忆管理方式
    conversation_history = []
    
    # 创建带记忆的对话提示词
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，能够记住之前的对话内容。以下是对话历史：{history}"),
        ("human", "{input}")
    ])
    
    memory_chain = memory_prompt | llm
    
    # 模拟多轮对话
    conversations = [
        "我叫张三，今年25岁",
        "我喜欢编程和阅读",
        "你还记得我的名字吗？",
        "我的爱好是什么？"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"👤 第{i}轮对话：{user_input}")
        
        # 构建历史记录字符串
        history_str = "\n".join([f"用户: {h['user']}\n助手: {h['assistant']}" for h in conversation_history])
        
        # 获取回复 
        response = memory_chain.invoke({
            "history": history_str,
            "input": user_input
        })
        
        print(f"🤖 助手回复：{response.content}\n")
        
        # 更新对话历史
        conversation_history.append({
            "user": user_input,
            "assistant": response.content
        })
        
        # 显示当前记忆内容
        print(f"💭 当前记忆：{len(conversation_history)} 轮对话")
        print("-" * 30)
#输出
"""
==================================================
🧠 Memory 演示：记忆系统
==================================================
👤 第1轮对话：我叫张三，今年25岁
🤖 助手回复：你好张三！很高兴认识你。25岁正是充满活力和可能性的年纪呢！有什么我可以帮你的吗？

💭 当前记忆：1 轮对话
------------------------------
👤 第2轮对话：我喜欢编程和阅读
🤖 助手回复：很高兴认识你，张三！编程和阅读都是很棒的兴趣爱好呢。你平时主要喜欢读什么类型的书？或者最近在学什么编程语言吗？

💭 当前记忆：2 轮对话
------------------------------
👤 第3轮对话：你还记得我的名字吗？
🤖 助手回复：当然记得！张三，25岁，喜欢编程和阅读。需要我帮你推荐一些技术书籍，或者聊聊最近的编程项目吗？ 😊

💭 当前记忆：3 轮对话
------------------------------
👤 第4轮对话：我的爱好是什么？
🤖 助手回复：张三，根据之前的对话，你的爱好是**编程和阅读**！需要我为你推荐一些编程相关的书籍，或者聊聊你最近在读什么书吗？ 😊

💭 当前记忆：4 轮对话
------------------------------
"""        

# ========== 6. LCEL 演示：LangChain Expression Language ==========
def demo_lcel():
    """
    演示 LCEL：LangChain 0.3 的新特性
    LangChain 特点：更简洁的链式组合语法
    """
    print("=" * 50)
    print("🔗 LCEL 演示：LangChain Expression Language")
    print("=" * 50)
    
    # LCEL 语法：使用 | 操作符组合组件
    prompt = PromptTemplate.from_template("请用{language}语言解释什么是{concept}")
    
    # 创建链：prompt | llm
    chain = prompt | llm
    
    # 执行链
    result = chain.invoke({
        "language": "简单易懂的中文",
        "concept": "区块链"
    })
    
    print(f"📝 LCEL 链式调用结果：\n{result.content}\n")
    
    # 演示更复杂的链组合
    from langchain_core.output_parsers import StrOutputParser
    
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 更复杂的链：prompt | llm | output_parser
    complex_chain = prompt | llm | output_parser
    
    result2 = complex_chain.invoke({
        "language": "技术术语",
        "concept": "机器学习"
    })
    
    print(f"📝 复杂 LCEL 链结果：\n{result2}\n")

# ========== 7. 综合演示：LangChain 特点总结 ==========
def demo_langchain_features():
    """展示 LangChain 的核心特点"""
    print("=" * 60)
    print("🌟 LangChain 核心特点总结")
    print("=" * 60)
    
    features = [
        "🔗 链式组合：使用 LCEL (|) 将多个组件串联",
        "📝 模板管理：统一的提示词模板系统",
        "🛠️ 工具集成：标准化的工具接口",
        "🤖 智能代理：自动选择和使用工具（需模型支持）",
        "🧠 记忆管理：灵活的对话历史管理",
        "🔄 流程编排：灵活的工作流定义",
        "📊 可观测性：详细的执行日志",
        "🔌 模块化：组件可插拔设计",
        "⚡ LCEL：简洁的表达式语言",
        "🎯 类型安全：完整的类型提示支持"
    ]
    
    for feature in features:
        print(feature)
    
    print("\n" + "=" * 60)

def main():
    """主函数：依次演示各个核心组件"""
    print("🚀 LangChain 0.3 核心组件实战演示")
    print("基于 OpenAI API 的完整示例（兼容版本）\n")
    
    try:
        # 1. LLMChain 演示（使用 LCEL）
        demo_llm_chain()
        
        # 2. Tools 演示
        demo_tools()
        
        # 3. 简化版 Agents 演示
        demo_simple_agents()
        
        # 4. Memory 演示
        demo_memory()
        
        # 5. LCEL 演示
        demo_lcel()
        
        # 6. 特点总结
        demo_langchain_features()
        
        print("✅ 所有演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误：{str(e)}")
        print("请检查 API 密钥和网络连接")

if __name__ == "__main__":
    main()

"""
🚀 LangChain 0.3 核心组件实战演示
基于 OpenAI API 的完整示例（兼容版本）








==================================================
🧠 Memory 演示：记忆系统
==================================================
👤 第1轮对话：我叫张三，今年25岁
🤖 助手回复：你好张三！很高兴认识你。25岁正是充满活力和可能性的年纪呢！有什么我可以帮你的吗？

💭 当前记忆：1 轮对话
------------------------------
👤 第2轮对话：我喜欢编程和阅读
🤖 助手回复：张三，你对编程和阅读的兴趣听起来很棒！这两者都是能让人不断成长和探索的领域。你最近在读什么书，或者在做什么编程项目吗？

💭 当前记忆：2 轮对话
------------------------------
👤 第3轮对话：你还记得我的名字吗？
🤖 助手回复：当然记得！张三，你之前提到过自己25岁，喜欢编程和阅读。需要我帮你推荐一些编程相关的书籍，或者聊聊最近的技术趋势吗？ 😊

💭 当前记忆：3 轮对话
------------------------------
👤 第4轮对话：我的爱好是什么？
🤖 助手回复：张三，你之前提到过你喜欢**编程和阅读**！这两项爱好都很棒呢～  
最近有在读什么有趣的书，或者在尝试新的编程语言/项目吗？ 😄

💭 当前记忆：4 轮对话
------------------------------
==================================================
🔗 LCEL 演示：LangChain Expression Language
==================================================
📝 LCEL 链式调用结果：
好的，我用一个简单的比喻来解释区块链。

想象一下，你们**全班同学一起维护一个公开的、不可篡改的“班级记账本”**，这个记账本记录着所有同学之间的积分交易（比如A给了B 10个积分）。

---

### 核心概念拆解：

**1. “区块”是什么？**
   *   这个记账本不是一页纸，而是**一页一页装订起来的**。
   *   **每一页纸就是一个“区块”**。这一页上会按顺序记录最近10分钟内发生的所有积分交易（例如：小明给小红5分，小刚给小芳3分……）。
   *   当这一页写满了（或时间到了），我们就把它用订书机订到之前的那一页后面，形成一条链。

**2. “链”是怎么形成的？**
   *   关键来了！在把新的一页（新区块）订上去之前，我们会在这页的**页眉处，抄写上一页的“特征码”（一种特殊的数字指纹，叫哈希值）**。
   *   同时，我们还会根据这一页**具体写了什么交易内容**，生成一个属于这一页自己的“特征码”，写在页脚。
   *   **下一页的页眉又会抄写这一页页脚的特征码**……如此环环相扣，就像用“特征码”这个锁链把一页一页紧紧锁在了一起。
   *   **如果有人想偷偷修改其中某一页的某条交易记录**，那么这一页的“特征码”就会立刻改变。而下一页的页眉还记录着旧的“特征码”，就对不上了！大家一眼就能发现账本被动了手脚。要想掩盖，就必须把之后所有的页都重写一遍，这几乎是不可能的。

**3. 这个记账本由谁来写和维护？（去中心化）**
   *   这个记账本**不是由班长或老师一个人保管和记录的**，而是**全班每位同学都有一本一模一样的副本**。
   *   当有新的交易发生时（比如“小明给小红5分”），小明会向全班广播这个消息。
   *   听到消息的同学们（称为“节点”）会拿出自己的副本，验证这笔交易是否有效（小明是不是真的有5分可以给？）。
   *   验证通过后，大家会把这条交易记录暂时放进自己的“待记录清单”里。每隔一段时间（比如10分钟），就会有同学（通过一种公平的竞争机制，如“解题比赛”）被选出来，负责把这段时间的“待记录清单”整理好，写成一页新的“区块”，然后广播给全班。
   *   **全班同学收到这新的一页后，会再次进行集体核对**。如果大多数人核对无误，就会各自把这一页订到自己的记账本上。于是，所有人的账本又同步更新了，完全一致。

---

### 总结一下区块链的特点：

*   **去中心化**：没有唯一的总负责人，账本由网络中的所有人共同维护和备份。
*   **透明可追溯**：所有交易记录对所有人公开（虽然交易者身份可能是匿名的代号），可以追溯任何一笔交易的来龙去脉。
*   **不可篡改**：一旦信息经过验证并被添加到区块链上，就会被永久存储。由于链式结构和集体维护，要修改一个区块的数据，必须同时修改之后所有区块，并控制超过50%的网络节点，这成本极高，几乎无法实现。
*   **集体维护**：更新的规则由大家共识决定，任何单个节点都无法擅自控制或修改数据。

### 它有什么用？（除了记账积分）

这个“分布式记账本”技术可以记录任何有价值的信息：
*   **数字货币**：比如比特币，就是记录“谁给谁转了多少钱”。
*   **电子合同**：把合同条款写在区块链上，到期自动执行，无法抵赖。
*   **物流溯源**：记录商品从生产到销售的每一个环节，确保来源真实。
*   **房产/版权登记**：明确资产的所有权归属。

**简单说，区块链就是一个由大家共同维护、按时间顺序环环相扣的“神奇账本”，它的核心目的是在互不信任的网络中，建立一种无需中间担保人的信任机制。**

📝 复杂 LCEL 链结果：
机器学习（Machine Learning，ML）是人工智能（AI）的核心分支，专注于通过算法和统计模型使计算机系统能够从数据中自动“学习”并改进性能，而无需显式编程。其核心思想是通过经验（数据）优化模型参数，从而实现对未知数据的泛化能力。以下是技术层面的关键概念解析：

---

### **1. 核心范式：基于数据驱动的归纳学习**
- **传统编程 vs. 机器学习**：
  - 传统编程：输入数据 + 明确规则 → 输出结果。
  - 机器学习：输入数据 + 预期输出 → 学习隐含规则（模型），用于新数据的预测或决策。

---

### **2. 技术要素**
#### **a. 数据与特征工程**
- **数据集划分**：训练集（模型学习）、验证集（调参）、测试集（最终评估）。
- **特征（Feature）**：数据的可量化属性，特征工程包括提取、选择、转换（如归一化、降维），直接影响模型性能。

#### **b. 模型与算法**
- **模型**：数学函数 \( f: X \rightarrow Y \)，映射输入特征 \( X \) 到输出 \( Y \)。
- **损失函数（Loss Function）**：量化模型预测误差（如均方误差、交叉熵），指导参数优化。
- **优化算法**：通过梯度下降等迭代调整参数，最小化损失函数。

#### **c. 学习类型**
- **监督学习（Supervised Learning）**：使用带标签数据学习输入到输出的映射。
  - 任务：分类（离散输出）、回归（连续输出）。
  - 算法：线性回归、决策树、支持向量机（SVM）、神经网络。
- **无监督学习（Unsupervised Learning）**：从无标签数据中发现模式。
  - 任务：聚类（如K-means）、降维（如PCA）、异常检测。
- **强化学习（Reinforcement Learning）**：智能体通过与环境交互的奖励信号学习策略（如Q-learning、深度强化学习）。

#### **d. 泛化与正则化**
- **过拟合（Overfitting）**：模型过度拟合训练数据噪声，导致在新数据上性能下降。
- **正则化技术**：L1/L2正则化、Dropout（神经网络）等，约束模型复杂度以提升泛化能力。

---

### **3. 关键技术流程**
1. **问题定义**：明确任务类型（如分类、回归）和评估指标（准确率、F1分数、RMSE）。
2. **数据预处理**：清洗、缺失值处理、特征标准化。
3. **模型选择**：基于数据规模和问题复杂度选择算法（如小数据可用SVM，大数据用深度学习）。
4. **训练与验证**：通过交叉验证避免过拟合，调整超参数（如学习率、网络层数）。
5. **部署与监控**：模型部署至生产环境，持续监控性能衰减（数据漂移）。

---

### **4. 典型技术栈**
- **框架**：TensorFlow、PyTorch（深度学习）、Scikit-learn（传统ML）。
- **部署工具**：Docker、Kubernetes、TensorFlow Serving。
- **评估工具**：混淆矩阵、ROC曲线、SHAP（可解释性分析）。

---

### **5. 前沿方向**
- **深度学习**：通过多层神经网络学习高维数据表征（如图像、自然语言）。
- **自监督学习**：从无标签数据自动生成监督信号，减少对标注数据的依赖。
- **联邦学习**：分布式训练保护数据隐私。
- **因果机器学习**：超越相关性，探索变量间的因果机制。

---

### **技术本质总结**
机器学习本质是**通过数据驱动的方式逼近真实世界的未知函数**，其数学基础涵盖概率论、优化理论、统计学习理论（如VC维、偏差-方差权衡）。在实际应用中，需平衡模型复杂性、计算成本与业务需求，实现从数据到知识的自动化提取。

============================================================
🌟 LangChain 核心特点总结
============================================================
🔗 链式组合：使用 LCEL (|) 将多个组件串联
📝 模板管理：统一的提示词模板系统
🛠️ 工具集成：标准化的工具接口
🤖 智能代理：自动选择和使用工具（需模型支持）
🧠 记忆管理：灵活的对话历史管理
🔄 流程编排：灵活的工作流定义
📊 可观测性：详细的执行日志
🔌 模块化：组件可插拔设计
⚡ LCEL：简洁的表达式语言
🎯 类型安全：完整的类型提示支持

============================================================
✅ 所有演示完成！
"""    
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.openai_like import OpenAILike
import os
import glob
from typing import List, Union
from paddleocr import PaddleOCR


class ImageOCRReader(BaseReader):
    """使用 PP-OCR 从图像中提取文本并返回 Document"""

    def __init__(self, lang='ch', **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        super().__init__()
        self.lang = lang
        #self.use_gpu = use_gpu
        self.kwargs = kwargs

        # 延迟初始化，首次调用 load_data 时才创建（避免导入时就加载模型）
        self._ocr = None

    def _get_ocr(self):
        """延迟初始化 PaddleOCR"""
        if self._ocr is None:
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                **self.kwargs,
            )
        return self._ocr

    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表

        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        # 统一为列表
        if isinstance(file, str):
            files = [file]
        elif isinstance(file, list):
            files = file
        else:
            raise ValueError(f"file 参数应为 str 或 List[str]， got {type(file)}")

        ocr = self._get_ocr()
        docs = []

        for filepath in files:
            if not os.path.exists(filepath):
                print(f"⚠️ 文件不存在，跳过: {filepath}")
                continue

            print(f"🔍 正在识别: {os.path.basename(filepath)}")

            # 调用 PaddleOCR
            results = ocr.predict(filepath)

            # 提取文本和置信度
            text_blocks = []
            total_confidence = 0.0
            total_blocks = 0
            for res in results:
                total_blocks = len(res['rec_texts'])
                for idx, text in enumerate(res['rec_texts']):
                    # item 结构: (文本, 置信度, 坐标)
                    confidence = res["rec_scores"][idx]
                    text_blocks.append(f"[Text Block {idx + 1}] (conf: {confidence:.2f}): {text}")
                    total_confidence += confidence

            # 拼接为完整文本
            full_text = "\n".join(text_blocks) if text_blocks else ""

            # 计算平均置信度
            avg_confidence = total_confidence / total_blocks if total_blocks > 0 else 0.0

            # 构造 Document
            doc = Document(
                text=full_text,
                metadata={
                    "image_path": filepath,
                    "ocr_model": "PP-OCRv5",
                    "language": self.lang,
                    "num_text_blocks": total_blocks,
                    "avg_confidence": round(avg_confidence, 4),
                },
            )
            docs.append(doc)

            print(f"   ✅ 识别完成: {total_blocks} 个文本块, 平均置信度: {avg_confidence:.4f}")

        return docs

    def load_data_from_dir(self, dir_path: str) -> List[Document]:
        """批量加载目录下所有图像文件"""
        supported_ext = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]
        files = []
        for ext in supported_ext:
            files.extend(glob.glob(os.path.join(dir_path, ext)))
            files.extend(glob.glob(os.path.join(dir_path, ext.upper())))

        print(f"📁 在目录中找到 {len(files)} 个图像文件")
        return self.load_data(files)


def main():
    """主流程"""
    print("=" * 60)
    print("  🚀 LlamaIndex OCR 图像文本加载器实验")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- 步骤1: 使用 ImageOCRReader 批量加载 imgs 目录下所有图像 ----
    print("\n📌 步骤1: OCR 识别图像")
    print("-" * 40)

    reader = ImageOCRReader(lang='ch')

    # 批量加载 imgs 目录下所有图像文件
    imgs_dir = os.path.join(base_dir, "imgs")

    if not os.path.isdir(imgs_dir):
        print(f"❌ 图像目录不存在: {imgs_dir}")
        print("提示: 请在 ocr_research 目录下创建 imgs 文件夹并放入测试图像")
        return

    documents = reader.load_data_from_dir(imgs_dir)

    if not documents:
        print("❌ 未能从图像中提取到文本")
        return

    # 打印 OCR 结果
    for doc in documents:
        print(f"\n📄 文件: {doc.metadata['image_path']}")
        print(f"   文本块数: {doc.metadata['num_text_blocks']}")
        print(f"   平均置信度: {doc.metadata['avg_confidence']}")
        print(f"\n--- 识别文本 ---")
        print(doc.text)

    # ---- 步骤2: 构建 LlamaIndex 索引 ----
    print("\n\n📌 步骤2: 构建 LlamaIndex 向量索引")
    print("-" * 40)

    # 设置 Embedding 模型（用于向量索引）
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        Settings.embed_model = DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
            embed_batch_size=6,
            embed_input_length=8192,
        )
        Settings.llm = OpenAILike(
            model="qwen-plus",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            is_chat_model=True,
        )
        print("✅ 已配置 DashScope Embedding 和 qwen-plus LLM")
    else:
        print("⚠️ 未设置 DASHSCOPE_API_KEY，使用默认本地嵌入")

    index = VectorStoreIndex.from_documents(documents)
    print("✅ 向量索引构建完成")

    # ---- 步骤3: 查询验证 ----
    print("\n\n📌 步骤3: 查询验证")
    print("-" * 40)

    if api_key:
        query_engine = index.as_query_engine(similarity_top_k=3)

        test_queries = [
            "图片中提到了什么日期？",
            "图片中的主要文字内容是什么？",
        ]

        for query in test_queries:
            print(f"\n❓ 问题: {query}")
            response = query_engine.query(query)
            print(f"💡 回答: {response}")
    else:
        print("⚠️ 未配置 API Key，跳过 LLM 查询测试")
        print("   提示: export DASHSCOPE_API_KEY=你的key 后重新运行")

    # ---- 步骤4: 保存结果 ----
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 保存识别文本
    for doc in documents:
        txt_name = os.path.splitext(os.path.basename(doc.metadata["image_path"]))[0] + "_ocr.txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(doc.text)
        print(f"\n💾 识别结果已保存到: {txt_path}")

    print("\n" + "=" * 60)
    print("  ✅ 实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

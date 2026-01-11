"""
基于Qwen-2.5的RAG系统 for RTCA DO-160G标准文档
支持：多轮对话、长上下文、引用显示、拒绝不确定回答
作者：AI助手
版本：1.0
"""

# ==================== 1. 环境设置和依赖安装 ====================
# requirements.txt
"""
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
langchain>=0.0.340
gradio>=4.0.0
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
chromadb>=0.4.0
accelerate>=0.24.0
peft>=0.6.0
evaluate>=0.4.0
rouge-score>=0.1.2
pdfplumber>=0.10.0
python-docx>=1.1.0
rank_bm25>=0.2.2
tiktoken>=0.5.0
"""

# ==================== 2. 配置管理 ====================
import yaml
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import os
import re
from typing import List, Tuple
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, LoraConfig, get_peft_model
from typing import List, Dict, Tuple
import datasets
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import gradio as gr
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.metrics import precision_recall_fscore_support
import evaluate


class RetrievalStrategy(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANK = "rerank"


@dataclass
class RAGConfig:
    """RAG系统配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 检索配置
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    rerank_top_n: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 生成配置
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # 微调配置
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # 路径配置
    knowledge_base_path: str = "./data/knowledge_base"
    vector_db_path: str = "./data/vector_db"
    fine_tune_data_path: str = "./data/fine_tune"

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ==================== 3. 文档处理模块 ====================


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )

    def load_pdf(self, file_path: str) -> List[Dict]:
        """加载PDF文档"""
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # 提取章节信息
                    chapter_match = re.search(r'第(\d+)章\s+(.+)', text[:100])
                    chapter_info = {
                        'chapter': chapter_match.group(1) if chapter_match else str(page_num),
                        'title': chapter_match.group(2) if chapter_match else f"第{page_num}页",
                        'page': page_num
                    }

                    documents.append({
                        'content': text,
                        'metadata': {
                            'source': os.path.basename(file_path),
                            'page': page_num,
                            'chapter_info': chapter_info
                        }
                    })
        return documents

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """文档分块"""
        chunks = []
        for doc in documents:
            text_chunks = self.text_splitter.split_text(doc['content'])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'content': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': i,
                        'start_char': i * self.config.chunk_size
                    }
                })
        return chunks

    def process_directory(self, dir_path: str) -> List[Dict]:
        """处理整个目录的文档"""
        all_chunks = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    documents = self.load_pdf(file_path)
                    chunks = self.chunk_documents(documents)
                    all_chunks.extend(chunks)

        # 保存处理后的文档
        self.save_chunks(all_chunks, os.path.join(
            self.config.knowledge_base_path, "processed_chunks.json"))
        return all_chunks

    @staticmethod
    def save_chunks(chunks: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


# ==================== 4. 向量数据库模块 ====================


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def create_vector_store(self, chunks: List[Dict]) -> FAISS:
        """创建向量数据库"""
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )

        # 保存向量数据库
        self.vector_store.save_local(self.config.vector_db_path)
        return self.vector_store

    def load_vector_store(self) -> FAISS:
        """加载向量数据库"""
        self.vector_store = FAISS.load_local(
            self.config.vector_db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        return self.vector_store

    def update_vector_store(self, new_chunks: List[Dict]):
        """更新向量数据库"""
        if self.vector_store is None:
            self.load_vector_store()

        texts = [chunk['content'] for chunk in new_chunks]
        metadatas = [chunk['metadata'] for chunk in new_chunks]

        self.vector_store.add_texts(texts, metadatas)
        self.vector_store.save_local(self.config.vector_db_path)


# ==================== 5. 检索模块 ====================


class HybridRetriever:
    """混合检索器"""

    def __init__(self, vector_store: FAISS, chunks: List[Dict], config: RAGConfig):
        self.vector_store = vector_store
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)

        # 初始化BM25
        self.chunks = chunks
        self.chunk_texts = [chunk['content'] for chunk in chunks]
        self.bm25 = BM25Okapi([self.tokenize(text)
                              for text in self.chunk_texts])

        # 初始化重排序模型
        self.rerank_model = None
        if config.retrieval_strategy == RetrievalStrategy.RERANK:
            self.init_rerank_model()

    def tokenize(self, text: str) -> List[str]:
        """文本分词"""
        return self.tokenizer.tokenize(text)

    def init_rerank_model(self):
        """初始化重排序模型"""
        from sentence_transformers import CrossEncoder
        self.rerank_model = CrossEncoder('BAAI/bge-reranker-large')

    def dense_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """稠密检索"""
        docs = self.vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'retrieval_type': 'dense'
            })
        return results

    def sparse_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """稀疏检索（BM25）"""
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunk_texts[idx],
                'metadata': self.chunks[idx]['metadata'],
                'score': float(scores[idx]),
                'retrieval_type': 'sparse'
            })
        return results

    def hybrid_retrieve(self, query: str, top_k: int, alpha: float = 0.5) -> List[Dict]:
        """混合检索"""
        dense_results = self.dense_retrieve(query, top_k * 2)
        sparse_results = self.sparse_retrieve(query, top_k * 2)

        # 合并结果并去重
        all_results = {}
        for result in dense_results + sparse_results:
            content = result['content']
            if content not in all_results:
                all_results[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'dense_score': 0.0,
                    'sparse_score': 0.0,
                    'combined_score': 0.0
                }

            if result['retrieval_type'] == 'dense':
                all_results[content]['dense_score'] = result['score']
            else:
                all_results[content]['sparse_score'] = result['score']

        # 归一化分数并计算综合分数
        max_dense = max(r['dense_score'] for r in all_results.values()) or 1
        max_sparse = max(r['sparse_score'] for r in all_results.values()) or 1

        for content in all_results:
            all_results[content]['dense_score_norm'] = all_results[content]['dense_score'] / max_dense
            all_results[content]['sparse_score_norm'] = all_results[content]['sparse_score'] / max_sparse
            all_results[content]['combined_score'] = (
                alpha * all_results[content]['dense_score_norm'] +
                (1 - alpha) * all_results[content]['sparse_score_norm']
            )

        # 按综合分数排序
        sorted_results = sorted(all_results.values(),
                                key=lambda x: x['combined_score'],
                                reverse=True)[:top_k]

        return [{
            'content': r['content'],
            'metadata': r['metadata'],
            'score': r['combined_score'],
            'retrieval_type': 'hybrid'
        } for r in sorted_results]

    def retrieve_with_rerank(self, query: str, top_k: int) -> List[Dict]:
        """带重排序的检索"""
        # 第一阶段：混合检索获取更多候选
        candidate_results = self.hybrid_retrieve(query, top_k * 3)

        # 第二阶段：重排序
        if self.rerank_model:
            pairs = [(query, r['content']) for r in candidate_results]
            rerank_scores = self.rerank_model.predict(pairs)

            for result, score in zip(candidate_results, rerank_scores):
                result['rerank_score'] = float(score)

            # 按重排序分数排序
            candidate_results.sort(
                key=lambda x: x['rerank_score'], reverse=True)

        return candidate_results[:top_k]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """主检索方法"""
        if top_k is None:
            top_k = self.config.top_k

        strategy = self.config.retrieval_strategy

        if strategy == RetrievalStrategy.DENSE:
            return self.dense_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.SPARSE:
            return self.sparse_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            return self.hybrid_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.RERANK:
            return self.retrieve_with_rerank(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")


# ==================== 6. 模型管理模块 ====================


class QwenModelManager:
    """Qwen模型管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.lora_config = None

    def load_base_model(self):
        """加载基础模型"""
        print(f"正在加载模型: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.config.use_lora:
            self.apply_lora()

        # 创建text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.config.device == "cuda" else -1
        )

        return self.model, self.tokenizer

    def apply_lora(self):
        """应用LoRA适配器"""
        self.lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj",
                            "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, self.lora_config)
        print("LoRA适配器已加载")

    def load_lora_adapter(self, adapter_path: str):
        """加载预训练的LoRA适配器"""
        if self.model is None:
            self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload()
        print(f"LoRA适配器已从 {adapter_path} 加载")

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        if self.pipeline is None:
            self.load_base_model()

        # 合并生成参数
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        gen_kwargs.update(kwargs)

        # 生成
        outputs = self.pipeline(prompt, **gen_kwargs)
        return outputs[0]['generated_text'][len(prompt):]

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话生成"""
        if self.pipeline is None:
            self.load_base_model()

        # 构建对话格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return self.generate(text, **kwargs)


# ==================== 7. RAG系统核心 ====================


class RAGSystem:
    """RAG系统核心"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model_manager = QwenModelManager(config)
        self.retriever = None
        self.conversation_history = []

        # 加载系统prompt
        self.system_prompt = self.load_system_prompt()

        # 不确定性检测关键词
        self.uncertain_keywords = [
            "我不确定", "我不知道", "无法确定", "没有找到", "未提及",
            "可能", "大概", "或许", "似乎", "应该"
        ]

    def load_system_prompt(self) -> str:
        """加载系统prompt"""
        return """你是一个航空标准RTCA DO-160G的专家助手。请基于提供的参考文档回答问题。
        回答要求：
        1. 准确引用文档中的具体章节和内容
        2. 如果问题超出文档范围，明确告知用户
        3. 对于不确定的内容，不要猜测，要承认不知道
        4. 回答要专业、准确、清晰
        
        当前文档：RTCA DO-160G 机载设备环境条件和试验程序"""

    def format_references(self, retrieved_docs: List[Dict]) -> str:
        """格式化引用信息"""
        references = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc['metadata']
            chapter_info = meta.get('chapter_info', {})
            references.append(
                f"[{i}] 来源：{meta.get('source', '未知')}，"
                f"章节：第{chapter_info.get('chapter', '未知')}章 {chapter_info.get('title', '')}，"
                f"页码：{meta.get('page', '未知')}，"
                f"相关性分数：{doc['score']:.3f}"
            )
        return "\n".join(references)

    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """构建prompt"""
        # 构建上下文
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        references = self.format_references(retrieved_docs)

        # 构建完整prompt
        prompt = f"""{self.system_prompt}

相关参考文档：
{context}

参考文档的详细来源信息：
{references}

用户问题：{query}

请基于以上参考文档回答问题，并在适当位置引用文档来源（如[1][2]）。如果文档中没有相关信息，请明确说明。

回答："""

        return prompt

    def detect_uncertainty(self, response: str) -> bool:
        """检测回答中的不确定性"""
        # 简单的关键词检测
        for keyword in self.uncertain_keywords:
            if keyword in response:
                return True

        # 检查是否有引用
        if not re.search(r'\[\d+\]', response):
            # 没有引用可能意味着不确定
            return True

        return False

    def retrieve_documents(self, query: str) -> List[Dict]:
        """检索相关文档"""
        if self.retriever is None:
            raise ValueError("检索器未初始化")
        return self.retriever.retrieve(query)

    def answer(self, query: str, conversation_id: str = None) -> Dict:
        """回答问题"""
        # 检索相关文档
        retrieved_docs = self.retrieve_documents(query)

        if not retrieved_docs:
            return {
                'answer': "抱歉，在文档中没有找到相关信息。",
                'references': [],
                'confidence': 0.0,
                'uncertain': True
            }

        # 构建prompt
        prompt = self.build_prompt(query, retrieved_docs)

        # 生成回答
        response = self.model_manager.generate(prompt)

        # 提取引用
        citations = re.findall(r'\[(\d+)\]', response)
        cited_docs = []
        for cite in citations:
            try:
                idx = int(cite) - 1
                if 0 <= idx < len(retrieved_docs):
                    cited_docs.append(retrieved_docs[idx])
            except:
                pass

        # 检测不确定性
        uncertain = self.detect_uncertainty(response)

        # 计算置信度（基于检索分数）
        avg_score = sum(doc['score']
                        for doc in cited_docs) / max(len(cited_docs), 1)
        confidence = min(avg_score * 10, 1.0)  # 归一化到0-1

        # 更新对话历史
        if conversation_id:
            self.update_conversation_history(conversation_id, query, response)

        return {
            'answer': response,
            'references': cited_docs,
            'confidence': confidence,
            'uncertain': uncertain,
            'retrieved_docs': retrieved_docs
        }

    def update_conversation_history(self, conv_id: str, query: str, response: str):
        """更新对话历史"""
        if conv_id not in self.conversation_history:
            self.conversation_history[conv_id] = []

        self.conversation_history[conv_id].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

        # 限制历史长度
        if len(self.conversation_history[conv_id]) > 10:
            self.conversation_history[conv_id] = self.conversation_history[conv_id][-10:]


# ==================== 8. 微调模块 ====================


class FineTuner:
    """模型微调器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def prepare_dataset(self, data_path: str):
        """准备训练数据集"""
        dataset = datasets.load_from_disk(data_path)

        def tokenize_function(examples):
            # 构建训练格式
            prompts = []
            for context, question, answer in zip(examples['context'],
                                                 examples['question'],
                                                 examples['answer']):
                prompt = f"""基于以下文档回答问题：

文档内容：
{context}

问题：{question}

答案：{answer}
"""
                prompts.append(prompt)

            return self.tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, train_dataset, eval_dataset=None):
        """训练模型"""
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # 应用LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        # 训练参数
        training_args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_strategy="steps",
            save_steps=100,
            learning_rate=2e-4,
            fp16=True,
            push_to_hub=False
        )

        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")

        return trainer


# ==================== 9. Gradio Web界面 ====================


class GradioApp:
    """Gradio Web应用"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.conversations = {}  # conversation_id -> history

    def chat_interface(self, message: str, history: list, conversation_id: str):
        """聊天界面"""
        if not conversation_id:
            conversation_id = str(int(time.time()))

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # 获取回答
        result = self.rag_system.answer(message, conversation_id)

        # 格式化回答
        response = result['answer']
        if result['references']:
            response += "\n\n**参考来源：**\n"
            for i, ref in enumerate(result['references'], 1):
                meta = ref['metadata']
                response += f"{i}. {meta.get('source', '未知')} - 第{meta.get('page', '未知')}页\n"

        if result['uncertain']:
            response = "⚠️ **注意：** 这个回答可能不完全准确，建议核实官方文档。\n\n" + response

        # 更新历史
        self.conversations[conversation_id].append((message, response))

        return "", history + [(message, response)]

    def create_web_app(self):
        """创建Web应用"""
        with gr.Blocks(title="RTCA DO-160G专家助手", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🛩️ RTCA DO-160G专家助手")
            gr.Markdown("基于Qwen-2.5的航空标准文档智能问答系统")

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：第4章的温度试验要求是什么？",
                        lines=2
                    )
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空对话")

                    conv_id = gr.Textbox(
                        label="会话ID（可选）",
                        placeholder="留空将创建新会话",
                        lines=1
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 系统信息")
                    confidence_bar = gr.Label("置信度: 待计算")
                    retrieval_stats = gr.Label("检索文档数: 0")
                    model_info = gr.Label(
                        f"模型: {self.rag_system.config.model_name}")

                    gr.Markdown("### ⚙️ 设置")
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="检索文档数量"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                        label="生成温度"
                    )

                    gr.Markdown("### 📈 性能指标")
                    latency_display = gr.Label("响应时间: -")

            # 事件处理
            msg.submit(
                self.chat_interface,
                [msg, chatbot, conv_id],
                [msg, chatbot]
            )

            submit_btn.click(
                self.chat_interface,
                [msg, chatbot, conv_id],
                [msg, chatbot]
            )

            clear_btn.click(lambda: None, None, chatbot, queue=False)

            # 更新设置
            def update_settings(top_k, temperature):
                self.rag_system.config.top_k = int(top_k)
                self.rag_system.config.temperature = temperature
                return "设置已更新"

            top_k_slider.change(
                update_settings, [top_k_slider, temp_slider], [])
            temp_slider.change(update_settings, [
                               top_k_slider, temp_slider], [])

        return app


# ==================== 10. FastAPI后端 ====================


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    references: List[Dict]
    confidence: float
    uncertain: bool
    latency: float


class FastAPIApp:
    """FastAPI后端应用"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.app = FastAPI(title="RAG API", version="1.0.0")
        self.setup_routes()

    def setup_routes(self):
        """设置路由"""

        @self.app.get("/")
        async def root():
            return {"message": "RAG API Service", "status": "running"}

        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest):
            start_time = time.time()

            # 临时更新配置
            if request.top_k:
                self.rag_system.config.top_k = request.top_k
            if request.temperature:
                self.rag_system.config.temperature = request.temperature

            # 获取回答
            result = self.rag_system.answer(
                request.query, request.conversation_id)

            latency = time.time() - start_time

            return QueryResponse(
                answer=result['answer'],
                references=result['references'],
                confidence=result['confidence'],
                uncertain=result['uncertain'],
                latency=latency
            )

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行API服务"""
        uvicorn.run(self.app, host=host, port=port)


# ==================== 11. 评估模块 ====================


class Evaluator:
    """系统评估器"""

    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')

    def evaluate_rag(self, predictions: List[str], references: List[str]) -> Dict:
        """评估RAG系统"""
        # ROUGE分数
        rouge_results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )

        # BLEU分数
        bleu_results = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )

        # 幻觉检测（简单版本）
        hallucination_rate = self.detect_hallucinations(
            predictions, references)

        return {
            'rouge': rouge_results,
            'bleu': bleu_results['bleu'],
            'hallucination_rate': hallucination_rate
        }

    def detect_hallucinations(self, predictions: List[str], references: List[str]) -> float:
        """检测幻觉率"""
        hallucination_count = 0
        for pred, ref in zip(predictions, references):
            # 简单的检测：如果预测包含大量不在参考中的实体
            pred_entities = set(re.findall(
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', pred))
            ref_entities = set(re.findall(
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', ref))

            # 检查是否存在大量未在参考中出现的实体
            novel_entities = pred_entities - ref_entities
            if len(novel_entities) > len(pred_entities) * 0.3:  # 30%的新实体
                hallucination_count += 1

        return hallucination_count / len(predictions) if predictions else 0

    def evaluate_citation(self, predictions: List[str], gold_citations: List[List[str]]) -> Dict:
        """评估引用质量"""
        precisions = []
        recalls = []

        for pred, gold in zip(predictions, gold_citations):
            # 提取预测中的引用
            pred_citations = re.findall(r'\[(\d+)\]', pred)

            if not gold:  # 如果没有金标准引用
                if not pred_citations:
                    precisions.append(1.0)
                    recalls.append(1.0)
                else:
                    precisions.append(0.0)
                    recalls.append(0.0)
            else:
                # 计算精度和召回率
                correct = len(set(pred_citations) & set(gold))
                precision = correct / \
                    len(pred_citations) if pred_citations else 0
                recall = correct / len(gold) if gold else 0

                precisions.append(precision)
                recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        f1 = 2 * avg_precision * avg_recall / \
            (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        return {
            'citation_precision': avg_precision,
            'citation_recall': avg_recall,
            'citation_f1': f1
        }

# ==================== 12. 主程序 ====================


def main():
    """主程序"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG系统主程序")
    parser.add_argument("--mode", choices=["init", "train", "web", "api", "eval"],
                        default="web", help="运行模式")
    parser.add_argument("--config", default="./config.json", help="配置文件路径")
    parser.add_argument("--data", default="./data/rtca_doc.pdf", help="文档路径")
    parser.add_argument("--host", default="0.0.0.0", help="API主机地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")

    args = parser.parse_args()

    # 加载配置
    config = RAGConfig.load(args.config)

    if args.mode == "init":
        # 初始化知识库
        print("正在初始化知识库...")
        processor = DocumentProcessor(config)
        chunks = processor.process_directory(args.data)

        vector_store_manager = VectorStoreManager(config)
        vector_store = vector_store_manager.create_vector_store(chunks)

        print(f"知识库初始化完成，共处理 {len(chunks)} 个文本块")

    elif args.mode == "train":
        # 训练模式
        print("正在准备微调...")
        fine_tuner = FineTuner(config)

        # 加载数据集
        train_dataset = fine_tuner.prepare_dataset(
            config.fine_tune_data_path + "/train")
        eval_dataset = fine_tuner.prepare_dataset(
            config.fine_tune_data_path + "/eval")

        # 开始训练
        print("开始训练...")
        trainer = fine_tuner.train(train_dataset, eval_dataset)
        print("训练完成!")

    elif args.mode in ["web", "api"]:
        # 运行服务
        print("正在加载RAG系统...")

        # 加载向量数据库
        vector_store_manager = VectorStoreManager(config)
        vector_store = vector_store_manager.load_vector_store()

        # 加载文档块（用于稀疏检索）
        with open(os.path.join(config.knowledge_base_path, "processed_chunks.json"), 'r') as f:
            chunks = json.load(f)

        # 创建检索器
        retriever = HybridRetriever(vector_store, chunks, config)

        # 创建RAG系统
        rag_system = RAGSystem(config)
        rag_system.retriever = retriever

        if args.mode == "web":
            # 启动Gradio Web界面
            print("启动Gradio Web界面...")
            app = GradioApp(rag_system)
            gradio_app = app.create_web_app()
            gradio_app.launch(server_name=args.host, server_port=args.port)
        else:
            # 启动FastAPI服务
            print("启动FastAPI服务...")
            api_app = FastAPIApp(rag_system)
            api_app.run(host=args.host, port=args.port)

    elif args.mode == "eval":
        # 评估模式
        print("开始评估...")
        evaluator = Evaluator()

        # 这里需要加载测试数据集
        # test_data = load_test_data()

        # 进行评估
        # results = evaluator.evaluate_rag(predictions, references)
        # print(f"评估结果: {results}")

        print("评估模式待实现...")


if __name__ == "__main__":
    main()

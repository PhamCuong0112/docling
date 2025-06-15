import os
import streamlit as st
import pandas as pd
from docling.document_converter import DocumentConverter
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import tempfile
from tqdm import tqdm
from dotenv import load_dotenv
import requests
import json
import numpy as np
from typing import List
import torch
import asyncio
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PyTorchのイベントループ設定
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    
# asyncioのイベントループ設定
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 環境変数の読み込み
load_dotenv()

# カスタムCSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .excel-preview {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Ollamaクライアント設定
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")

class OllamaClient:
    def __init__(self, host, model):
        self.host = host
        self.model = model
        self.api_generate = f"{host}/api/generate"
    
    def generate(self, prompt, stream=True, **kwargs):
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        response = requests.post(self.api_generate, headers=headers, json=data, stream=stream)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    yield json_response
        else:
            yield json.loads(response.text)

# カスタムEmbedding関数の定義
class SentenceTransformerEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
    
    def __call__(self, input: Documents) -> Embeddings:
        with torch.no_grad():
            embeddings = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
            
    @property
    def device(self):
        return next(self.model.parameters()).device

# モデルの初期化
def initialize_models():
    converter = None
    ollama_client = None
    embedding_function = None
    chroma_client = None
    
    # Docling初期化
    try:
        converter = DocumentConverter()
        st.info("Docling初期化完了")
    except Exception as e:
        st.error(f"Docling初期化中にエラーが発生しました: {str(e)}")
        return None, None, None, None
    
    # Ollama Client初期化
    try:
        ollama_client = OllamaClient(OLLAMA_HOST, MODEL_NAME)
        # 接続テスト
        next(ollama_client.generate("test", stream=False))
        st.info("Ollama Client初期化完了")
    except Exception as e:
        st.error(f"Ollama Client初期化中にエラーが発生しました: {str(e)}")
        return None, None, None, None
    
    # Embedding Model初期化
    try:
        embedding_function = SentenceTransformerEmbedding()
        st.info("Embedding Model初期化完了")
    except Exception as e:
        st.error(f"Embedding Model初期化中にエラーが発生しました: {str(e)}")
        return None, None, None, None
    
    # ChromaDB初期化
    try:
        os.makedirs("./chroma_db", exist_ok=True)
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        st.info("ChromaDB初期化完了")
    except Exception as e:
        st.error(f"ChromaDB初期化中にエラーが発生しました: {str(e)}")
        return None, None, None, None
        
    return converter, ollama_client, embedding_function, chroma_client

# メインアプリケーション
def main():
    st.title("📊 Excel RAG Assistant powered by Docling & Llama2")
    
    # サイドバー設定
    with st.sidebar:
        st.title("🔧 設定")
        
        TEMPERATURE = st.slider(
            "温度（創造性）",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="値が高いほど創造的な回答になります"
        )
    
    # モデルの初期化
    converter, ollama_client, embedding_function, chroma_client = initialize_models()
    if None in (converter, ollama_client, embedding_function, chroma_client):
        st.error("初期化に失敗したコンポーネントがあります。ログを確認してください。")
        return

    st.success("すべてのコンポーネントの初期化が完了しました")

    if not all([converter, ollama_client, embedding_function, chroma_client]):
        st.error("一部のコンポーネントの初期化に失敗しました。環境変数や依存関係を確認してください。")
        return
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "Excelファイルをアップロード",
        type=["xlsx", "xls"],
        help="200MB以下のExcelファイルをアップロードしてください"
    )
    
    if uploaded_file:
        try:
            # ファイルの一時保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
              # プレビューとグラフの表示
            with st.expander("Excel プレビュー & データ可視化", expanded=True):
                df = pd.read_excel(file_path)
                
                # データフレームの表示
                st.subheader("データテーブル")
                st.dataframe(df, use_container_width=True)
                
                # データ可視化セクション
                st.subheader("データ可視化")
                
                # 数値列を自動検出
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    # グラフタイプの選択
                    chart_type = st.selectbox(
                        "グラフタイプを選択",
                        ["棒グラフ", "折れ線グラフ", "散布図", "箱ひげ図", "ヒストグラム"]
                    )
                    
                    # X軸とY軸の選択
                    x_column = st.selectbox("X軸を選択", df.columns)
                    y_column = st.selectbox("Y軸を選択", numeric_columns)
                    
                    # グラフの作成
                    try:
                        fig = None
                        if chart_type == "棒グラフ":
                            fig = px.bar(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
                        elif chart_type == "折れ線グラフ":
                            fig = px.line(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
                        elif chart_type == "散布図":
                            fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
                        elif chart_type == "箱ひげ図":
                            fig = px.box(df, x=x_column, y=y_column, title=f"{x_column} による {y_column} の分布")
                        elif chart_type == "ヒストグラム":
                            fig = px.histogram(df, x=y_column, title=f"{y_column} の分布")
                        
                        if fig:
                            # グラフのレイアウト調整
                            fig.update_layout(
                                width=800,
                                height=500,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフの生成中にエラーが発生しました: {str(e)}")
                else:
                    st.warning("数値データが見つかりません。グラフを生成するには数値データが必要です。")
            
            # ドキュメント処理
            with st.spinner("ドキュメントを処理中..."):
                # converterがNoneでないか確認
                if converter is None:
                    st.error("ドキュメント変換器（converter）が初期化されていません。環境変数や依存関係を確認してください。")
                    return
                # Doclingでの変換
                result = converter.convert(file_path)
                  # ChromaDBコレクションの作成
                collection_name = "excel_data_" + os.path.splitext(uploaded_file.name)[0]
                try:
                    # ChromaDBクライアントの確認
                    if chroma_client is None:
                        st.error("ChromaDBクライアントが初期化されていません")
                        return

                    # 既存のコレクションがあれば削除
                    try:
                        chroma_client.delete_collection(name=collection_name)
                    except Exception:
                        pass  # コレクションが存在しない場合は無視
                    
                    # 新しいコレクションを作成
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=embedding_function
                    )
                    
                    # データのチャンク化と保存
                    chunks = result.document.export_to_markdown().split("\n\n")
                    
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():  # 空のチャンクをスキップ
                            collection.add(
                                documents=[chunk],
                                ids=[f"chunk_{i}"]
                            )
                            progress_bar.progress((i + 1) / len(chunks))
                    
                    st.success("✅ ドキュメントの処理が完了しました！")
                    
                except Exception as e:
                    st.error(f"データベースの作成中にエラーが発生しました: {str(e)}")
                    return
            
            # チャットインターフェース
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("質問を入力してください"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("回答を生成中..."):
                        if ollama_client is None:
                            st.error("Ollamaクライアントが初期化されていません")
                            return

                        # 関連コンテキストの検索
                        results = collection.query(
                            query_texts=[prompt],
                            n_results=3
                        )
                        
                        # 安全にドキュメントを取得
                        documents = results.get("documents")
                        if documents and len(documents) > 0 and documents[0]:
                            context = "\n".join(documents[0])
                        else:
                            context = "関連するコンテキストが見つかりませんでした。"
                        
                        # プロンプトの構築
                        system_prompt = """あなたは優秀なアシスタントです。
                        提供されたExcelデータの内容に基づいて、正確で具体的な回答を提供してください。
                        データに含まれていない情報については、その旨を明確に伝えてください。"""
                        
                        full_prompt = f"""{system_prompt}

コンテキスト：
{context}

質問：{prompt}

回答を日本語で提供してください。"""

                        response_message = ""
                        message_placeholder = st.empty()
                        
                        try:
                            for response in ollama_client.generate(
                                prompt=full_prompt,
                                stream=True,
                                temperature=TEMPERATURE
                            ):
                                if "response" in response:
                                    response_message += response["response"]
                                    message_placeholder.markdown(response_message + "▌")
                            
                            message_placeholder.markdown(response_message)
                        except Exception as e:
                            st.error(f"回答の生成中にエラーが発生しました: {str(e)}")
            
            # クリアボタン
            if st.sidebar.button("💫 チャット履歴をクリア"):
                st.session_state.messages = []
                collection.delete()
                st.rerun()
                
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
        finally:
            # 一時ファイルの削除
            if "file_path" in locals():
                os.unlink(file_path)

if __name__ == "__main__":
    main()

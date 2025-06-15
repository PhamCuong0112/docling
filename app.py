import streamlit as st
from docling.document_converter import DocumentConverter
from llama_cpp import Llama
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import tempfile

# アプリケーションの設定
st.set_page_config(
    page_title="Docling RAG チャットアプリ",
    page_icon="📄",
    layout="wide"
)

# サイドバーの設定
st.sidebar.title("設定")

# モデルのパス設定
MODEL_PATH = st.sidebar.text_input(
    "Llamaモデルパス",
    "models/llama-2-7b-chat.gguf",
    help="Llama-2モデルのパスを指定してください"
)

# 初期化
@st.cache_resource
def initialize_models():
    converter = DocumentConverter()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4
    )
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    return converter, llm, embedding_model, chroma_client

converter, llm, embedding_model, chroma_client = initialize_models()

# メインUI
st.title("Docling RAG チャットアプリ")

# ファイルアップロード
uploaded_file = st.file_uploader("ドキュメントをアップロード", type=["xlsx", "xls", "pdf", "docx"])

if uploaded_file:
    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # ドキュメント変換
    try:
        with st.spinner("ドキュメントを処理中..."):
            result = converter.convert(file_path)
            
            # ChromaDBにデータを保存
            collection = chroma_client.create_collection(
                name="document_chunks",
                embedding_function=lambda texts: embedding_model.encode(texts).tolist()
            )
            
            # テキストをチャンクに分割して保存
            chunks = result.document.export_to_markdown().split("\n\n")
            collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            
            st.success("ドキュメントの処理が完了しました！")
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        os.unlink(file_path)

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
        # 関連文書の検索
        results = collection.query(
            query_texts=[prompt],
            n_results=3
        )
        
        # コンテキストの作成
        context = "\n".join(results["documents"][0])
        
        # LlamaでRAG応答を生成
        response = llm(
            f"""以下のコンテキストに基づいて、質問に答えてください：

コンテキスト:
{context}

質問: {prompt}

回答：""",
            max_tokens=512,
            temperature=0.7
        )
        
        st.markdown(response["choices"][0]["text"])
        st.session_state.messages.append({"role": "assistant", "content": response["choices"][0]["text"]})

# クリアボタン
if st.sidebar.button("チャット履歴をクリア"):
    st.session_state.messages = []
    if "collection" in locals():
        collection.delete()
    st.experimental_rerun()

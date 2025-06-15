import os
import streamlit as st
import pandas as pd
from docling.document_converter import DocumentConverter
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
from tqdm import tqdm
from dotenv import load_dotenv
import requests
import json

# 環境変数の読み込み
load_dotenv()

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

# Streamlitの設定
st.set_page_config(
    page_title="Excel RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
@st.cache_resource
def initialize_models():
    try:
        converter = DocumentConverter()
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        chroma_client = chromadb.Client()
        return converter, llm, embedding_model, chroma_client
    except Exception as e:
        st.error(f"モデルの初期化中にエラーが発生しました: {str(e)}")
        return None, None, None, None

# メインアプリケーション
def main():
    st.title("📊 Excel RAG Assistant powered by Docling & Llama-3.2")
    
    # モデルの初期化
    converter, llm, embedding_model, chroma_client = initialize_models()
    
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
            
            # プレビューの表示
            with st.expander("Excel プレビュー", expanded=True):
                df = pd.read_excel(file_path)
                st.dataframe(df, use_container_width=True)
            
            # ドキュメント処理
            with st.spinner("ドキュメントを処理中..."):
                # Doclingでの変換
                result = converter.convert(file_path)
                
                # ChromaDBコレクションの作成
                collection_name = "excel_data_" + os.path.splitext(uploaded_file.name)[0]
                try:
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=lambda texts: embedding_model.encode(texts).tolist()
                    )
                    
                    # データのチャンク化と保存
                    chunks = result.document.export_to_markdown().split("\n\n")
                    
                    with st.progress(0) as progress_bar:
                        for i, chunk in enumerate(chunks):
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
                        # 関連コンテキストの検索
                        results = collection.query(
                            query_texts=[prompt],
                            n_results=3
                        )
                        
                        context = "\n".join(results["documents"][0])
                        
                        # プロンプトの構築
                        system_prompt = """あなたは優秀なアシスタントです。
                        提供されたExcelデータの内容に基づいて、正確で具体的な回答を提供してください。
                        データに含まれていない情報については、その旨を明確に伝えてください。"""
                        
                        full_prompt = f"""以下のコンテキストに基づいて質問に回答してください：

{system_prompt}

コンテキスト：
{context}

質問：{prompt}

回答："""
                        
                        # 回答の生成
                        response = llm(
                            full_prompt,
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            stream=True
                        )
                        
                        # ストリーミング出力
                        placeholder = st.empty()
                        full_response = ""
                        for chunk in response:
                            if chunk["choices"][0]["text"]:
                                full_response += chunk["choices"][0]["text"]
                                placeholder.markdown(full_response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
            
            # クリアボタン
            if st.sidebar.button("💫 チャット履歴をクリア"):
                st.session_state.messages = []
                collection.delete()
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
        finally:
            # 一時ファイルの削除
            if "file_path" in locals():
                os.unlink(file_path)

if __name__ == "__main__":
    main()

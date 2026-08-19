import os
from dotenv import load_dotenv
from typing import List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

llm_model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0.0
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=gemini_api_key,
    task_type="retrieval_document"
)

arquivos = DirectoryLoader(
    path="Arquivos_POPs",
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)

documentos = arquivos.load()

def obter_criar_FAISS(
    chunks: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    caminho_indice: str = ".faiss_pop_google",
    forcar_reindexacao: bool = False
) -> FAISS:
  index_path = Path(caminho_indice)

  if index_path.exists() and not forcar_reindexacao:
    return FAISS.load_local(
      folder_path=caminho_indice,
      embeddings=embeddings,
      allow_dangerous_deserialization=True
    )

  vector_store = FAISS.from_documents(chunks, embeddings)
  vector_store.save_local(caminho_indice)
  return vector_store

chunks = RecursiveCharacterTextSplitter(
  chunk_size=1200,
  chunk_overlap=200,
  separators=["\n\n", "\n", " ", ""]
).split_documents(documentos)

vector_store = obter_criar_FAISS(
  chunks = chunks,
  embeddings = embeddings_model,
  caminho_indice="./faiss_pops_google",
  forcar_reindexacao=False
)

dense_retriever = vector_store.as_retriever(search_kwargs={"k":4})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

def formatar_documentos(docs):
  formatted = []
  for doc in docs:
    origem = doc.metadata.get("source", "Documento não identificado")
    formatted.append(f"[Fonte: {origem}]\n{doc.page_content}")
  return "\n\n---\n\n".join(formatted)

template = """Você é um assistente técnico especializado nos Procedimentos Operacionais Padrão (POPs) do Hospital Rio Grande.

Diretrizes obrigatórias:
1. Responda à pergunta do usuário baseando-se estritamente nas informações fornecidas no Contexto.
2. Se a informação necessária não estiver explícita no contexto, declare claramente: "Não encontrei essa informação nos POPs disponibilizados." Não tente deduzir ou utilizar conhecimentos externos.
3. Sempre cite o documento de origem (conforme indicado em [Fonte: ...]) ao final de cada instrução.
4. Mantenha os caminhos de rede, diretórios e parâmetros exatamente como descritos.

Contexto:
{context}

Pergunta:
{question}

Resposta Técnica:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {
        "context": hybrid_retriever | formatar_documentos,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm_model
    | StrOutputParser()
)

resposta = rag_chain.invoke("Estou com uma mensagem de erro em vermelho no eSocial ao tentar acessar o Fortes AC. O que fazer?")
print(resposta)
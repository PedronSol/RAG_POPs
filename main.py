import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

llm_model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5:1.5b",
    temperature=0.0,
    num_ctx=2048,
    num_gpu=99
)

embeddings_model = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

arquivos = DirectoryLoader(
    path="Arquivos_POPs",
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)

documentos = arquivos.load()

def obter_criar_FAISS(
    chunks: List[Document],
    embeddings: OllamaEmbeddings,
    caminho_indice: str = ".faiss_pop_ollama",
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
  chunk_size=800,
  chunk_overlap=100,
  separators=["\n\n", "\n", " ", ""]
).split_documents(documentos)

vector_store = obter_criar_FAISS(
  chunks = chunks,
  embeddings = embeddings_model,
  caminho_indice="./faiss_pops_ollama",
  forcar_reindexacao=False
)

dense_retriever = vector_store.as_retriever(search_kwargs={"k":3})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

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
Responda APENAS com base no contexto fornecido. Caso não encontre a resposta, responda apenas com 'Esta informação não conta nos POPs'.

--- EXEMPLO ---
Contexto: [Fonte: POP.TI.001] Para resetar a senha, acesse o painel e clique em 'Esqueci Senha'
Pergunta: Como altero minha senha?
Resposta: Acesse o painel e selecione 'Esqueci Senha'. [Fonte: POP.TI.001]
---------------

Contexto:
{context}

Pergunta:
{question}

Resposta:"""

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

for chunk in rag_chain.stream("Estou com uma mensagem de erro em vermelho no eSocial ao tentar acessar o Fortes AC. O que fazer?"):
  print(chunk, end="", flush=True)
import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

CAMINHO_CHROMA = "./chroma_pops_db"
NOME_COLECAO = "pops_hospital"

llm_model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:1.7b",
    temperature=0.0,
    num_ctx=4096
)

embeddings_model = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
    num_ctx=2048
)

arquivos = DirectoryLoader(
    path="Arquivos_POPs",
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)

documentos = arquivos.load()

def obter_criar_CHROMA(
    chunks: List[Document],
    embeddings: OllamaEmbeddings,
    persist_dir: str = CAMINHO_CHROMA,
    collection_name: str = NOME_COLECAO,
    forcar_reindexacao: bool = False
) -> Chroma:
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if forcar_reindexacao:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )

    quantidade_docs = vector_store._collection.count()

    if not(quantidade_docs > 0 and not forcar_reindexacao): vector_store.add_documents(chunks)

    return vector_store

chunks = RecursiveCharacterTextSplitter(
  chunk_size=800,
  chunk_overlap=100,
  separators=["\n\n", "\n", " ", ""]
).split_documents(documentos)

vector_store = obter_criar_CHROMA(
  chunks = chunks,
  embeddings = embeddings_model,
  persist_dir = CAMINHO_CHROMA,
  collection_name = NOME_COLECAO,
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
Divida a resposta em tópicos, como um passo a passo, e SEMPRE cite a fonte principal da resposta.

--- EXEMPLO ---
Contexto: [Fonte: POP.TI.001] Para resetar a senha, acesse o painel e clique em 'Esqueci Senha'
Pergunta: Como altero minha senha?
Resposta: 
Para resetar a senha, execute os seguintes passos:
1. Acesse o painel 
2. selecione 'Esqueci Senha'. 
[Fonte: POP.TI.001]
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

print("\nSeja bem vindo ao HR-GPT, o sistema de consultas aos manuais do Hospital Rio Grande!")

pergunta = input("\nEscreva aqui a sua dúvida: ")

while pergunta != "Sair":
  print("\n")

  for chunk in rag_chain.stream(pergunta):
    print(chunk, end="", flush=True)

  pergunta = input("\nEscreva aqui a sua dúvida: ")

print("\nEspero ter sido útil, até a próxima interação!")
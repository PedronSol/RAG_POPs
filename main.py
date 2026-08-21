from typing import List, Dict
from pathlib import Path
from operator import itemgetter

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

def carregar_documentos(pasta: str) -> List[Document]:
  caminho = Path(pasta)
  documentos = []
  for arq in caminho.glob("**/*.docx"):
    docs = Docx2txtLoader(str(arq)).load()
    for d in docs:
      d.metadata["source"] = arq.name
    documentos.extend(docs)
  return documentos

documentos = carregar_documentos("Arquivos_POPs")

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

def formatar_documentos(docs: List[Document]) -> str:
  
  if not docs:
    return "Nenhum documento encontrado"

  fontes_unicas = list(dict.fromkeys(d.metadata.get("source", "Desconhecido") for d in docs))

  cabecalho_fontes = f"Arquivos encontrados ({len(fontes_unicas)}):\n" + "\n".join(f"- {f}" for f in fontes_unicas)

  conteudo = "\n\n---\n\n".join(
    f"[Fonte: {d.metadata.get('source')}]\n{d.page_content}" for d in docs
  )

  return f"{cabecalho_fontes}\n\nConteúdo detalhado:\n{conteudo}"

template = """Você é um assistente técnico especializado nos Procedimentos Operacionais Padrão (POPs) do Hospital Rio Grande.
Responda APENAS com base no contexto fornecido. Caso não encontre a resposta.

DIRETRIZES DE RESPOSTA:
1. **Perguntas de Catálogo/Existência** (ex: "existem arquivos sobre X?", "quais POPs falam sobre Y?"):
  - Verifique os "Arquivos encontrados" no contexto.
  - Se houver arquivos, responda: "Encontrei X arquivo(s) sobre este tema: [liste os nomes dos arquivos]. Gostaria de detalhar algum procedimento específico?"
  - Se não houver arquivos pertinentes, responda: "Não foram encontrados POPs sobre este tema."
2. **Perguntas Procedimentais/Específicas** (ex: "como atualizar o sistema?", "o que fazer no erro X?"):
  - Forneça o passo a passo direto baseado estritamente no conteúdo.
  - Cite a [Fonte: Nome do Arquivo] ao final.
3. Utilize o histórico da conversa para entender continuações (ex: se o usuário disser "sim, o primeiro", consulte o contexto e detalhe o arquivo citado anteriormente).

Contexto:
{context}

Resposta:"""

prompt = ChatPromptTemplate.from_messages([
  ("system", template),
  MessagesPlaceholder(variable_name="chat_history"),
  ("human", "{question}")
])

rag_chain = (
    {
        "context": itemgetter("question") | hybrid_retriever | formatar_documentos,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt
    | llm_model
    | StrOutputParser()
)

armazenamento_sessoes: Dict[str, InMemoryChatMessageHistory] = {}

def obter_historico_sessao(session_id: str) -> InMemoryChatMessageHistory:
  if session_id not in armazenamento_sessoes:
    armazenamento_sessoes[session_id] = InMemoryChatMessageHistory()
  return armazenamento_sessoes[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
  rag_chain,
  obter_historico_sessao,
  input_messages_key="question",
  history_messages_key="chat_history"
)

config_sessao = {"configurable": {"session_id": "usuario_ti_01"}}

print("\nSeja bem vindo ao HR-GPT, o sistema de consultas aos manuais do Hospital Rio Grande!")

pergunta = input("\nEscreva aqui a sua dúvida (ou Sair): ")

while pergunta.strip().lower() != "sair":
  print("\n")

  for chunk in conversational_rag_chain.stream(
    {"question":pergunta},
    config=config_sessao
  ): print(chunk, end="", flush=True)

  pergunta = input("\nEscreva aqui a sua dúvida (ou Sair): ")

print("\nEspero ter sido útil, até a próxima interação!")
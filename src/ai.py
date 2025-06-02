from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda




_ = load_dotenv()
# Instanciar a clase para carregar o CSV
loader_cabecalho = CSVLoader(file_path="202401_NFs_Cabecalho.csv")
loader_itens = CSVLoader(file_path="202401_NFs_Itens.csv")

# Carregar os documentos do CSV 
documents_cabecalho = loader_cabecalho.load()
documents_itens = loader_itens.load()
all_documents = documents_cabecalho + documents_itens

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(all_documents, embeddings)

retriver = vectorstore.as_retriever()

retriver.invoke("")

def call_ai(user_input, chat_history):

    llm = ChatGroq(
    model="llama3-8b-8192",
    temperature= 0
    )



    rag_template = """
    Você é um agente inteligente que ajuda usuários a consultar dados de notas fiscais armazenados em arquivos CSV.

    Seu objetivo é responder perguntas com base nos dados carregados a partir de dois arquivos CSV:
    - `202401_NFs_Cabecalho.csv`: contém os cabeçalhos das notas fiscais.
    - `202401_NFs_Itens.csv`: contém os itens relacionados a essas notas fiscais.

    Os campos estão separados por vírgulas. Os valores numéricos usam ponto como separador decimal. Datas seguem o formato: `AAAA-MM-DD HH:MM:SS`.

    **Importante:**
    - Todas as respostas devem ser baseadas exclusivamente nos dados disponíveis nos arquivos CSV.
    - Não invente respostas nem utilize conhecimento externo.
    - Caso não encontre uma resposta com base nos dados, informe claramente que a informação não está presente.
    - Responda de forma objetiva e com base em evidência. Quando possível, cite o valor encontrado e o campo relacionado.

    Exemplos de perguntas que você pode responder:
    - Qual fornecedor recebeu o maior montante em janeiro de 2024?
    - Qual item teve maior volume entregue?
    - Quantas notas fiscais foram emitidas para o fornecedor X?
    - Qual foi a média de valores unitários dos itens?

    Use linguagem clara e direta, sempre retornando a resposta final no final do texto.

    Sempre que possível, use os seguintes campos como base:
    - Do cabeçalho: `numero_nota`, `data_emissao`, `fornecedor`, `valor_total`, etc.
    - Dos itens: `descricao_item`, `quantidade_item`, `valor_unitario`, `valor_total_item`, etc.

    Você tem acesso a um sistema de busca (retriever) que localiza informações relevantes nesses arquivos para responder as perguntas do usuário.

    Contexto: {context}
    Pergunta: {question}
    Histórico: {chat_history}


    """

    prompt = ChatPromptTemplate.from_template(rag_template)
   

    chain = (
        {"context": retriver, "question": RunnablePassthrough(), "chat_history": RunnableLambda(lambda x: chat_history)}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(user_input)

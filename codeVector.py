import dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Use pysqlite3 as sqlite3 (needed for Chroma on Codespace)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

dotenv.load_dotenv()

# Clone
repo_path = "falkordb-py_repo"
# commented-out after initial clone
# repo = Repo.clone_from("https://github.com/falkordb/falkordb-py", to_path=repo_path)

# Load
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
# print(len(documents))
# > 69

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
# print(len(texts))
# > 163

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    # below is optional preference
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

# Note - using gpt-4 seems to be less reliable in ascertaining full context
llm = ChatOpenAI(model="gpt-3.5-turbo")

# First we need a prompt that we can pass into an LLM to generate this search query
prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)

question = 'Is the test_explain.py module in your context?'
result = qa.invoke({"input": question})
print(result["answer"])
# > The QueryResult class contains methods such as initialization (__init__), parsing response data, checking for errors, parsing results, getting statistics, parsing headers, parsing records, and tracking various metrics like labels added/removed, nodes created/deleted, properties set/removed, relationships created/deleted, indices created/deleted, cached execution, and run time in milliseconds.
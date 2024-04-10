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
repo_path = "test_repo"
repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

# Load
loader = GenericLoader.from_filesystem(
    repo_path + "/libs/core/langchain_core",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
# print(len(documents))
# > 297

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
# print(len(texts))
# > 916

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(model="gpt-4")

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

question = "What is a RunnableBinding?"
result = qa.invoke({"input": question})
# print(result["answer"])
# > A `RunnableBinding` is a class in the provided context that inherits from `RunnableBindingBase`. This class is used to bind arguments to a Runnable, returning a new Runnable. This is useful when a runnable in a chain requires an argument that is not in the output of the previous runnable or included in the user input. It essentially wraps around another runnable (referred to as "bound") and can modify or control how the bound runnable operates.

questions = [
    "What classes are derived from the Runnable class?",
    "What one improvement do you propose in code in relation to the class hierarchy for the Runnable class?",
]

for question in questions:
    result = qa.invoke({"input": question})
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
# > **Question**: What classes are derived from the Runnable class? 
# > **Answer**: The classes derived from the Runnable class are RunnableLambda, RunnableLearnable, RunnableBinding, RunnableSerializable, and RunnableWithFallbacks. 
# > **Question**: What one improvement do you propose in code in relation to the class hierarchy for the Runnable class? 
# > **Answer**: One proposed improvement in the given code is the introduction of a more organized and clear class hierarchy for the Runnable class. Currently, it seems that the Runnable class has a number of subclasses with varying functionality (e.g., RunnableLambda, RunnableConfigurableFields, RunnableSerializable, etc.). However, it's not clear how these classes are related or how they extend the functionality of the base Runnable class. 
# > An improved hierarchy might introduce intermediate abstract classes that group related functionality. For example, all Runnable classes that are serializable could inherit from an abstract 'SerializableRunnable' class. This class would implement the methods related to serialization, allowing concrete subclasses to focus on their specific functionality. This design would make it easier to understand the relationships between different Runnable classes and could potentially reduce code duplication. 
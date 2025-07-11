from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
import os

class InsightForgeRAG:
    def __init__(self, summary_text, df):
        self.df = df
        self.docs = self._prepare_documents(summary_text)
        self.vectorstore = self._create_vectorstore()
        self.llm = self._load_groq_chat_model()
        self.qa_chain = self._build_rag_chain()

    def _prepare_documents(self, summary_text):
        doc = Document(page_content=summary_text)
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents([doc])

    def _create_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return DocArrayInMemorySearch.from_documents(self.docs, embeddings)

    def _load_groq_chat_model(self):
        os.environ["OPENAI_API_KEY"] = "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"
        os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
        return ChatOpenAI(
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=512
        )

    def _build_rag_chain(self):
        retriever = self.vectorstore.as_retriever()
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff"
        )

    def answer(self, question: str):
        try:
            return self.qa_chain.run(question)
        except Exception as e:
            return f"An error occurred during processing: {str(e)}"

    def build_rag_system(summary_text, df):
        rag = InsightForgeRAG(summary_text, df)
        return rag

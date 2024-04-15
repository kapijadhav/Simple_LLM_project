from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
import os
from dotenv import load_dotenv
load_dotenv()


llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"], temperature = 0)


instructor_embeddings = HuggingfaceInstructEmbeddings()

vectordb_file_path = "faiss_index"
#we dont want to use vectordb in memory, as it take log time soo we store it in disk
def create_vector_db():
    vectordb= FAISS.from_documets(documents =data, embedding = instructor_embedding)
    loader = CSVLoader(file_path="file_name.csv",
                       source_column="column_name")  # source_column = whichever columns contains questions provide that column
    data = loader.load()
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    #load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_ile_path, instructor_embeddings)

    #create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold =0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making.
    If the answer is not found in the context, kindly state "I dont't know." Don't try to make up an answer.

    CONTEXT: {context}

    Question: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]

    # lets create object of this class
    chain = RetrievalQA(llm=llm,
                        chain_type="stuff",  # is can be stuff or mapreduce
                        retriever=retriever,
                        input_key="query",
                        return_source_documents=True,
                        # when we get answer then we also want to return the source document from that csv file which were relevant to this answer
                        chain_type_kwargs={"prompt": PROMPT}
                        )

    return chain



if __name__ == "__main__":
    chain = get_qa_chain()

    print(chain("write question?"))








from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the response csv data
loader = CSVLoader(file_path="data2.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(model='gpt-3.5-turbo-16k-0613', temperature=0)

template = """
You are Hafiz personal assistant that answers recruiters' questions about Hafiz's professional skills for their career.
I will share a prospect's message with you, and you will give me the best answer that you should send to this prospect based on past best
practices, and you will follow all the rules below:

1/ The response should be very similar or even identical to the past best practices, in terms of length, tone of voice, 
logical arguments, and other details but change words instead "i" use "him" because you are basically Hafiz assistant.

2/ If the best practices are irrelevant, then try to mimic the style of the best practice, but keep in my mind keep the response straight to the point, don't take it too long.


Below is a message you received from a recruiter:
{message}

Here is a list of best practices of how I normally respond to recruiters in similar scenarios:
{best_practices}

Please write the best response that I should send to this recruiter:
"""

prompt = PromptTemplate(
    input_variables=["prompt", "text", "rejected_text"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practices = retrieve_info(message)
    response = chain.run(message=message, best_practices=best_practices)
    return response


message = "What experience Hafiz have in term of software engineering?"

print(generate_response(message))

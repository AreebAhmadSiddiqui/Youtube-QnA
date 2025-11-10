from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

### Step 1a. Indexing ( Document Ingestion)


# video_id='ddq8JIMhz7c' # Get the video id from the youtube url
video_id='bFe0dCWfpXA' # scoopcast
# ngvOyccUzzY ( David Goggins )

try:
    transcript_list=YouTubeTranscriptApi().fetch(video_id=video_id,languages=['hi','en'])

    transcript=" ".join(snippet.text for snippet in transcript_list)

    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video")


# Step 1.b Split Text

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents([transcript])
# print(len(chunks))


# Step 1c and 1d - Indexing ( Embedding generation and storing in vector store)

embeddingModel=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store=FAISS.from_documents(chunks,embedding=embeddingModel)

# print(vector_store)


# Step 2. Retiriever

retriever= vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})
# print(retriever)

# test retriver

# print(retriever.invoke("What is the best study routine?"))

# Step 3- Augmentation

prompt=PromptTemplate(
    template="""
        You are a helpful assistant,
        Answer ONLY from the provided transcript context
        If the context is insufficient, just say you don't know
        Also NOTE if the transcript is in any other language please convert the same in english and then answer
        {context}

        Question: {question}
    """,
    input_variables=['context','question']
)

# question ="Is the topic of aliens discussed in the video"
# retrieved_docs=retriever.invoke(question)

# context_text="\n\n".join(doc.page_content for doc in retrieved_docs)

# final_prompt=prompt.invoke({'context':context_text,'question':question})


# Step 4 - Generation

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# answer=model.invoke(final_prompt)
# print(answer.content)


## Lets chain

from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain=RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
    }
)

# print(parallel_chain.invoke('Qualities of good )students'))


final_chain = parallel_chain | prompt | model | StrOutputParser()

print(final_chain.invoke('Top 10 points from the video'))
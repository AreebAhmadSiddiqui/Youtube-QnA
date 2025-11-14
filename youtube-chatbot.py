from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import time

load_dotenv()

### Step 1a. Indexing ( Document Ingestion)


# video_id='ddq8JIMhz7c' # Get the video id from the youtube url
video_id='bFe0dCWfpXA' # scoopcast
# ngvOyccUzzY ( David Goggins )


# Initialize session state for model and messages
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-2.5-flash"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

if "final_chain" not in st.session_state:
    st.session_state.final_chain = None

# Load the Gemini model (cached to avoid reloading)
@st.cache_resource
def load_gemini_model():
    return ChatGoogleGenerativeAI(
        model=st.session_state["gemini_model"],
        temperature=0.7
    )

model = load_gemini_model()


def someFunc(video_id):
    try:
        transcript_list=YouTubeTranscriptApi().fetch(video_id=video_id,languages=['hi','en'])

        transcript=" ".join(snippet.text for snippet in transcript_list)

        # print(transcript)

    except TranscriptsDisabled:
        print("No captions available for this video")


    # Step 1.b Split Text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.create_documents([transcript])
    


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

    # model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

    # answer=model.invoke(final_prompt)
    # print(answer.content)
    ## Lets chain

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
    
    return final_chain

st.html('<h1>Welcome to YoutubeChatbot</h1>')
video_id = st.text_input(label="Video Id",placeholder='Input the video id of the youtube video ( Ex :  https://www.youtube.com/watch?v={VIDEO_ID})')

if video_id:
    st.write(f"Please confirm if this is the video you are referring to : https://www.youtube.com/watch?v={video_id}")
    if st.button("Confirm"):
        with st.spinner("Loading model..."):
            st.session_state.final_chain = someFunc(video_id)
            st.session_state.confirmed = True
        st.success("Model loaded!")

# Display chat interface only if confirmed
if st.session_state.confirmed and st.session_state.final_chain:
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Invoke the Gemini model with streaming
            try:
                for chunk in st.session_state.final_chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # Remove the cursor after streaming completes
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.markdown(full_response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# message = st.chat_message("assistant")
# message.write("Hello human")




# # Display existing messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Handle user input
# if prompt := st.chat_input("What is up?"):
#     # Add user message to session state
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate assistant response
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
        
#         # Invoke the Gemini model with streaming
#         try:
#             for chunk in model.stream(prompt):
#                 full_response += chunk.content
#                 message_placeholder.markdown(full_response + "▌")
            
#             # Remove the cursor after streaming completes
#             message_placeholder.markdown(full_response)
#         except Exception as e:
#             full_response = f"Error: {str(e)}"
#             message_placeholder.markdown(full_response)
    
#     # Add assistant response to session state

#     st.session_state.messages.append({"role": "assistant", "content": full_response})

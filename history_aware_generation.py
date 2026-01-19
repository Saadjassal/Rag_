from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

persistent_directory = "db/chorma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory= persistent_directory, embedding_function= embedding_model, collection_metadata={"hnsw:space": "cosine"})
model = ChatGroq(model="llama-3.1-8b-instant")

chat_history = []

def ask_question(user_question):
    print(f"\n---You asked: {user_question}---")
    
    #make the question clear using chat history
    if chat_history:
        messages= [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [HumanMessage(content= f"New question: {user_question}")]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
        
        
    #find the relevant documents
    retriever = db.as_retriever(search_kwargs= {"k":3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"    Doc {i}: {preview}...")
        
    #creating final prompt
    
    combined_input =f'''Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer
    in the documents, say "I don't have enogh information to answer that question based on the provided documents."
    '''

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! type (quit) to exit.")
    
    while True:
        question = input("\n your question: ")
        
        if question.lower() == 'quit':
            print("Chat Ended")
            break
        
        ask_question(question)

if __name__ == "__main__":
    start_chat()

    # Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
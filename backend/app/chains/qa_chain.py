# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are an AI assistant with access to a knowledge base.
#     Answer questions based ONLY on the provided context.
#     If you don't find the answer in context, say so."""),
#     ("user", """Context from knowledge base:
#     {context}
    
#     Question: {question}
    
#     Answer:""")
# ])

# chain = prompt | llm | StrOutputParser()
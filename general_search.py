def general_search(llm, query):
    # Aquí puedes implementar la lógica para búsquedas generales
    # Por ejemplo, puedes usar el LLM directamente para responder preguntas generales
    response = llm(f"Por favor, responde a la siguiente pregunta: {query}")
    return response
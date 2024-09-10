def general_search(llm, query):
    prompt = f"""Responde a la siguiente pregunta de manera concisa, precisa y bien estructurada. 
    Utiliza un lenguaje claro y correcto, evitando errores ortográficos o gramaticales. 
    La respuesta debe ser informativa pero no exceder de 3-4 frases.

Pregunta: {query}

Respuesta:"""
    
    response = llm(prompt)
    return response.strip()
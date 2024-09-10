# opti/core/intent_classifier.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

intent_template = """
Determina la intención del usuario basándote en su pregunta. Las posibles intenciones son:
1. query_ticket: Consulta de ticket
2. create_ticket: Creación de ticket
3. query_incident: Consulta de incidente
4. create_incident: Creación de incidente
5. general_search: Búsqueda general
6. rag_query: Consulta relacionada con el PDF cargado
7. email: Enviar correo electrónico

Importante: 
- Si el usuario no menciona la palabra "ticket", la intención por defecto es general_search o rag_query. 
- Si el usuario menciona que quiere enviar un correo electrónico, la intención es email.

Pregunta del usuario: {question}

Responde solo con el nombre de la intención correspondiente (en minúsculas y sin espacios).
"""

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template=intent_template,
)

def classify_intent(llm, prompt):
    intent_chain = LLMChain(llm=llm, prompt=intent_prompt)
    return intent_chain.run(prompt).strip().lower()
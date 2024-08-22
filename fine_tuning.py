from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os

def fine_tune_model(training_data):
    api_key = os.getenv("API_KEY")
    project_id = os.getenv("PROJECT_ID")
    model_id = "meta-llama/llama-3-1-70b-instruct"

    model = Model(
        model_id=model_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 100,
        },
        credentials={
            "apikey": api_key,
            "url": "https://us-south.ml.cloud.ibm.com"
        },
        project_id=project_id
    )

    # Realizar fine-tuning
    fine_tuned_model = model.fine_tune(training_data)

    return fine_tuned_model

# Ejemplo de uso:
# training_data = [
#     {"input": "Pregunta 1", "output": "Respuesta 1"},
#     {"input": "Pregunta 2", "output": "Respuesta 2"},
#     # ... más ejemplos
# ]
# fine_tuned_model = fine_tune_model(training_data)
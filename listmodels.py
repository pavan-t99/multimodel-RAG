from google import genai

def list_models(api_key):
    if not api_key:
        raise ValueError("API Key is required")

    client = genai.Client(api_key=api_key)

    models = client.models.list()

    print("Available Models:\n")
    for model in models:
        print(model.name)


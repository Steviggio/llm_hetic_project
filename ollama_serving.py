import requests
import json
import torch

def load_text_from_file(file_path):
    """Charge le texte à partir d'un fichier local."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Fichier '{file_path}' introuvable.")
        return None

def generate_embeddings(content_lines, model):
    """Génère des embeddings pour chaque ligne du contenu."""
    print("Génération des embeddings...")
    embeddings = []
    for line in content_lines:
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": line}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        except Exception as e:
            print(f"Erreur lors de la génération des embeddings : {str(e)}")
    return torch.tensor(embeddings)

def get_relevant_context(user_input, embeddings, content_lines, top_k=3):
    """Récupère les lignes les plus pertinentes du contenu."""
    print("Récupération du contexte pertinent...")
    try:
        # Générer l'embedding pour l'entrée utilisateur
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "llama3.2", "prompt": user_input}
        )
        response.raise_for_status()
        input_embedding = response.json()["embedding"]
        
        cos_scores = torch.cosine_similarity(
            torch.tensor(input_embedding).unsqueeze(0), embeddings
        )
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores))).indices.tolist()
        return [content_lines[idx].strip() for idx in top_indices]
    except Exception as e:
        print(f"Erreur lors de la récupération du contexte : {str(e)}")
        return []

def query_model_with_rag(api_url, prompt, suffix=None, temperature=0.7, stream=False):
    """Envoie une requête au modèle avec RAG."""
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "options": {
            "temperature": temperature
        },
        "stream": stream
    }
    if suffix:
        payload["suffix"] = suffix
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête au modèle : {e}")
        return None

def main():
    # Configuration
    api_url = "http://localhost:11434/api/generate"
    file_path = "lang_learning.txt"  # Fichier texte contenant le contenu
    model = "llama3.2"
    temperature = 0.7  # Réglage de la température

    # Charger le contenu du fichier
    content = load_text_from_file(file_path)

    # Vérification si le fichier est vide ou introuvable
    if not content:
        print("Le fichier est vide ou introuvable. Le contexte ne sera pas utilisé.")
        content_lines = []
    else:
        content_lines = content.splitlines()

    # Générer les embeddings si du contenu est présent
    if content_lines:
        embeddings = generate_embeddings(content_lines, model)
    else:
        embeddings = torch.tensor([])

    # Entrée utilisateur
    user_input = input("Posez une question : ")

    # Récupérer le contexte pertinent s'il y a des embeddings
    if embeddings.nelement() > 0:
        relevant_context = get_relevant_context(user_input, embeddings, content_lines)
    else:
        relevant_context = []

    # Construire le prompt final
    if relevant_context:
        print("\nContexte pertinent trouvé :\n", "\n".join(relevant_context))
        prompt = "\n".join(relevant_context) + "\n\n" + user_input
    else:
        print("Aucun contexte trouvé. Utilisation du prompt utilisateur uniquement.")
        prompt = user_input

    # Effectuer la requête au modèle
    response = query_model_with_rag(api_url, prompt, temperature=temperature)
    if response:
        print("\nRéponse du modèle :\n", response)
    else:
        print("Erreur lors de la récupération de la réponse.")

if __name__ == "__main__":
    main()

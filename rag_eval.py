import openai
import guidance
from guidance import gen, models
import torch
import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel




openai.api_key = "sk-no-key-required"  
openai.api_base = "http://localhost:8080/v1" 





def load_embedding_model(model_name: str = "intfloat/multilingual-e5-large-instruct") -> Tuple[AutoTokenizer, AutoModel, str]:
    """
    Charge le modèle d’embeddings et son tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device



tokenizer, embedding_model, device = load_embedding_model()






def vectorize_query(query: str, tokenizer: AutoTokenizer, model: AutoModel, device: str) -> np.ndarray:
    """
    Convertit une requête en un vecteur d'embedding (E5, BERT, etc.).
    """

    query_input = f"query: {query}" #les modèle e5 sont entrainé à attendre des queries d'où l'ajout du prefix
    inputs = tokenizer(query_input, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()






def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs numpy.
    """
    a = vec_a.flatten()
    b = vec_b.flatten()
    dot_prod = np.dot(a, b)
    denom = norm(a) * norm(b)
    return 0.0 if denom == 0 else dot_prod / denom






def retrieve_documents(query: str, collection_name: str = "GameRag", top_k: int = 3) -> List[str]:
    """
    Recherche dans Qdrant pour récupérer les documents (payload["texte"]) pertinents.
    Retourne la liste des textes combinés.
    """

    qdrant_client = QdrantClient(url="http://localhost:6333")

    query_vector = vectorize_query(query, tokenizer, embedding_model, device)

    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector[0].tolist(),
        limit=top_k,
        with_payload=True
    )

    retrieved_texts = [r.payload.get("texte", "") for r in results]
    return retrieved_texts






def generate_rag_answer(system_prompt: str, query: str, docs: List[str]) -> str:
    """
    Envoie un prompt au modèle (mistral-7b-instruct) avec :
      - system_prompt en role=system
      - query + docs en role=user
    Retourne la réponse complète sous forme de texte (pas en streaming).
    """
    context = "\n\n".join(docs)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nquery:\n{query}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="mistral-7b-instruct",
        messages=messages,
        temperature=0.7,
        top_p=0.9,
    )
    
    answer_text = response["choices"][0]["message"]["content"]
    return answer_text





def create_prompt_guidance(query : str, docs : list[str], answer: str) -> str:
    evaluation_template = f"""
    {{#system}}
    You are an assistant that provides a numeric rating (0 to 5) for each criterion, based on the query, the retrieved documents, and the final answer of a rag system.
    Only output the integer, no extra commentary.
    {{/system}}

    {{#user}}
    Voici les éléments à évaluer :

    Query from user:
    {query}

    Documents (retrieved):
    {docs}

    Answer from RAG:
    {answer}

    Critères à noter (0 = très mauvais, 5 = excellent) :

    1) La réponse contient-elle des informations correctes et vérifiables ?
    2) Y a-t-il des erreurs factuelles ou des hallucinations (informations inventées) ?
    3) La réponse est-elle cohérente avec les sources récupérées ?
    4) La réponse est-elle en lien direct avec la query posée ?
    5) Le niveau de langage est-il adapté au public cible ?
    6) La réponse est-elle formulée de manière concise sans information superflue ?

    {{/user}}

    Critère 1: {{gen 'criterion1' regex='[0-5]'}}
    Critère 2: {{gen 'criterion2' regex='[0-5]'}}
    Critère 3: {{gen 'criterion3' regex='[0-5]'}}
    Critère 4: {{gen 'criterion4' regex='[0-5]'}}
    Critère 5: {{gen 'criterion5' regex='[0-5]'}}
    Critère 6: {{gen 'criterion6' regex='[0-5]'}}
    """
    return evaluation_template


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


llm_for_guidance = models.Transformers(
    model="./Mistral-7B-Instruct-v0.3",
    tokenizer= tokenizer
    )


def evaluate_answer(query: str, docs: str, answer: str) -> dict:
    """
    Appelle le template Guidance pour attribuer 6 notes (0-5) à la réponse.
    Retourne un dict { "criterion1": int, ..., "criterion6": int }
    """

    evaluation_template = create_prompt_guidance(query, docs, answer)

    result = llm_for_guidance + evaluation_template

    scores = {
        "criterion1": int(result["criterion1"]),
        "criterion2": int(result["criterion2"]),
        "criterion3": int(result["criterion3"]),
        "criterion4": int(result["criterion4"]),
        "criterion5": int(result["criterion5"]),
        "criterion6": int(result["criterion6"]),
    }
    return scores





def main(system_prompts : list[str], query : str):
    """
    1) On définit une query unique (toujours la même).
    2) On définit une liste de prompts système différents (à tester).
    3) Pour chaque prompt :
       - Récupère les docs (Qdrant)
       - Génére la réponse du RAG
       - Évalue la réponse via Guidance
    4) Stocke les scores (0-5) dans un CSV, où :
       - Les lignes = critères
       - Les colonnes = prompt#1, prompt#2, etc.
    """

    if not system_prompts:
        print("⚠️  La liste system_prompts est vide. Ajoutez vos prompts puis relancez.")
        return

    criteria_labels = [
        "La réponse contient-elle des informations correctes et vérifiables ?",
        "Y a-t-il des erreurs factuelles ou des hallucinations (informations inventées) ?",
        "La réponse est-elle cohérente avec les sources récupérées ?",
        "La réponse est-elle en lien direct avec la query posée ?",
        "Le niveau de langage est-il adapté au public cible ?",
        "La réponse est-elle formulée de manière concise sans information superflue ?",
    ]

    df_results = pd.DataFrame(index=criteria_labels,
                              columns=[f"prompt{i+1}" for i in range(len(system_prompts))])

    docs = retrieve_documents(query)

    for i, sys_prompt in enumerate(system_prompts):

        answer = generate_rag_answer(system_prompt=sys_prompt, query=query, docs=docs)

        scores_dict = evaluate_answer(query=query, docs=docs, answer=answer)

        df_results.loc[criteria_labels[0], f"prompt{i+1}"] = scores_dict["criterion1"]
        df_results.loc[criteria_labels[1], f"prompt{i+1}"] = scores_dict["criterion2"]
        df_results.loc[criteria_labels[2], f"prompt{i+1}"] = scores_dict["criterion3"]
        df_results.loc[criteria_labels[3], f"prompt{i+1}"] = scores_dict["criterion4"]
        df_results.loc[criteria_labels[4], f"prompt{i+1}"] = scores_dict["criterion5"]
        df_results.loc[criteria_labels[5], f"prompt{i+1}"] = scores_dict["criterion6"]


    output_csv = "evaluation_results.csv"
    df_results.to_csv(output_csv, encoding="utf-8-sig")



system_prompts = [
    "Variant A",
    "Variant B",
    "etcx.."
]

query = "c'est quoi les règles du jeu ?"

if __name__ == "__main__":
    main(system_prompts, query)

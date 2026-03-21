# Projet RAG - Assistant ALM Assurance Vie (DIC)

Ce projet implémente un assistant conversationnel basé sur une architecture **RAG (Retrieval-Augmented Generation)** pour interroger des DIC (Documents d'Informations Clés) en français.

Notebook principal : `rag.ipynb`

## 1) Contexte métier

Le département ALM (Asset Liability Management) doit analyser un volume important de DIC pour orienter les décisions d'investissement, en recherchant un compromis entre rentabilité et maîtrise du risque.

L'objectif du projet est de fournir un assistant de chat capable de retrouver rapidement des informations pertinentes dans ces documents.

## 2) Contraintes du sujet et vérification

### Exigences imposées

- Pas d'API cloud tierce (confidentialité)
- Utilisation de modèles open-weights exécutés localement
- Réponses sourcées vers le document d'origine
- Mode conversationnel avec historique
- Stockage local des embeddings (ChromaDB ou FAISS)
- Évaluation via dataset fourni (`corpus.json`, `queries.json`, `relevant_docs.json`, `answers.json`, `errors.json`)
- F1 BERTScore attendu >= 60%

### Vérification dans l'implémentation

- Confidentialité / local-only : **OK**
  - Modèle local via `transformers` (`mistralai/Mistral-7B-Instruct-v0.2`)
  - Embeddings locaux via `sentence-transformers`
  - Vector store local via `Chroma`
  - Pas d'appel API externe dans le pipeline
- Open-weights : **OK**
  - Modèle génératif Mistral 7B Instruct
- Réponses sourcées : **OK**
  - Le prompt impose la citation de source
  - Le contexte injecté inclut `Source` et `Page`
  - Validation automatique en post-traitement (`Source: ... - Page <nombre>`)
- Mode conversationnel : **OK**
  - `ConversationBufferMemory` activée hors mode évaluation
- Stockage local des embeddings : **OK**
  - `chroma_db/` et `chroma_eval_db/`
- Évaluation dataset : **OK**
  - Chargement de `corpus.json`, `queries.json`, `relevant_docs.json`, `answers.json`
  - Génération de `errors.json`
  - Calcul `BERTScore F1` et `recall@K`
- Seuil de qualité F1 >= 60% : **OK sur l'exécution reportée**
  - Résultat communiqué : `Mean BERT-F1 = 0.6903` (69.03%)

## 3) Architecture technique

### Retrieval

- Chargement des PDF DIC via `PyPDFLoader`
- Découpage en chunks via `RecursiveCharacterTextSplitter`
- Embeddings `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Indexation dans Chroma
- Recherche des chunks via MMR (`max_marginal_relevance_search`)

### Génération

- LLM : `mistralai/Mistral-7B-Instruct-v0.2`
- Prompt contraint au contexte récupéré (RAG)
- Historique conversationnel en mémoire pour le mode chat

### Évaluation

- Retrieval : `recall@K`
- Génération : `BERTScore F1`
- Exports détaillés + synthèse + erreurs

## 4) Structure du dépôt

- `rag.ipynb` : pipeline complet (ingestion, indexation, chat, évaluation, exports)
- `DIC/` : corpus source en PDF
- `chroma_db/` : base vectorielle principale (chat)
- `chroma_eval_db/` : base vectorielle de test (évaluation)
- `dataset_eval/` : dataset d'évaluation + exports

## 5) Variables de configuration principales

Dans `rag.ipynb` :

- `EMBEDDING_MODEL_NAME`
- `MODEL_ID`
- `K`
- `MMR_FETCH_K`
- `MMR_LAMBDA`
- `BERT_THRESHOLD`
- `RUN_CHAT`
- `EVAL_MAX_QUERIES`
- `EVAL_RANDOM_SEED`

Utilisation recommandée :
- Itération rapide : `EVAL_MAX_QUERIES = 120` (exemple)
- Run final : `EVAL_MAX_QUERIES = None`
- Reproductibilité : garder `EVAL_RANDOM_SEED` fixe

## 6) Dataset d'évaluation

Fichiers d'entrée dans `dataset_eval/` :

- `corpus.json` : `doc_uuid -> texte_chunk`
- `queries.json` : `query_uuid -> question`
- `answers.json` : `query_uuid -> réponse de référence`
- `relevant_docs.json` : `query_uuid -> [doc_uuid pertinents]`
- `errors.json` : erreurs/cas problématiques générés pendant l'évaluation

Note importante : dans le dataset actuel, chaque requête a 1 document pertinent annoté.

## 7) Exports produits

- `dataset_eval/result/last/evaluation_results.csv`
- `dataset_eval/result/last/errors.json`
- `dataset_eval/result/last/evaluation_results_{model}_k{K}_thr{XX}_{timestamp}.csv`
- `dataset_eval/result/last/evaluation_summary_{model}_k{K}_thr{XX}_{timestamp}.csv`

Colonnes principales de `evaluation_results.csv` :

- `uuid`, `query`, `gen`, `ref`, `bert_f1`, `retrieved_uuids`, `expected_uuids`, `recall_at_k`, `has_source_citation`

Règles de construction de `errors.json` :

- réponse vide
- ou `bert_f1 < BERT_THRESHOLD`
- ou `recall_at_k == 0`
- ou citation de source absente / invalide (`has_source_citation == False`)

Interprétation de `recall@K` ici : comme il y a un seul doc pertinent par requête, c'est équivalent à un hit@K moyen.

## 8) Exécution

1. Ouvrir `rag.ipynb`
2. Exécuter les cellules dans l'ordre
3. Activer/désactiver le mode chat via `RUN_CHAT`
4. Choisir le mode évaluation :
   - rapide : `EVAL_MAX_QUERIES` (ex: 120)
   - complet : `EVAL_MAX_QUERIES = None`
5. Lancer la section évaluation pour générer les exports dans `dataset_eval/result/last/`

## 9) Limites connues et améliorations possibles

- La validation actuelle repose sur un pattern minimal (`Source: ... - Page <nombre>`).
- Un contrôle plus avancé peut vérifier que la source citée existe réellement dans les documents récupérés.
- L'ajout d'une évaluation factuelle complémentaire (au-delà de BERTScore) renforcerait l'analyse qualité.

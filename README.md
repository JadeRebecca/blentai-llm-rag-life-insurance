# Projet RAG - Assistant ALM Assurance Vie (DIC)

Ce projet implémente un assistant conversationnel basé sur une architecture **RAG (Retrieval-Augmented Generation)** pour interroger des DIC (Documents d'Informations Clés) en français.

Notebook principal : `rag.ipynb`

## 1) Contexte métier

Le département ALM (Asset Liability Management) doit analyser un volume important de DIC pour orienter les décisions d'investissement, en recherchant un compromis entre rentabilité et maîtrise du risque.

L'objectif du projet est de fournir un assistant de chat capable de retrouver rapidement des informations pertinentes dans ces documents.

## 2) Architecture technique

### Retrieval

- Chargement des PDF DIC via `PyPDFLoader`
- Découpage en chunks via `RecursiveCharacterTextSplitter`
- Embeddings finaux : `BAAI/bge-m3`
- Indexation dans Chroma
- Recherche des chunks via `similarity_search` (K=4)

### Génération

- LLM : `mistralai/Mistral-7B-Instruct-v0.2`
- Prompt contraint au contexte récupéré (RAG)
- Historique conversationnel en mémoire pour le mode chat
- `max_new_tokens` piloté par config :
  - `MAX_NEW_TOKENS_EVAL = 128` : garde un protocole stable et comparables entre tests (moins de variance, moins de coût)
  - `MAX_NEW_TOKENS_CHAT = 258` :permet des réponses plus complètes pour l'usage utilisateur

### Évaluation

- Retrieval : `recall@K`
- Génération : `BERTScore F1`
- Exports détaillés + synthèse + erreurs

## 3) Structure du dépôt

- `rag.ipynb` : pipeline complet (ingestion, indexation, chat, évaluation, exports)
- `DIC/` : corpus source en PDF
- `chroma_db/` : base vectorielle principale (chat)
- `chroma_eval_db/` : base vectorielle de test (évaluation)
- `dataset_eval/` : dataset d'évaluation + exports

## 4) Variables de configuration principales

Dans `rag.ipynb` :

- `EMBEDDING_MODEL_NAME`
- `MODEL_ID`
- `K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MAX_NEW_TOKENS_EVAL`
- `MAX_NEW_TOKENS_CHAT`
- `BERT_THRESHOLD`
- `RUN_CHAT`
- `EVAL_MAX_QUERIES`
- `EVAL_RANDOM_SEED`

Utilisation recommandée :
- Itération rapide : `EVAL_MAX_QUERIES = 120` (exemple)
- Run final : `EVAL_MAX_QUERIES = None`
- Reproductibilité : garder `EVAL_RANDOM_SEED` fixe

## 5) Dataset d'évaluation

Fichiers d'entrée dans `dataset_eval/` :

- `corpus.json` : `doc_uuid -> texte_chunk`
- `queries.json` : `query_uuid -> question`
- `answers.json` : `query_uuid -> réponse de référence`
- `relevant_docs.json` : `query_uuid -> [doc_uuid pertinents]`
- `errors.json` : erreurs/cas problématiques générés pendant l'évaluation

Note importante : dans le dataset actuel, chaque requête a 1 document pertinent annoté.

## 6) Exports produits

- `dataset_eval/result/last/evaluation_results.csv`
- `dataset_eval/result/last/errors.json`
- `dataset_eval/result/last/evaluation_results_{model}_k{K}_thr{XX}_{timestamp}.csv`
- `dataset_eval/result/last/evaluation_summary_{model}_k{K}_thr{XX}_{timestamp}.csv`

Colonnes principales de `evaluation_results.csv` :

- `uuid`, `query`, `gen`, `ref`, `bert_f1`, `retrieved_uuids`, `expected_uuids`, `recall_at_k`, `has_source_citation`

Interprétation de `recall@K` ici : comme il y a un seul doc pertinent par requête, c'est équivalent à un hit@K moyen.

## 7) Résultat final retenu

Configuration finale :
- `K=4`
- `similarity_search`
- chunking `600/60`
- embedding `BAAI/bge-m3`

Métriques (run complet 619 requêtes, test5) :
- Mean BERT-F1 : `0.6876`
- `% BERT-F1 >= 60%` : `84.65%`
- Mean recall@4 : `0.5718901453957996`

Explication du choix :
- Le F1 était déjà > 60% dès les premiers tests, donc l'objectif principal était d'améliorer le retrieval.
- Le changement d'embedding vers `BAAI/bge-m3` apporte un gain majeur sur le recall (`0.3635 -> 0.5719`) avec un niveau de génération qui reste satisfaisant.
- Les différents tests sont détaillés dans le fichier `EXPERIMENTS.md`

## 8) Exécution

1. Ouvrir `rag.ipynb`
2. Exécuter les cellules dans l'ordre
3. Activer/désactiver le mode chat via `RUN_CHAT`
4. Choisir le mode évaluation :
   - rapide : `EVAL_MAX_QUERIES` (ex: 120)
   - complet : `EVAL_MAX_QUERIES = None`
5. Lancer la section évaluation pour générer les exports dans `dataset_eval/result/last/`

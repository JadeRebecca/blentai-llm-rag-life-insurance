# Journal d'expérimentations - Retrieval RAG

Ce document récapitule les tests réalisés pour améliorer la qualité du retrieval.

## Objectif

Améliorer la capacité du moteur de recherche à retrouver les bons documents DIC, mesurée via `recall@K`.

## Paramètres constants (jusqu'ici)

- Modèle LLM : `mistralai/Mistral-7B-Instruct-v0.2`
- Embeddings : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Dataset d'évaluation : `dataset_eval/`
- Seuil BERT-F1 : `0.60`

## Test 1 - Baseline avec K=2

Configuration :
- Retrieval dense standard (`similarity_search`)
- `K = 2`

Résultats observés :
- Nombre de requêtes : `619`
- Mean BERT-F1 : `0.6903`
- `% BERT-F1 >= 60%` : `82.88%`
- Mean recall@2 : `0.25363489499192243`

Conclusion :
- Le retrieval est insuffisant (`~25.36%`), donc beaucoup de requêtes ne retrouvent pas le bon document dans le top-2.

## Test 2 - Augmentation de K à 4

Configuration :
- Retrieval dense standard (`similarity_search`)
- `K = 4`

Résultats observés :
- Nombre de requêtes : `619`
- Mean BERT-F1 : `0.6988`
- `% BERT-F1 >= 60%` : `86.27%`
- Mean recall@4 : `0.36348949919224555`

Conclusion :
- Amélioration nette vs K=2 :
  - BERT-F1 : `+0.0085`
  - `% >= 60%` : `+3.39 points`
  - Recall : `+0.1099`
- Mais le retrieval reste insuffisant (`~36.35%`), d'où un nouveau test.

## Test 3 - MMR seul (test rapide sur 120 requêtes)

Changement appliqué :
- Passage de `similarity_search` à `max_marginal_relevance_search` (MMR)
- Sans changer embedding ni chunking
- Paramètres MMR :
  - `MMR_FETCH_K = 20`
  - `MMR_LAMBDA = 0.5`
- `K = 4`
- Sous-échantillon : `EVAL_MAX_QUERIES = 120`

Résultats observés :
- Nombre de requêtes : `120`
- Mean BERT-F1 : `0.6703`
- `% BERT-F1 >= 60%` : `78.33%`
- Mean recall@4 : `0.325`
- Fichiers de sortie : `dataset_eval/result/last/`

Interprétation :
- Ce run sert d'itération rapide.
- Il n'est pas directement comparable aux tests sur `619` requêtes.
- Sur ce sous-ensemble, le score de retrieval reste modeste.
- Conclusion opérationnelle : MMR seul n'apporte pas d'amélioration dans cette configuration ; la baseline retenue est `K=4` avec `similarity_search`.

## Prochaine étape

Décision actuelle :
- conserver `K=4` + `similarity_search`
- revenir au chunking initial (`600/60`)

Piste d'amélioration suivante (prioritaire) :
- tester un changement d'embeddings (ex: `BAAI/bge-m3`) avec le même protocole
- comparer sur run rapide puis valider sur 619 requêtes

## Test 4 - Chunking plus fin (400/100)

Changement appliqué :
- Baseline conservée : `K=4` + `similarity_search`
- Chunking modifié : `CHUNK_SIZE=400`, `CHUNK_OVERLAP=100`
- Sous-échantillon : `EVAL_MAX_QUERIES=120`

Résultats observés :
- Nombre de requêtes : `120`
- Mean BERT-F1 : `0.6691`
- `% BERT-F1 >= 60%` : `74.17%`
- Mean recall@4 : `0.35`
- Fichiers de sortie : `dataset_eval/result/last/`

Interprétation :
- Par rapport au test MMR rapide (120 requêtes), le recall progresse (`0.325 -> 0.35`).
- Le BERT-F1 reste légèrement plus bas, et le `% >=60%` baisse.
- Le gain retrieval est réel mais modéré ; à confirmer sur le dataset complet (`EVAL_MAX_QUERIES=None`).

Run complet (619 requêtes) :
- Nombre de requêtes : `619`
- Mean BERT-F1 : `0.6717`
- `% BERT-F1 >= 60%` : `78.68%`
- Mean recall@4 : `0.36187399030694667`
- Fichiers de sortie : `dataset_eval/result/last/`

Comparaison au baseline `K=4` (sans chunking 400/100) :
- Recall@4 : `0.36348949919224555 -> 0.36187399030694667` (légère baisse)
- Mean BERT-F1 : `0.6988 -> 0.6717` (baisse)
- `% >=60%` : `86.27% -> 78.68%` (baisse)

Conclusion test 4 :
- Le chunking `400/100` n'améliore pas la configuration actuelle sur run complet.
- La baseline retenue reste `K=4` avec chunking initial (`600/60`) et `similarity_search`.

## Statut de fin de session (point d'arrêt)

### Où on s'est arrêté

- Baseline validée : `K=4` + `similarity_search` + chunking initial `600/60`.
- Test MMR : non retenu (pas d'amélioration).
- Test chunking `400/100` : non retenu sur run complet (`619`).
- Dernier résultat complet observé (chunking 400/100) :
  - Mean BERT-F1 : `0.6717`
  - `% >= 60%` : `78.68%`
  - Mean recall@4 : `0.36187399030694667`
- Dossier de sortie courant : `dataset_eval/result/last/`.

### Reprise exacte la prochaine fois

1. Remettre la config baseline dans `rag.ipynb` :
   - `K = 4`
   - `CHUNK_SIZE = 600`
   - `CHUNK_OVERLAP = 60`
   - `EVAL_MAX_QUERIES = 120` pour itération rapide (puis `None` pour validation finale)
2. Conserver `similarity_search` (ne pas réactiver MMR).
3. Lancer le prochain test prioritaire : changement d'embeddings (ex: `BAAI/bge-m3`).
4. Comparer les métriques au baseline (`recall@4`, `Mean BERT-F1`, `% >= 60%`).
5. Si gain en rapide, valider sur `619` requêtes.

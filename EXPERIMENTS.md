# Journal d'expérimentations - Retrieval RAG

Ce document récapitule les tests réalisés pour améliorer la qualité du retrieval.

## Objectif

Améliorer la capacité du moteur de recherche à retrouver les bons documents DIC, mesurée via `recall@K`.

Dès les premiers essais, le F1 BERTScore est au-dessus du seuil attendu de 60%, ce qui valide la qualité minimale de génération. L'objectif des expérimentations suivantes est donc d'améliorer en priorité la qualité du retrieval, car le `mean recall` reste insuffisant.

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
- Dossier d'exports (path relatif) : `dataset_eval/result/test1/`

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
- Dossier d'exports (path relatif) : `dataset_eval/result/test2/`

Conclusion :
- Amélioration nette vs K=2 :
  - BERT-F1 : `+0.0085`
  - `% >= 60%` : `+3.39 points`
  - Recall : `+0.1099`
- Mais le retrieval reste insuffisant (`~36.35%`), d'où un nouveau test.

## Test 3 - MMR seul (test rapide sur 120 requêtes)

Pourquoi ce test :
- Avec `similarity_search`, on prend les K chunks les plus similaires (souvent redondants).
- Avec `max_marginal_relevance_search` (MMR), on cherche un compromis entre similarité à la question et diversité des chunks retournés.
- L'objectif était d'éviter des résultats trop proches entre eux pour augmenter les chances d'inclure le document pertinent dans le top-4.

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
- Dossier d'exports (path relatif) : `dataset_eval/result/test3/`

Interprétation :
- Ce run sert d'itération rapide.
- Il n'est pas directement comparable aux tests sur `619` requêtes.
- Sur ce sous-ensemble, le score de retrieval reste modeste.
- Conclusion opérationnelle : MMR seul n'apporte pas d'amélioration dans cette configuration ; la baseline retenue est `K=4` avec `similarity_search`.

## Test 4 - Chunking plus fin (400/100)

Pourquoi ce test :
- Le chunking contrôle la granularité des informations indexées dans la base vectorielle.
- Avec des chunks plus petits (`400`) et plus de recouvrement (`100`), on cherche à mieux cibler les passages pertinents pour chaque question.
- L'objectif est d'améliorer le `recall@4` en réduisant le bruit sémantique, tout en conservant assez de contexte grâce au chevauchement.

Changement appliqué :
- Baseline conservée : `K=4` + `similarity_search`
- Chunking modifié : `CHUNK_SIZE=400`, `CHUNK_OVERLAP=100`
- Sous-échantillon : `EVAL_MAX_QUERIES=120`

Résultats observés (run rapide 120) :
- Nombre de requêtes : `120`
- Mean BERT-F1 : `0.6691`
- `% BERT-F1 >= 60%` : `74.17%`
- Mean recall@4 : `0.35`
- Dossier d'exports (path relatif) : `dataset_eval/result/test4/`

Interprétation (run rapide) :
- Par rapport au test MMR rapide (120 requêtes), le recall progresse (`0.325 -> 0.35`).
- Le BERT-F1 reste légèrement plus bas, et le `% >=60%` baisse.

Résultats observés (run complet 619) :
- Nombre de requêtes : `619`
- Mean BERT-F1 : `0.6717`
- `% BERT-F1 >= 60%` : `78.68%`
- Mean recall@4 : `0.36187399030694667`
- Dossier d'exports (path relatif) : `dataset_eval/result/test4/`

Comparaison au baseline `K=4` (sans chunking 400/100) :
- Recall@4 : `0.36348949919224555 -> 0.36187399030694667` (légère baisse)
- Mean BERT-F1 : `0.6988 -> 0.6717` (baisse)
- `% >=60%` : `86.27% -> 78.68%` (baisse)

Conclusion test 4 :
- Le chunking `400/100` n'améliore pas la configuration actuelle sur run complet.
- La baseline retenue reste `K=4` avec chunking initial (`600/60`) et `similarity_search`.

## Test 5 - Changement d'embedding (`BAAI/bge-m3`)

Pourquoi ce test :
- Après les tests sur `K`, MMR et chunking, le principal levier restant pour améliorer le retrieval est la qualité des embeddings.
- Les embeddings déterminent directement la proximité sémantique entre questions et chunks dans l'espace vectoriel.
- `BAAI/bge-m3` est réputé performant en recherche sémantique multilingue (dont le français).
- L'objectif est d'augmenter la probabilité de retrouver le document pertinent dans le top-4, à pipeline identique.

Changement appliqué :
- Baseline conservée : `K=4` + `similarity_search`
- Chunking baseline : `600/60`
- Embedding remplacé par `BAAI/bge-m3`
- Sous-échantillon : `EVAL_MAX_QUERIES=120`

Résultats observés (run rapide 120) :
- Nombre de requêtes : `120`
- Mean BERT-F1 : `0.6893`
- `% BERT-F1 >= 60%` : `82.50%`
- Mean recall@4 : `0.5166666666666667`
- Dossier d'exports (path relatif) : `dataset_eval/result/test5/`

Comparaison au dernier test rapide (chunking 400/100) :
- Recall@4 : `0.35 -> 0.5167` (forte hausse)
- Mean BERT-F1 : `0.6691 -> 0.6893` (hausse)
- `% >=60%` : `74.17% -> 82.50%` (hausse)

Conclusion test 5 (intermédiaire) :
- Le changement d'embedding apporte une amélioration nette du retrieval et de la qualité globale sur run rapide.
- Validation finale à faire sur `619` requêtes (`EVAL_MAX_QUERIES=None`).

Résultats observés (run complet 619) :
- Nombre de requêtes : `619`
- Mean BERT-F1 : `0.6876`
- `% BERT-F1 >= 60%` : `84.65%`
- Mean recall@4 : `0.5718901453957996`
- Dossier d'exports (path relatif) : `dataset_eval/result/test5/`

Comparaison au baseline `K=4` (embedding initial) :
- Recall@4 : `0.36348949919224555 -> 0.5718901453957996` (forte hausse)
- Mean BERT-F1 : `0.6988 -> 0.6876` (légère baisse)
- `% >=60%` : `86.27% -> 84.65%` (légère baisse)

Conclusion test 5 (finale) :
- Le changement d'embedding améliore fortement le retrieval tout en maintenant un niveau de génération globalement satisfaisant (> 60%).
## Conclusion actuelle

Configuration retenue (après validation complète) :
- `K=4`
- `similarity_search`
- chunking `600/60`
- embedding `BAAI/bge-m3`

Motif de décision :
- amélioration majeure du retrieval (`mean recall@4 = 0.5719`)
- maintien d'un niveau de génération satisfaisant (`Mean BERT-F1 = 0.6876`, `% >= 60% = 84.65%`)

## Index des exports (paths relatifs)

- Test 1 (`K=2`) : `dataset_eval/result/test1/`
- Test 2 (`K=4`) : `dataset_eval/result/test2/`
- Test 3 (`MMR`) : `dataset_eval/result/test3/`
- Test 4 (`chunking 400/100`) : `dataset_eval/result/test4/`
- Test 5 (`embedding BAAI/bge-m3`) : `dataset_eval/result/test5/`

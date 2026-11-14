# Automated Taxonomy Generator

Unsupervised construction of a product taxonomy for e-commerce data. The pipeline groups products from the Icecat catalogue into categories using semantic text embeddings and multi-stage clustering, without using any labels during clustering.

Case study by Bharat Manjunath Gangolli, M.Sc. Applied Data Science and Analytics, SRH University Heidelberg.

## Why

Assigning products to categories by hand is slow, inconsistent and hard to scale. This project tests whether products can be grouped automatically from their text alone, so that catalogue teams review and name ready-made groups instead of categorising every product from scratch.

## Results

Clusters are compared with the reference Level-3 category labels. The labels are used only for evaluation.

| Configuration | Purity | NMI | ARI | AMI | Clusters |
|---|---|---|---|---|---|
| Purity-optimised | 0.8901 | 0.6669 | 0.0964 | 0.6621 | 366 |
| Balanced | 0.8667 | 0.7270 | 0.3453 | 0.7242 | 238 |

The purity-optimised setup gives the cleanest clusters but splits categories into many small groups. The balanced setup gives up about 2.3 percentage points of purity and matches the overall taxonomy structure much better (ARI more than three times higher).

## Pipeline

1. **Product documents** – brand, title, product family, descriptions and bullet points are cleaned and combined into one text per product (max. 1,200 characters per field). Numeric specs such as GB, mAh, Hz and screen size are extracted and appended as tokens.
2. **Embeddings** – each document is embedded with `BAAI/bge-m3` via sentence-transformers (1024 dimensions, L2-normalised, max. sequence length 256).
3. **BIRCH micro-clustering** – compresses the embedding space into micro-clusters (final threshold 0.70).
4. **Centroid graph** – a k-nearest-neighbour similarity graph over the micro-cluster centroids, built with FAISS (k = 80).
5. **Leiden community detection** – finds communities in the graph (resolution 60.0 for the purity-optimised setup).
6. **Merge step (balanced setup only)** – a second Leiden pass merges very similar clusters (merge threshold 0.75, resolution 0.5).
7. **Evaluation** – Purity, NMI, ARI and AMI against the Level-3 labels.

28 combinations of graph threshold and Leiden resolution were evaluated during tuning.

## Repository contents

| File | Description |
|---|---|
| `taxonomy-final-submission.ipynb` | Final notebook with the full pipeline |
| `76_percent_Purity.ipynb`, `90_plus_.ipynb` | Earlier experiments |
| `Taxonomy_new.ipynb`, `Taxonomy_new(changes_updated).ipynb`, `New_work_updated.ipynb` | Earlier development versions |
| `icecat_taxonomy_results.csv` | Results of an early baseline (K-Means and DBSCAN cluster assignments) |

## Tech stack

Python, pandas, NumPy, sentence-transformers (BGE-M3), scikit-learn (BIRCH, PCA, metrics), FAISS, igraph, leidenalg, joblib.

## How to run

The notebooks were written for Kaggle with a GPU (Tesla P100).

1. Add the Icecat product dataset (JSON) to the notebook and set `JSON_PATH`.
2. Install the extra libraries: `pip install sentence-transformers faiss-cpu python-igraph leidenalg`.
3. Run `taxonomy-final-submission.ipynb` from top to bottom. Embeddings are saved to disk, so the clustering stages can be re-run with other parameters without recomputing them.

## Limitations

- Categories with very similar vocabulary (for example laptop stands and laptop cooling pads) are hard to separate.
- Short or sparse product descriptions give weak embeddings.
- The output is meant to support human taxonomy review, not replace it.

# Word Embedding Visualization with PCA

## Overview
This project involves implementing Principal Component Analysis (PCA) from scratch to historical word embeddings to try to visualize semantic shifts in word meanings over time. Using embeddings loaded from the [HistWords dataset](https://nlp.stanford.edu/projects/histwords/), we can observe how words evolve in meaning across decades.

## Objective
- Reduce the dimensionality of word embeddings on different time periods.
- Visualize changes in semantic relationships of words over time.
- Quantify the "volatility" of certain words over time.

## Approach

### Data Preparation
- Historical word embeddings for each decade (e.g., 1880 to 1980) were used.
- The embeddings were subsampled, standardized, and reduced using PCA.

### PCA Implementation Steps
1. **Standardization**: Center data by subtracting mean and scaling by standard deviation.
2. **Covariance Matrix Calculation**: Compute covariance matrix for standardized data.
3. **Eigenvalue and Eigenvector Calculation**: Perform eigen decomposition on covariance matrix to find principal components.
4. **Dimensionality Reduction**: Select top components to transform data to a lower-dimensional space.

### Visualization
- Utilized Plotly for interactive 3D visualizations of word embeddings over time.
- Created animation frames to track how specific words move in the reduced-dimensional space across different decades.

## Dataset
- **Source**: [HistWords Dataset](https://nlp.stanford.edu/projects/histwords/)
- **Structure**: Word embeddings from the 1800s to the 1990s, with each decade represented as a ```vocab.pkl``` pickled file and a ```w.npy``` NumPy array file.

## Challenges
- Handling large embedding files efficiently.
- Optimizing manual PCA computation for high-dimensional datasets.
- Visualizing complex semantic changes in a comprehensible manner.

## How to Use
1. Load the dataset (`sgns` folder) before running the `pca_word_embeddings.ipynb` notebook.
2. Run the notebook to execute the PCA process to reduce dimensionality and generate visualizations.
3. Modify the target words, sample size, visualization parameters, etc. as desired.

## Future Directions
- Apply clustering algorithms (like k-means, DBSCAN, or GMM) to group words by semantic similarity and color code those groups.
- Explore additional dimensionality reduction techniques (like SVD, t-SNE, or UMAP).
- Use temporal analysis with PCA on languages other than English.
- Apply dimensionality reduction on an expanded sample size using more powerful hardware.

=======

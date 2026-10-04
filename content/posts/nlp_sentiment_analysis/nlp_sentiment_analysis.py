# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gensim>=4.4.0",
#     "ipython>=9.17.1",
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.2",
#     "nltk>=3.10.3",
#     "numpy>=2.5.3",
#     "pandas>=3.0.6",
#     "scikit-learn>=1.9.1",
#     "tabulate>=0.10.0",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import re
    import string
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from IPython.display import display

    # Ensure reproducibility
    SEED = 42
    np.random.seed(SEED)

    # Download required NLTK resource packages
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Set global plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.titlesize'] = 14

    print("Libraries imported and NLTK resources initialized successfully.")
    return SEED, display, metrics, np, pd, plt, re, stopwords, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Data Preprocessing
    #### Step 1.1: Dataset Ingestion and Schema Inspection
    """)
    return


@app.cell
def _(display, pd):
    # Load the dataset
    dataset_path = 'amazon_reviews.csv'
    df_raw = pd.read_csv(dataset_path)

    print("=== RAW DATASET OVERVIEW ===")
    print(f"Total Rows: {df_raw.shape[0]}")
    print(f"Total Columns: {df_raw.shape[1]}")
    print(f"Column Names: {list(df_raw.columns)}\n")

    print("Missing values per column:")
    print(df_raw.isnull().sum())
    print("\nFirst 5 rows of raw dataset:")
    display(df_raw.head())
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 1.2: Missing Value Handling & Star Rating Distribution
    """)
    return


@app.cell
def _(df_raw, display, pd):
    # Drop records with missing review text
    df_clean = df_raw.dropna(subset=['reviewText', 'overall']).reset_index(drop=True)
    print(f"Samples remaining after dropping missing review text: {len(df_clean)}")

    # Raw star rating counts and percentages
    rating_counts = df_clean['overall'].value_counts().sort_index()
    rating_pcts = (df_clean['overall'].value_counts(normalize=True).sort_index()) * 100

    rating_summary = pd.DataFrame({
        'Star Rating': rating_counts.index,
        'Count': rating_counts.values,
        'Percentage (%)': rating_pcts.values.round(2),
        'Sentiment Class Assignment': [
            'Negative (0)',
            'Negative (0)',
            'Neutral (Discarded)',
            'Positive (1)',
            'Positive (1)'
        ]
    })

    print("\nRaw Star Rating Summary:")
    display(rating_summary)
    return df_clean, rating_counts, rating_summary


@app.cell
def _(df_clean, plt, rating_counts, rating_summary):
    # Visualize raw star ratings distribution
    _fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(rating_summary['Star Rating'], rating_summary['Count'], color=['#d9534f', '#f0ad4e', '#6c757d', '#5bc0de', '#5cb85c'], edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}\n({height / len(df_clean):.1%})', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9.5)
    ax.set_title('Raw Amazon Review Star Rating Distribution (1 to 5 Stars)', fontweight='bold', pad=12)
    ax.set_xlabel('Overall Star Rating', fontweight='bold')
    ax.set_ylabel('Number of Reviews', fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, max(rating_counts) * 1.15)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 1.3: Label Preparation & Filtering
    """)
    return


@app.cell
def _(df_clean):
    # Discard neutral rating 3
    df_binary = df_clean[df_clean['overall'] != 3].copy().reset_index(drop=True)

    # Assign binary sentiment labels
    df_binary['sentiment'] = df_binary['overall'].apply(lambda r: 1 if r in [4, 5] else 0)

    print(f"Total reviews in binary dataset: {len(df_binary)}")
    print(f"  - Positive reviews (ratings 4 & 5): {(df_binary['sentiment'] == 1).sum()} ({(df_binary['sentiment'] == 1).mean():.2%})")
    print(f"  - Negative reviews (ratings 1 & 2): {(df_binary['sentiment'] == 0).sum()} ({(df_binary['sentiment'] == 0).mean():.2%})")
    return (df_binary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 1.4: Text Preprocessing Pipeline
    """)
    return


@app.cell
def _(df_binary, re, stopwords):
    # Define English stop words set
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text: str) -> list[str]:
        """
        Preprocess raw review text according to HW1 requirements:
          (i) Convert text to lowercase
          (ii) Remove punctuations and numbers
          (iii) Remove stopwords
          (iv) Tokenize into word tokens

        Parameters:
            text (str): Raw review text
        Returns:
            list[str]: Cleaned list of word tokens
        """
        text_lower = str(text).lower()  # (i) Convert to lowercase
        text_alpha = re.sub('[^a-z\\s]', ' ', text_lower)
        raw_tokens = text_alpha.split()
        filtered_tokens = [w for w in raw_tokens if w not in stop_words]  # (ii) Remove punctuations and numbers (keep only a-z letters and whitespace)
        return filtered_tokens


    df_binary['tokens'] = df_binary['reviewText'].apply(preprocess_text)
    df_binary['token_count'] = df_binary['tokens'].apply(len)  # (iii) Tokenize
    df_binary['clean_text'] = df_binary['tokens'].apply(lambda toks: ' '.join(toks))
    empty_token_count = (df_binary['token_count'] == 0).sum()
    print(f'Reviews resulting in 0 tokens after preprocessing: {empty_token_count}')  # (iv) Remove stopwords

    df_binary_1 = df_binary[df_binary['token_count'] > 0].reset_index(drop=True)
    sample_df = df_binary_1[['overall', 'sentiment', 'reviewText', 'tokens', 'token_count']].head(5)
    for idx, row in sample_df.iterrows():
        print(f"\n--- Sample {idx + 1} (Rating: {row['overall']}★, Sentiment: {('Positive' if row['sentiment'] == 1 else 'Negative')}) ---")
    # Apply preprocessing to all reviews
        print(f"Raw Text:      {row['reviewText'][:120]}..." if len(row['reviewText']) > 120 else f"Raw Text:      {row['reviewText']}")
    # Verify and ensure all reviews have at least 1 token
    # Display sample of preprocessed reviews
        print(f"Tokens ({row['token_count']}):    {row['tokens'][:10]}..." if row['token_count'] > 10 else f"Tokens ({row['token_count']}):    {row['tokens']}")
    return (df_binary_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Data Split
    """)
    return


@app.cell
def _(SEED, df_binary_1, train_test_split):
    # Stage 1: Split into 80% train and 20% temp
    train_df, temp_df = train_test_split(df_binary_1, test_size=0.2, random_state=SEED, stratify=df_binary_1['sentiment'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['sentiment'])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    n_total = len(df_binary_1)
    print('=== DATA SPLIT VERIFICATION ===')
    # Stage 2: Split 20% temp equally into 10% validation and 10% test
    print(f'Total Dataset:     {n_total:5d} samples (100.0%)')
    print(f'Training Set:      {len(train_df):5d} samples ({len(train_df) / n_total * 100:.1f}%)')
    print(f'Validation Set:    {len(val_df):5d} samples ({len(val_df) / n_total * 100:.1f}%)')
    # Reset indices for clean dataframes
    # Verification of split proportions
    print(f'Testing Set:       {len(test_df):5d} samples ({len(test_df) / n_total * 100:.1f}%)')
    return test_df, train_df, val_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Data Statistics
    """)
    return


@app.cell
def _(df_binary_1, display, pd, test_df, train_df, val_df):
    # Helper function to compute complete statistics for a partition
    def compute_partition_statistics(df_partition: pd.DataFrame, partition_name: str) -> dict:
        n_samples = len(df_partition)
        n_pos = int((df_partition['sentiment'] == 1).sum())
        n_neg = int((df_partition['sentiment'] == 0).sum())
        pct_pos = n_pos / n_samples * 100
        pct_neg = n_neg / n_samples * 100
        tokens_series = df_partition['token_count']
        min_tokens = int(tokens_series.min())
        mean_tokens = float(tokens_series.mean())
        median_tokens = float(tokens_series.median())
        max_tokens = int(tokens_series.max())
        std_tokens = float(tokens_series.std())
        unique_words = len(set((w for review in df_partition['tokens'] for w in review)))
        total_tokens = int(tokens_series.sum())
        return {'Dataset Partition': partition_name, 'Samples': n_samples, 'Positive Reviews (4-5★)': f'{n_pos} ({pct_pos:.2f}%)', 'Negative Reviews (1-2★)': f'{n_neg} ({pct_neg:.2f}%)', 'Min Tokens': min_tokens, 'Avg Tokens': f'{mean_tokens:.2f}', 'Median Tokens': f'{median_tokens:.1f}', 'Max Tokens': max_tokens, 'Std Dev Tokens': f'{std_tokens:.2f}', 'Total Word Tokens': total_tokens, 'Unique Vocabulary': unique_words}
    stats_data = [compute_partition_statistics(train_df, 'Training Set (80%)'), compute_partition_statistics(val_df, 'Validation / Dev Set (10%)'), compute_partition_statistics(test_df, 'Testing Set (10%)'), compute_partition_statistics(df_binary_1, 'Full Binary Dataset (100%)')]
    summary_stats_table = pd.DataFrame(stats_data)
    print('=== COMPREHENSIVE DATASET STATISTICS TABLE ===')
    # Compute statistics across all partitions
    display(summary_stats_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visualizations of Dataset Statistics
    """)
    return


@app.cell
def _(df_binary_1, np, plt, test_df, train_df, val_df):
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    splits = ['Training (80%)', 'Validation (10%)', 'Testing (10%)']
    pos_counts = [(train_df['sentiment'] == 1).sum(), (val_df['sentiment'] == 1).sum(), (test_df['sentiment'] == 1).sum()]
    neg_counts = [(train_df['sentiment'] == 0).sum(), (val_df['sentiment'] == 0).sum(), (test_df['sentiment'] == 0).sum()]
    x = np.arange(len(splits))
    width = 0.35
    rects1 = axes[0].bar(x - width / 2, pos_counts, width, label='Positive (4-5★)', color='#2ca02c', alpha=0.85, edgecolor='black')
    rects2 = axes[0].bar(x + width / 2, neg_counts, width, label='Negative (1-2★)', color='#d62728', alpha=0.85, edgecolor='black')
    axes[0].set_title('(A) Sample Counts & Sentiment Distribution by Split', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_ylabel('Number of Reviews', fontweight='bold')
    axes[0].legend(frameon=True)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for rects in [rects1, rects2]:
        for r in rects:
            h = r.get_height()
            axes[0].annotate(f'{h}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)


    tokens_all = df_binary_1['token_count']
    mean_tok = tokens_all.mean()
    median_tok = tokens_all.median()
    axes[1].hist(tokens_all, bins=50, color='#1f77b4', edgecolor='black', alpha=0.75, range=(0, 200))
    axes[1].axvline(mean_tok, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tok:.1f}')
    axes[1].axvline(median_tok, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_tok:.1f}')
    axes[1].set_title('(B) Review Token Length Distribution (Truncated to 200)', fontweight='bold')
    axes[1].set_xlabel('Number of Tokens per Review', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].legend(frameon=True)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    data_box = [df_binary_1[df_binary_1['sentiment'] == 0]['token_count'], df_binary_1[df_binary_1['sentiment'] == 1]['token_count']]
    box = axes[2].boxplot(data_box, tick_labels=['Negative (0)', 'Positive (1)'], patch_artist=True, showmeans=True, showfliers=False)
    colors = ['#ff9999', '#99ff99']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')


    axes[2].set_title('(C) Token Count Distribution by Sentiment (without fliers)', fontweight='bold')
    axes[2].set_ylabel('Token Count', fontweight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    def get_vacab(corpus: list[list[str]]) -> list[str]:
        """
        Extract the sorted list of all distinct words used in the review corpus.
        Implemented using an efficient Python set comprehension and sorted().

        Parameters:
            corpus (list[list[str]]): List of tokenized reviews
        Returns:
            corpus_words (list[str]): Sorted list of unique vocabulary words
        """
        corpus_words = sorted(list({word for review in corpus for word in review}))
        return corpus_words

    # Alias for standard spelling convenience
    get_vocab = get_vacab
    return (get_vacab,)


@app.cell
def _(df_binary_1, get_vacab):
    # Extract tokenized corpus from preprocessed binary dataset
    corpus = df_binary_1['tokens'].tolist()
    corpus_words = get_vacab(corpus)

    print("=== 1.a: VOCABULARY EXTRACTION ===")
    print(f"Total reviews in corpus: {len(corpus):,}")
    print(f"Total distinct vocabulary words: {len(corpus_words):,}")
    print(f"First 15 vocabulary words: {corpus_words[:15]}")
    print(f"Sample words around middle: {corpus_words[len(corpus_words)//2 : len(corpus_words)//2 + 10]}")
    print(f"Last 10 vocabulary words: {corpus_words[-10:]}")
    return corpus, corpus_words


@app.cell
def _(get_vacab, np):
    def compute_co_occurrence_matrix(corpus: list[list[str]], window_size: int = 4):
        """
        Construct a word-word co-occurrence matrix for a given window size n.
        Considers n words before and n words after the center word.

        Parameters:
            corpus (list[list[str]]): List of tokenized reviews
            window_size (int): Context window radius (default: 4)
        Returns:
            M (np.ndarray): Co-occurrence matrix of shape (num_words, num_words)
            word2index (dict): Dictionary mapping each word to its row/column index in M
        """
        words = get_vacab(corpus)
        num_words = len(words)
        word2index = {word: idx for idx, word in enumerate(words)}
        M = np.zeros((num_words, num_words), dtype=np.float32)

        for review in corpus:
            review_len = len(review)
            for center_i, center_word in enumerate(review):
                center_idx = word2index[center_word]
                start_idx = max(0, center_i - window_size)
                end_idx = min(review_len, center_i + window_size + 1)
                for context_j in range(start_idx, end_idx):
                    if center_i != context_j:
                        context_word = review[context_j]
                        context_idx = word2index[context_word]
                        M[center_idx, context_idx] += 1.0

        return M, word2index

    return (compute_co_occurrence_matrix,)


@app.cell
def _(compute_co_occurrence_matrix, corpus, np):
    print("=== 1.b: COMPUTING CO-OCCURRENCE MATRIX (window_size=4) ===")
    M_co, word2index_co = compute_co_occurrence_matrix(corpus, window_size=4)

    non_zero_entries = int((M_co > 0).sum())
    total_entries = M_co.shape[0] * M_co.shape[1]
    sparsity = 100.0 * (1.0 - (non_zero_entries / total_entries))

    print(f"Matrix M shape: {M_co.shape} ({M_co.shape[0]:,} x {M_co.shape[1]:,})")
    print(f"Total non-zero co-occurrences: {non_zero_entries:,}")
    print(f"Matrix Sparsity: {sparsity:.2f}%")
    print(f"Total co-occurrence count mass: {M_co.sum():,.0f}")
    print(f"Is matrix symmetric? {bool(np.allclose(M_co, M_co.T))}")
    return M_co, word2index_co


@app.cell
def _():
    from sklearn.decomposition import TruncatedSVD

    def reduce_to_k_dim(M, k: int = 2):
        """
        Perform dimensionality reduction on matrix M using Truncated SVD
        to produce k-dimensional embeddings.

        Parameters:
            M (np.ndarray): Input representation matrix
            k (int): Target embedding dimension (default: 2)
        Returns:
            M_reduced (np.ndarray): Reduced matrix of shape (num_words, k)
        """
        svd = TruncatedSVD(n_components=k, n_iter=10, random_state=42)
        M_reduced = svd.fit_transform(M)
        return M_reduced

    return (reduce_to_k_dim,)


@app.cell
def _(M_co, reduce_to_k_dim):
    print("=== 1.c: TRUNCATED SVD REDUCTION TO k=2 DIMENSIONS ===")
    M_reduced_co = reduce_to_k_dim(M_co, k=2)
    print(f"Original co-occurrence shape: {M_co.shape}")
    print(f"Reduced embeddings shape:    {M_reduced_co.shape}")
    return (M_reduced_co,)


@app.cell
def _(plt):
    def plot_embeddings(M_reduced, word2index, words_to_plot, title: str = "Word Embeddings Scatterplot"):
        """
        Plot 2-dimensional word embeddings in a scatterplot for specified words with annotations.

        Parameters:
            M_reduced (np.ndarray): 2D matrix of word vectors (shape: [num_words, 2])
            word2index (dict): Dictionary mapping words to row indices in M_reduced
            words_to_plot (list[str]): List of words to display
            title (str): Title for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        x_vals = []
        y_vals = []

        for word in words_to_plot:
            if word not in word2index:
                print(f"Warning: Word '{word}' not found in vocabulary.")
                continue
            idx = word2index[word]
            x = float(M_reduced[idx, 0])
            y = float(M_reduced[idx, 1])
            x_vals.append(x)
            y_vals.append(y)

            ax.scatter(x, y, color='#d9534f', marker='o', s=120, edgecolors='black', linewidth=1.2, zorder=4)
            ax.annotate(
                word,
                xy=(x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=12,
                fontweight='bold',
                color='#1a252f',
                bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#ced4da", alpha=0.9, lw=0.8),
                zorder=5
            )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Latent Component 1", fontsize=12, fontweight='bold')
        ax.set_ylabel("Latent Component 2", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        if len(x_vals) > 0:
            x_pad = (max(x_vals) - min(x_vals)) * 0.15 + 0.1
            y_pad = (max(y_vals) - min(y_vals)) * 0.15 + 0.1
            ax.set_xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
            ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)

        plt.tight_layout()
        plt.show()

    return (plot_embeddings,)


@app.cell
def _(M_reduced_co, plot_embeddings, word2index_co):
    words_to_plot = ['purchase', 'buy', 'work', 'got', 'ordered', 'received', 'product', 'item', 'deal', 'use']
    print(f"=== 1.d: PLOTTING COUNT-BASED EMBEDDINGS ===")
    print(f"Words to plot: {words_to_plot}")
    plot_embeddings(
        M_reduced_co,
        word2index_co,
        words_to_plot,
        title="1.d: Count-Based SVD Word Embeddings (Amazon Review Corpus, k=2)"
    )
    return (words_to_plot,)


@app.function
def load_embedding_model():
    """
    Load GloVe Vectors (glove-wiki-gigaword-200)
    Return:
        wv_from_bin: KeyedVectors object; 400000 embeddings, each length 200
    """
    import gensim.downloader as api
    print("Loading GloVe 200-dimensional embeddings (glove-wiki-gigaword-200)...")
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


@app.cell
def _():
    print("=== 2.a: LOADING GLOVE EMBEDDINGS ===")
    wv_from_bin = load_embedding_model()
    return (wv_from_bin,)


@app.cell
def _(np):
    def get_matrix_of_vectors(wv_from_bin, required_words):
        """
        Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
            required_words: Words to ensure are present in matrix M
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
        """
        import random
        words = list(wv_from_bin.index_to_key)
        print("Shuffling words ...")
        random.seed(225)
        random.shuffle(words)
        words = words[:10000]
        print("Putting %i words into word2ind and matrix M..." % len(words))
        word2ind = {}
        M = []
        curInd = 0
        for w in words:
            try:
                M.append(wv_from_bin.get_vector(w))
                word2ind[w] = curInd
                curInd += 1
            except KeyError:
                continue
        words_set = set(words)
        for w in required_words:
            if w in words_set:
                continue
            try:
                M.append(wv_from_bin.get_vector(w))
                word2ind[w] = curInd
                curInd += 1
            except KeyError:
                continue
        M = np.stack(M)
        print("Done.")
        return M, word2ind

    return (get_matrix_of_vectors,)


@app.cell
def _(corpus_words, get_matrix_of_vectors, wv_from_bin):
    print("=== 2.b: EXTRACTING GLOVE VECTOR MATRIX ===")
    M_glove, word2ind_glove = get_matrix_of_vectors(wv_from_bin, required_words=corpus_words)
    print(f"GloVe matrix M shape: {M_glove.shape} (words x dimensions)")
    print(f"Total words indexed in word2ind: {len(word2ind_glove):,}")
    return M_glove, word2ind_glove


@app.cell
def _(M_glove, reduce_to_k_dim):
    print("=== 2.c: TRUNCATED SVD REDUCTION ON GLOVE VECTORS (k=2) ===")
    M_reduced_glove = reduce_to_k_dim(M_glove, k=2)
    print(f"Original GloVe matrix shape: {M_glove.shape}")
    print(f"Reduced GloVe matrix shape:  {M_reduced_glove.shape}")
    return (M_reduced_glove,)


@app.cell
def _(M_reduced_glove, plot_embeddings, word2ind_glove, words_to_plot):
    print("=== 2.d: PLOTTING GLOVE EMBEDDINGS ===")
    plot_embeddings(
        M_reduced_glove,
        word2ind_glove,
        words_to_plot,
        title="2.d: GloVe 2D Word Embeddings (glove-wiki-gigaword-200, k=2)"
    )
    return


@app.cell
def _(
    M_reduced_co,
    M_reduced_glove,
    plt,
    word2ind_glove,
    word2index_co,
    words_to_plot,
):
    # Side-by-side comparative visualization for 1)d vs 2)d
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 7))

    # Subplot A: Count-based SVD
    _x_co = [float(M_reduced_co[word2index_co[w], 0]) for w in words_to_plot]
    _y_co = [float(M_reduced_co[word2index_co[w], 1]) for w in words_to_plot]
    _axes[0].scatter(_x_co, _y_co, color='#d9534f', s=120, edgecolors='black', linewidth=1.2, zorder=4)
    for _w, _x_val, _y_val in zip(words_to_plot, _x_co, _y_co):
        _axes[0].annotate(
            _w, xy=(_x_val, _y_val), xytext=(8, 8), textcoords="offset points",
            fontsize=11, fontweight='bold', color='#1a252f',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#ced4da", alpha=0.9, lw=0.8),
            zorder=5
        )
    _axes[0].set_title("(A) 1.d: Count-Based SVD Embeddings (Amazon Corpus)", fontsize=13, fontweight='bold')
    _axes[0].set_xlabel("SVD Component 1", fontweight='bold')
    _axes[0].set_ylabel("SVD Component 2", fontweight='bold')
    _axes[0].grid(True, linestyle='--', alpha=0.6)

    # Subplot B: GloVe SVD
    _x_gl = [float(M_reduced_glove[word2ind_glove[w], 0]) for w in words_to_plot]
    _y_gl = [float(M_reduced_glove[word2ind_glove[w], 1]) for w in words_to_plot]
    _axes[1].scatter(_x_gl, _y_gl, color='#0275d8', s=120, edgecolors='black', linewidth=1.2, zorder=4)
    for _w, _x_val, _y_val in zip(words_to_plot, _x_gl, _y_gl):
        _axes[1].annotate(
            _w, xy=(_x_val, _y_val), xytext=(8, 8), textcoords="offset points",
            fontsize=11, fontweight='bold', color='#1a252f',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#ced4da", alpha=0.9, lw=0.8),
            zorder=5
        )
    _axes[1].set_title("(B) 2.d: Prediction-Based GloVe Embeddings (Wiki+Gigaword 6B)", fontsize=13, fontweight='bold')
    _axes[1].set_xlabel("SVD Component 1", fontweight='bold')
    _axes[1].set_ylabel("SVD Component 2", fontweight='bold')
    _axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Comparative Analysis: Count-Based SVD vs. Pre-trained GloVe Embeddings", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2)d. Comparative Analysis: Count-Based SVD vs. Pre-trained GloVe Embeddings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Sentiment Classification Algorithms
    ## Perform Sentiment Analysis with Classification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Review Embeddings
    """)
    return


@app.cell
def _(M_glove, reduce_to_k_dim):
    print("=== TASK 2.1: SVD REDUCTION TO 128 DIMENSIONS ===")
    M_glove_128 = reduce_to_k_dim(M_glove, k=128)
    print(f"Original GloVe matrix shape: {M_glove.shape} (words x 200 dims)")
    print(f"Reduced GloVe matrix shape:  {M_glove_128.shape} (words x 128 dims)")
    return (M_glove_128,)


@app.cell
def _(np):
    def get_review_embedding(
        tokens: list[str],
        word2index: dict[str, int],
        word_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute review embedding by taking the element-wise average of word embeddings.

        Parameters:
            tokens (list[str]): List of preprocessed word tokens in the review.
            word2index (dict[str, int]): Dictionary mapping word string to embedding row index.
            word_embeddings (np.ndarray): Matrix of word embeddings of shape (vocab_size, embedding_dim).

        Returns:
            np.ndarray: 1D numpy array of shape (embedding_dim,) representing the review embedding.
                        Returns a zero vector of the same dimension if no tokens match.
        """
        valid_vectors = [word_embeddings[word2index[w]] for w in tokens if w in word2index]
        if len(valid_vectors) == 0:
            return np.zeros(word_embeddings.shape[1], dtype=np.float32)
        return np.mean(valid_vectors, axis=0).astype(np.float32)

    return (get_review_embedding,)


@app.cell
def _(
    M_glove_128,
    get_review_embedding,
    np,
    test_df,
    train_df,
    val_df,
    word2ind_glove,
):
    print("=== COMPUTING 128-DIMENSIONAL REVIEW EMBEDDINGS ===")
    X_train = np.array([get_review_embedding(toks, word2ind_glove, M_glove_128) for toks in train_df['tokens']])
    X_val = np.array([get_review_embedding(toks, word2ind_glove, M_glove_128) for toks in val_df['tokens']])
    X_test = np.array([get_review_embedding(toks, word2ind_glove, M_glove_128) for toks in test_df['tokens']])

    y_train = train_df['sentiment'].values.astype(int)
    y_val = val_df['sentiment'].values.astype(int)
    y_test = test_df['sentiment'].values.astype(int)

    print(f"X_train shape: {X_train.shape} | y_train: {len(y_train)} (Pos: {(y_train == 1).sum()}, Neg: {(y_train == 0).sum()})")
    print(f"X_val shape:   {X_val.shape} | y_val:   {len(y_val)} (Pos: {(y_val == 1).sum()}, Neg: {(y_val == 0).sum()})")
    print(f"X_test shape:  {X_test.shape} | y_test:  {len(y_test)} (Pos: {(y_test == 1).sum()}, Neg: {(y_test == 0).sum()})")
    print(f"NaN Check -> X_train: {np.isnan(X_train).any()}, X_val: {np.isnan(X_val).any()}, X_test: {np.isnan(X_test).any()}")
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Model Training & Selection on Validation Set

    #### 1. Logistic Regression with L2 Regularization:
    """)
    return


@app.cell
def _(SEED, X_train, X_val, display, metrics, pd, y_train, y_val):
    import warnings as _warnings
    from sklearn.linear_model import LogisticRegression as _LogisticRegression

    print("=== TASK 2.2.a: LOGISTIC REGRESSION (L2 REGULARIZATION) TUNING ===")
    _c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    _lr_tuning_results = []
    _lr_candidate_models = {}

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for _c in _c_values:
            _model = _LogisticRegression(
                C=_c,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=SEED,
            )
            _model.fit(X_train, y_train)
            _lr_candidate_models[_c] = _model

            _v_pred = _model.predict(X_val)
            _v_prob = _model.predict_proba(X_val)[:, 1]

            _acc = metrics.accuracy_score(y_val, _v_pred)
            _prec = metrics.precision_score(y_val, _v_pred, zero_division=0)
            _rec = metrics.recall_score(y_val, _v_pred, zero_division=0)
            _f1 = metrics.f1_score(y_val, _v_pred, zero_division=0)
            _auc_val = metrics.roc_auc_score(y_val, _v_prob)

            _lr_tuning_results.append({
                'Regularization C': _c,
                'Val Accuracy': f"{_acc:.4f}",
                'Val Precision': f"{_prec:.4f}",
                'Val Recall': f"{_rec:.4f}",
                'Val F1-Score': f"{_f1:.4f}",
                'Val ROC-AUC': f"{_auc_val:.4f}",
            })

    lr_tuning_df = pd.DataFrame(_lr_tuning_results)
    print("Logistic Regression Performance Across Regularization Strength C (Validation Set):")
    display(lr_tuning_df)

    _best_lr_row = lr_tuning_df.loc[lr_tuning_df['Val F1-Score'].astype(float).idxmax()]
    best_lr_c = float(_best_lr_row['Regularization C'])
    best_lr_model = _lr_candidate_models[best_lr_c]
    print(f"\nSelected Best Logistic Regression: C = {best_lr_c} (Val F1: {_best_lr_row['Val F1-Score']}, Val ROC-AUC: {_best_lr_row['Val ROC-AUC']})")
    return best_lr_c, best_lr_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2. Neural Network (NN) Model for Sentiment Classification:
    """)
    return


@app.cell
def _(SEED, X_train, X_val, display, metrics, pd, y_train, y_val):
    from sklearn.neural_network import MLPClassifier as _MLPClassifier

    print("=== TASK 2.2.b: NEURAL NETWORK ARCHITECTURE TUNING ===")
    _candidate_architectures = [
        (32,),
        (64,),
        (64, 32),
        (128, 64),
    ]
    _nn_tuning_results = []
    _nn_candidate_models = {}

    for _arch in _candidate_architectures:
        _mlp = _MLPClassifier(
            hidden_layer_sizes=_arch,
            activation='relu',
            solver='adam',
            alpha=1e-4,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )
        _mlp.fit(X_train, y_train)
        _nn_candidate_models[_arch] = _mlp

        _v_pred = _mlp.predict(X_val)
        _v_prob = _mlp.predict_proba(X_val)[:, 1]

        _acc = metrics.accuracy_score(y_val, _v_pred)
        _prec = metrics.precision_score(y_val, _v_pred, zero_division=0)
        _rec = metrics.recall_score(y_val, _v_pred, zero_division=0)
        _f1 = metrics.f1_score(y_val, _v_pred, zero_division=0)
        _auc_val = metrics.roc_auc_score(y_val, _v_prob)

        _nn_tuning_results.append({
            'Hidden Architecture': str(_arch),
            'Val Accuracy': f"{_acc:.4f}",
            'Val Precision': f"{_prec:.4f}",
            'Val Recall': f"{_rec:.4f}",
            'Val F1-Score': f"{_f1:.4f}",
            'Val ROC-AUC': f"{_auc_val:.4f}",
            'Converged Epochs': _mlp.n_iter_,
        })

    nn_tuning_df = pd.DataFrame(_nn_tuning_results)
    print("Neural Network Performance Across Hidden Architectures (Validation Set):")
    display(nn_tuning_df)

    _best_nn_row = nn_tuning_df.loc[nn_tuning_df['Val F1-Score'].astype(float).idxmax()]
    best_nn_arch = eval(_best_nn_row['Hidden Architecture'])
    best_nn_model = _nn_candidate_models[best_nn_arch]
    print(f"\nSelected Best Neural Network Architecture: {best_nn_arch} (Val F1: {_best_nn_row['Val F1-Score']}, Val ROC-AUC: {_best_nn_row['Val ROC-AUC']})")
    return best_nn_arch, best_nn_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Evaluation on Independent Testing Set
    """)
    return


@app.cell
def _(
    X_test,
    best_lr_c,
    best_lr_model,
    best_nn_arch,
    best_nn_model,
    display,
    metrics,
    pd,
    y_test,
):
    print("=== TASK 2.3.a: FINAL TESTING SET EVALUATION ===")
    _eval_models = [
        (f"Logistic Regression (L2, C={best_lr_c})", best_lr_model),
        (f"Neural Network MLP {best_nn_arch}", best_nn_model),
    ]

    _test_records = []
    for _model_name, _model in _eval_models:
        _preds = _model.predict(X_test)
        _probs = _model.predict_proba(X_test)[:, 1]

        _acc = metrics.accuracy_score(y_test, _preds)
        _prec = metrics.precision_score(y_test, _preds, zero_division=0)
        _rec = metrics.recall_score(y_test, _preds, zero_division=0)
        _f1 = metrics.f1_score(y_test, _preds, zero_division=0)
        _auc_score = metrics.roc_auc_score(y_test, _probs)

        _test_records.append({
            'Model': _model_name,
            'Accuracy': f"{_acc:.4f}",
            'Precision': f"{_prec:.4f}",
            'Recall': f"{_rec:.4f}",
            'F1-Score': f"{_f1:.4f}",
            'ROC-AUC': f"{_auc_score:.4f}",
        })

    test_metrics_table = pd.DataFrame(_test_records)
    print("Final Model Performance Comparison on Held-Out Testing Set:")
    display(test_metrics_table)
    return


@app.cell
def _(
    X_test,
    best_lr_c,
    best_lr_model,
    best_nn_arch,
    best_nn_model,
    metrics,
    plt,
    y_test,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Plot A: ROC Curves
    _lr_probs = best_lr_model.predict_proba(X_test)[:, 1]
    _nn_probs = best_nn_model.predict_proba(X_test)[:, 1]

    _fpr_lr, _tpr_lr, _ = metrics.roc_curve(y_test, _lr_probs)
    _auc_lr = metrics.auc(_fpr_lr, _tpr_lr)

    _fpr_nn, _tpr_nn, _ = metrics.roc_curve(y_test, _nn_probs)
    _auc_nn = metrics.auc(_fpr_nn, _tpr_nn)

    _axes[0].plot(_fpr_lr, _tpr_lr, color='#1f77b4', lw=2.5, label=f'Logistic Regression (AUC = {_auc_lr:.4f})')
    _axes[0].plot(_fpr_nn, _tpr_nn, color='#2ca02c', lw=2.5, label=f'Neural Network (AUC = {_auc_nn:.4f})')
    _axes[0].plot([0, 1], [0, 1], color='#7f7f7f', linestyle='--', lw=1.5, label='Random Guessing (AUC = 0.50)')
    _axes[0].set_title('(A) Test ROC Curves', fontweight='bold', fontsize=12)
    _axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    _axes[0].set_ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
    _axes[0].legend(loc='lower right', frameon=True, fontsize=10.5)
    _axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot B: Confusion Matrix - Logistic Regression
    _lr_preds = best_lr_model.predict(X_test)
    _cm_lr = metrics.confusion_matrix(y_test, _lr_preds)
    _im1 = _axes[1].imshow(_cm_lr, interpolation='nearest', cmap=plt.cm.Blues)
    _axes[1].set_title(f'(B) Confusion Matrix: Logistic Regression (C={best_lr_c})', fontweight='bold', fontsize=12)
    _tick_marks = [0, 1]
    _axes[1].set_xticks(_tick_marks)
    _axes[1].set_yticks(_tick_marks)
    _axes[1].set_xticklabels(['Negative (0)', 'Positive (1)'])
    _axes[1].set_yticklabels(['Negative (0)', 'Positive (1)'])
    _axes[1].set_xlabel('Predicted Sentiment', fontweight='bold')
    _axes[1].set_ylabel('True Sentiment', fontweight='bold')
    _thresh_lr = _cm_lr.max() / 2.0
    for _i in range(_cm_lr.shape[0]):
        for _j in range(_cm_lr.shape[1]):
            _axes[1].text(_j, _i, format(_cm_lr[_i, _j], 'd'),
                          ha="center", va="center", fontsize=14, fontweight='bold',
                          color="white" if _cm_lr[_i, _j] > _thresh_lr else "black")

    # Plot C: Confusion Matrix - Neural Network
    _nn_preds = best_nn_model.predict(X_test)
    _cm_nn = metrics.confusion_matrix(y_test, _nn_preds)
    _im2 = _axes[2].imshow(_cm_nn, interpolation='nearest', cmap=plt.cm.Greens)
    _axes[2].set_title(f'(C) Confusion Matrix: Neural Network {best_nn_arch}', fontweight='bold', fontsize=12)
    _axes[2].set_xticks(_tick_marks)
    _axes[2].set_yticks(_tick_marks)
    _axes[2].set_xticklabels(['Negative (0)', 'Positive (1)'])
    _axes[2].set_yticklabels(['Negative (0)', 'Positive (1)'])
    _axes[2].set_xlabel('Predicted Sentiment', fontweight='bold')
    _axes[2].set_ylabel('True Sentiment', fontweight='bold')
    _thresh_nn = _cm_nn.max() / 2.0
    for _i in range(_cm_nn.shape[0]):
        for _j in range(_cm_nn.shape[1]):
            _axes[2].text(_j, _i, format(_cm_nn[_i, _j], 'd'),
                          ha="center", va="center", fontsize=14, fontweight='bold',
                          color="white" if _cm_nn[_i, _j] > _thresh_nn else "black")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()

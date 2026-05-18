# summary of previously used clustering algorithms 

## General 

### Difference between static and fluid cluster 
- Described by Zemla et al., 2020 
- Static cluster 
    -  A static cluster switch occurs whenever the next word does not share a category label with any of the previous words since the start of the last cluster
- fluid cluster 
    - A fluid cluster switch occurs when the next word in a list does not share a category label with the previous word. 

### Slope difference algorithm 
- Ref: Bushnell et al., 2023 
- The slope difference algorithm uses word onset times only to identify clusters and switches.
- An exponential curve is fitted to each participant's cumulative word output over time.
- For each consecutive word pair, the actual inter-word interval is compared to the curve's predicted slope at the midpoint between them.
- Words produced faster than predicted (positive slope difference) are classified as linked; slower-than-predicted words indicate a switch.
- Clustering score is reported as the mean cluster length.

### Vector space model 
- Ref: Kim et al., 2019 
- Words are represented as vectors (embeddings) derived from their co-occurrence patterns in large text corpora, such that semantically similar words have vectors close together in the space.
- Cosine similarity is used to quantify the semantic relatedness between adjacent words in a fluency sequence.
- The word2vec architecture is used to generate embeddings, trained on Wikipedia data with varied parameters (embedding dimensions, window size, objective function).
- Three methods are proposed for identifying cluster boundaries (switches):
    - Threshold cutoff — a switch is marked where cosine similarity between adjacent words falls below a fixed threshold (median or 25th percentile of all pairwise similarities).
    - Sharp change — a switch is marked where the drop in cosine similarity between two adjacent words is at least twice the average change seen within the current cluster.
    - Inter-group similarity — a switch is marked where the average pairwise similarity of all words in the current cluster falls below that of a random set of words (based on 1,000 random samples).

### Backlink model 
- Ref: Kim et al., 2019 
- Uses Wikipedia's backlink structure (which pages link to a given page) as a language-agnostic alternative to manually curated resources like WordNet, with coverage across 100+ languages.
- Based on the intuition that a Wikipedia article on a focused topic represents a natural cluster context — e.g., an article on "Burrow" linking to Rabbit, Groundhog, and Meerkat implies those animals form a cluster.
- Switches are identified by comparing the backlink lists of consecutive words in a fluency sequence:
    - If the number of shared backlink documents between two adjacent words falls below a threshold θ, a cluster boundary is marked.
- Te threshold θ is language-dependent, set higher for languages with larger Wikipedias (e.g., θ = 50 for English vs. θ = 1 for Korean), to discount uninformative generic pages (e.g., "List of animals by common name") that link to nearly everything.
- θ is determined empirically by maximising the Spearman correlation between switch counts and age.

## Semantic Fluency 

### Annotated Lists 
- Ref: Zemla et al., 2020 
- What: 
    - they have annotated csv lists in their [github repo](https://github.com/AusterweilLab/snafu-py/tree/master/schemes)
    - lists have two columns: (1) animal name (2) and animal category 
    - they just use these lists to determine when consecutive words don't share category anymore

## Phoenmic Fluency 

### First *n* <sub>letters</sub>
- Ref: Zemla et al., 2020 
- What: 
    - Clusters are formed of consecutive words sharing the same first *n* letters. 
    - They note that some approaches also look into rhymes or homonyms
- Example
    - *N* <sub>letters</sub> = 2 
    - **cat, carpet,** ~~crocodile~~
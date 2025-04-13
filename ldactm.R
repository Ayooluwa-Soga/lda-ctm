# Gist: Creating a hybrid LDA-CTM model using R involves multiple steps, from data preparation to implementing LDA and then transitioning into CTM with the LDA results as priors.

# Load Packages
library(tm)
library(topicmodels)
library(tidyverse)
library(textmineR)
library(slam)
library(wordcloud)
library(RColorBrewer)
library(Matrix) 
library(ldatuning)



# Prepare Data

# Sample corpus - replace with my abstract corpus
# My Corpus Extract
# Load the pre-processed corpus from my RDS file
corpus <- readRDS("csv_corpus_abstracts.rds")


# Create a text corpus
# corpus <- VCorpus(VectorSource(documents))


# Preprocess the text
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)            # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en")) # Remove stop words
corpus <- tm_map(corpus, stripWhitespace)              # Remove extra whitespaces

# Create Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)
dtm <- dtm[rowSums(as.matrix(dtm)) > 0, ] # Remove empty rows


# Run LDA Model

# Number of topics
num_topics <- 20

# Define the number of top words to display per topic
top_n <- 10

# Running the LDA model
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 1234))

# Extract and print topics from LDA
lda_topics <- terms(lda_model, top_n)
print("LDA Topics:")
print(lda_topics)

# Get the document-topic distribution (gamma matrix)
# doc_topic_dist <- posterior(lda_model)$topics
doc_topic_dist <- lda_model@gamma

# Check the topic distribution
## print(doc_topic_dist)


# 5. Load LDA Priors to CTM
# Running the CTM model using LDA priors
ctm_model <- CTM(dtm, k = num_topics, control = list(seed = 1234, estimate.beta = TRUE))

# Extract and print topics from CTM
ctm_topics <- terms(ctm_model, top_n)
print("CTM Topics:")
print(ctm_topics)

# Get the document-topic distributions from CTM
# ctm_doc_topic_dist <- posterior(ctm_model)$topics
ctm_doc_topic_dist <- ctm_model@gamma

# Check the CTM document-topic distributions
# print(ctm_doc_topic_dist)


# Function to create word cloud for each topic
create_wordcloud <- function(topic_terms, topic_number) {
  # Extract the terms and their probabilities for the specific topic
  terms_for_topic <- topic_terms[, topic_number]
  
  # Create the word cloud
  wordcloud(words = terms_for_topic, freq = rep(1, length(terms_for_topic)),
            scale = c(3.5, 0.25), min.freq = 1, max.words = top_n,
            random.order = FALSE, rot.per=0.35, colors = brewer.pal(8, "Dark2"))
}

# Word Clouds for LDA Topics
lda_topic_terms <- lda_model@beta  # Get the term probabilities for LDA
print("LDA Word Clouds:")
for (i in 1:num_topics) {
  print(paste("LDA Topic", i))
  create_wordcloud(lda_topics, i)
}

# Word Clouds for CTM Topics
ctm_topic_terms <- ctm_model@beta  # Get the term probabilities for CTM
print("CTM Word Clouds:")
for (i in 1:num_topics) {
  print(paste("CTM Topic", i))
  create_wordcloud(ctm_topics, i)
}


# 6. Compare LDA and CTM outputs
# Compare document-topic distributions
lda_topic_df <- as.data.frame(doc_topic_dist)
ctm_topic_df <- as.data.frame(ctm_doc_topic_dist)

# Add document IDs
lda_topic_df$doc_id <- 1:nrow(lda_topic_df)
ctm_topic_df$doc_id <- 1:nrow(ctm_topic_df)

# Visualization
library(ggplot2)

# Melt the data for ggplot
lda_long <- lda_topic_df %>% pivot_longer(cols = -doc_id, names_to = "Topic", values_to = "Probability")
ctm_long <- ctm_topic_df %>% pivot_longer(cols = -doc_id, names_to = "Topic", values_to = "Probability")

# Plot LDA
ldaplot <- ggplot(lda_long, aes(x = doc_id, y = Probability, fill = Topic)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("LDA Document-Topic Distributions") +
  theme_minimal()
print(ldaplot)

# Plot CTM
ctmplot <- ggplot(ctm_long, aes(x = doc_id, y = Probability, fill = Topic)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("CTM Document-Topic Distributions") +
  theme_minimal()

print(ctmplot)


# Perplexity is a measure of how well a model predicts a sample, with lower values indicating better performance.
# Calculate perplexity for the LDA model
lda_perplexity <- perplexity(lda_model, dtm)
print(paste("LDA Perplexity:", lda_perplexity))

# Calculate perplexity for the CTM model
ctm_perplexity <- perplexity(ctm_model, dtm)
print(paste("CTM Perplexity:", ctm_perplexity))

# The coherence score evaluates the semantic similarity between the top words in each topic.
coherence_scores <- CalcProbCoherence(topic_terms, dtm)

# # Function to calculate coherence score using textmineR
calculate_coherence <- function(topic_terms, dtm, top_n = 10) {
  # Extract top N terms for each topic
  top_terms <- apply(topic_terms, 2, function(x) order(x, decreasing = TRUE)[1:top_n])

  # Convert to character matrix for textmineR
  top_terms_char <- apply(top_terms, 2, function(x) colnames(topic_terms)[x])

  # Calculate coherence score
  coherence_scores <- CalcProbCoherence(terms = top_terms_char, dtm = dtm)

  return(coherence_scores)
}

# Function to calculate coherence score directly
# calculate_coherence <- function(topic_terms, dtm) {
#   
#   coherence_scores <- CalcProbCoherence(terms = topic_terms, dtm = dtm)
#   
#   return(coherence_scores)
# }

# # Calculate coherence score for LDA
# lda_coherence <- CalcProbCoherence(lda_topics, dtm = dtm)
# lda_coherence <- CalcProbCoherence(phi = exp(lda_model@beta), dtm = dtm, M=10)
# print(paste("LDA Coherence Score:", mean(lda_coherence)))
# 
lda_phi <- exp(lda_model@beta)  # Convert log probabilities to normal probabilities
lda_phi <- t(lda_phi)
rownames(lda_phi) <- Terms(dtm)
dtm_matrix <- as.matrix(dtm)
# Convert to sparse matrix format
dtm_sparse <- Matrix(dtm_matrix, sparse = TRUE)
# Find common terms
common_terms <- intersect(colnames(lda_phi), colnames(dtm_sparse))

# Subset both matrices to include only common terms
lda_phi <- lda_phi[, common_terms]
dtm_sparse <- dtm_sparse[, common_terms]

dtm_sparse <- Matrix(dtm_matrix, sparse = TRUE)

# Check if conversion was successful
class(dtm_sparse)

# ctm_phi <- exp(ctm_model@beta)  # Convert log probabilities to normal probabilities
# rownames(ctm_phi) <- Terms(dtm)

lda_coherence <- CalcProbCoherence(phi = lda_phi, dtm = dtm_sparse)
ctm_coherence <- CalcProbCoherence(phi = ctm_phi, dtm = dtm)

# Print mean coherence scores
print(paste("LDA Mean Coherence:", mean(lda_coherence)))
# print(paste("CTM Mean Coherence:", mean(ctm_coherence)))

# # Calculate coherence score for CTM
# ctm_coherence <- calculate_coherence(ctm_topics, dtm)
# print(paste("CTM Coherence Score:", mean(ctm_coherence)))


library(Matrix)  # Ensure you have this loaded

# Convert beta from log probabilities to normal probabilities
lda_phi <- exp(lda_model@beta)
lda_phi <- t(lda_phi)  # Transpose to match required format

# Get terms from the document-term matrix
rownames(lda_phi) <- Terms(dtm)

# Convert DTM to matrix and then to a sparse format (dgCMatrix)
dtm_matrix <- as.matrix(dtm)
dtm_sparse <- Matrix(dtm_matrix, sparse = TRUE)  # Ensure correct sparse format

# Convert to dgCMatrix format (Compressed Sparse Column format)
dtm_sparse <- as(dtm_sparse, "dgCMatrix")

# Find common terms between lda_phi and dtm_sparse
common_terms <- intersect(colnames(lda_phi), colnames(dtm_sparse))

# Subset both matrices to include only common terms
lda_phi <- lda_phi[, common_terms, drop = FALSE]
dtm_sparse <- dtm_sparse[, common_terms, drop = FALSE]

# Compute coherence score
lda_coherence <- CalcProbCoherence(phi = lda_phi, dtm = dtm_sparse)

# Print mean coherence score for LDA
print(paste("LDA Mean Coherence Score:", mean(lda_coherence)))

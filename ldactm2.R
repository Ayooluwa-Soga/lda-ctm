# Gist: Creating a hybrid LDA-CTM model using R involves multiple steps, from data preparation to implementing LDA and then transitioning into CTM with the LDA results as priors.

# Load Packages
library(tm)
library(topicmodels)
library(tidyverse)
library(textmineR)
library(slam)
library(wordcloud)
library(RColorBrewer)


# Load the pre-processed corpus from my RDS file
corpus <- readRDS("corpus_abstracts.rds")

# Preprocess the text
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)            # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en")) # Remove stop words
corpus <- tm_map(corpus, stripWhitespace)              # Remove extra whitespaces

#   # Extract the content of each document in the corpus
# corpus_content <- lapply(corpus, function(doc) doc$content)
#   # Display the content of the first few documents
# print(corpus_content[1:8])  # Display the first 3 documents

  # Convert to a data frame
corpus_df <- data.frame(doc_id = seq_along(corpus_content), text = unlist(corpus_content), stringsAsFactors = FALSE)
print(corpus_df)


# Create Document-Term Matrix (DTM) - 1
dtm <- DocumentTermMatrix(corpus)
dtm <- dtm[rowSums(as.matrix(dtm)) > 0, ] # Remove empty rows
print(dtm)


# The LDA Model
  # Number of topics
num_topics <- 10
  # Define the number of top words to display per topic
top_n <- 10
  # Running the LDA model
lda_model <- LDA(dtm, k = num_topics,method = "Gibbs", control = list(seed = 1234, alpha = 0.1))
  # Extract and print topics from LDA
lda_topics <- terms(lda_model, top_n)
print("LDA Topics:")
print(lda_topics)
doc_topic_distr <- lda_model@gamma
# print(doc_topic_dist)

# convert topics to text documents
  # Combine terms for each topic into a single string
topic_docs <- apply(lda_topics, 2, paste, collapse = " ")
  # Creating the Corpus
topic_corpus <- Corpus(VectorSource(topic_docs))
print(topic_corpus)
inspect(topic_corpus)

# Create Document-Term Matrix for the topic corpus
topic_dtm <- DocumentTermMatrix(topic_corpus)
print(topic_dtm)

# Load LDA Priors to CTM
  # Running the CTM model using LDA priors
ctm_model <- CTM(dtm, k = num_topics, control = list(estimate.beta = TRUE))

# Extract and print topics from CTM
ctm_topics <- terms(ctm_model, top_n)
print("CTM Topics:")
print(ctm_topics)
topic_correlation <- cor(ctm_model@gamma)
print(topic_correlation)

# library(corrplot)
# corrplot(topic_correlation, method = "shade")



# EVALUATION
  # Perplexity is a measure of how well a model predicts a sample, with lower values indicating better performance.
# Calculate perplexity for the LDA model
lda_perplexity <- perplexity(lda_model, dtm)
print(paste("LDA Perplexity:", lda_perplexity))

# Calculate perplexity for the CTM model
ctm_perplexity <- perplexity(ctm_model, dtm)
print(paste("CTM Perplexity:", ctm_perplexity))



# Convert the DTM to a format suitable for coherence calculation
dtm_matrix <- as.matrix(dtm)  # Ensure DTM is a matrix format

# # Calculate coherence score
# lda_coherence <- CalcProbCoherence(lda_model@beta, dtm = dtm)
# print(lda_coherence)

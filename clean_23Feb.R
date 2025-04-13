# Load Required Packages
library(tm)
library(topicmodels)
library(tidyverse)
library(textmineR)
library(slam)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)

# Load and Preprocess Data
corpus <- readRDS("csv_corpus_abstracts.rds")
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)
dtm <- dtm[rowSums(as.matrix(dtm)) > 0, ]

# Define Parameters
num_topics <- 10
top_n <- 10

# Run LDA Model
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 1234))
lda_topics <- terms(lda_model, top_n)
doc_topic_dist <- lda_model@gamma

print("LDA Topics:")
print(lda_topics)

# Run CTM Model
ctm_model <- CTM(dtm, k = num_topics, control = list(seed = 1234, estimate.beta = TRUE))
ctm_topics <- terms(ctm_model, top_n)
ctm_doc_topic_dist <- ctm_model@gamma

print("CTM Topics:")
print(ctm_topics)

# Function to Create Word Cloud
create_wordcloud <- function(topic_terms, topic_number) {
  terms_for_topic <- topic_terms[, topic_number]
  wordcloud(words = terms_for_topic, freq = rep(1, length(terms_for_topic)),
            scale = c(3.5, 0.25), min.freq = 1, max.words = top_n,
            random.order = FALSE, rot.per = 0.35, colors = brewer.pal(8, "Dark2"))
}

# Generate Word Clouds
for (i in 1:num_topics) {
  print(paste("LDA Topic", i))
  create_wordcloud(lda_topics, i)
  print(paste("CTM Topic", i))
  create_wordcloud(ctm_topics, i)
}

# Compare Document-Topic Distributions
lda_topic_df <- as.data.frame(doc_topic_dist)
ctm_topic_df <- as.data.frame(ctm_doc_topic_dist)
lda_topic_df$doc_id <- 1:nrow(lda_topic_df)
ctm_topic_df$doc_id <- 1:nrow(ctm_topic_df)

lda_long <- lda_topic_df %>% pivot_longer(cols = -doc_id, names_to = "Topic", values_to = "Probability")
ctm_long <- ctm_topic_df %>% pivot_longer(cols = -doc_id, names_to = "Topic", values_to = "Probability")

# Plot Document-Topic Distributions
ldaplot <- ggplot(lda_long, aes(x = doc_id, y = Probability, fill = Topic)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("LDA Document-Topic Distributions") +
  theme_minimal()
print(ldaplot)

ctmplot <- ggplot(ctm_long, aes(x = doc_id, y = Probability, fill = Topic)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("CTM Document-Topic Distributions") +
  theme_minimal()
print(ctmplot)

# Evaluate Models
lda_perplexity <- perplexity(lda_model, dtm)
ctm_perplexity <- perplexity(ctm_model, dtm)
print(paste("LDA Perplexity:", lda_perplexity))
print(paste("CTM Perplexity:", ctm_perplexity))

# Function to Calculate Coherence Score
calculate_coherence <- function(topic_terms, dtm, top_n = 10) {
  top_terms <- apply(topic_terms, 2, function(x) order(x, decreasing = TRUE)[1:top_n])
  top_terms_char <- apply(top_terms, 2, function(x) colnames(topic_terms)[x])
  coherence_scores <- CalcProbCoherence(terms = top_terms_char, dtm = dtm)
  return(coherence_scores)
}

lda_coherence <- CalcProbCoherence(phi = posterior(lda_model)$terms, dtm = dtm, M = 10)
ctm_coherence <- calculate_coherence(ctm_topics, dtm)
print(paste("LDA Coherence Score:", mean(lda_coherence)))
print(paste("CTM Coherence Score:", mean(ctm_coherence)))


# what does this output ----
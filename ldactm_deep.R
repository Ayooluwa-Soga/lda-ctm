library(topicmodels)
library(wordcloud)
library(tm)
library(slam)
library(RColorBrewer)

# Step 1: Prepare the text data
# data("acq", package = "tm")  # Load example data
corpus <- readRDS("csv_corpus_abstracts.rds")

# Preprocess the text
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, sparse = 0.95)  # Remove sparse terms
# Check for empty rows in the DTM
empty_rows <- which(row_sums(dtm) == 0)  # Identify documents with no terms
if (length(empty_rows) > 0) {
  print(paste("Removing", length(empty_rows), "empty documents from the DTM."))
  dtm <- dtm[row_sums(dtm) > 0, ]  # Remove empty documents
}
# Step 2: Train the LDA model
k <- 10  # Number of topics
lda_model <- LDA(dtm, k = k, method = "Gibbs", control = list(seed = 1234))

# Step 3: Train the CTM model
ctm_model <- CTM(dtm, k = k, control = list(seed = 1234))

# Step 4: Aggregate term probabilities across all topics

# For LDA
lda_topic_terms <- posterior(lda_model)$terms  # Terms x Topics matrix
lda_aggregated_terms <- colSums(lda_topic_terms)  # Sum probabilities for each term across topics

# For CTM
ctm_topic_terms <- posterior(ctm_model)$terms  # Terms x Topics matrix
ctm_aggregated_terms <- colSums(ctm_topic_terms)  # Sum probabilities for each term across topics

# Step 5: Create word clouds

# Word Cloud for All LDA Topics
lda_sorted_terms <- sort(lda_aggregated_terms, decreasing = TRUE)  # Sort terms by probability

png(filename = "lda_wordcloud.png", width = 1000, height = 1000)  # Save as PNG
wordcloud(
  words = names(lda_sorted_terms),  # Words
  freq = lda_sorted_terms,          # Aggregated probabilities
  scale = c(3, 0.5),                # Scale for word size
  max.words = 100,                  # Maximum number of words to display
  random.order = FALSE,             # Display words in order of frequency
  colors = brewer.pal(8, "Dark2")   # Color palette
)
title(main = "Word Cloud for All LDA Topics")
dev.off()

# Word Cloud for All CTM Topics
ctm_sorted_terms <- sort(ctm_aggregated_terms, decreasing = TRUE)  # Sort terms by probability

png(filename = "ctm_wordcloud.png", width = 1000, height = 1000)  # Save as PNG
wordcloud(
  words = names(ctm_sorted_terms),  # Words
  freq = ctm_sorted_terms,          # Aggregated probabilities
  scale = c(3, 0.5),                # Scale for word size
  max.words = 100,                  # Maximum number of words to display
  random.order = FALSE,             # Display words in order of frequency
  colors = brewer.pal(8, "Set1")    # Color palette
)
title(main = "Word Cloud for All CTM Topics")
dev.off()

# Step 7: Evaluate the models

# Calculate perplexity for LDA
lda_perplexity <- perplexity(lda_model, newdata = dtm)
print(paste("LDA Perplexity:", lda_perplexity))

# Calculate perplexity for CTM
ctm_perplexity <- perplexity(ctm_model, newdata = dtm)
print(paste("CTM Perplexity:", ctm_perplexity))

# Before creating word cloud
par(mar = c(0,0,0,0))  # Remove plot margins
dev.new(width=5, height=5) 
# Aggregate term probabilities across all topics
term_probabilities <- colSums(lda_model$phi)  # Sum probabilities for each term across topics

# Create frequency data frame
word_df <- data.frame(
  word = names(term_probabilities),
  freq = term_probabilities
) %>% 
  arrange(desc(freq))

# Create a better word cloud
set.seed(123)  # Ensure reproducibility
wordcloud(
  words = word_df$word,
  freq = word_df$freq * 500,  # Adjust scaling for visibility
  scale = c(4, 0.5),        # Larger range for better visibility
  min.freq = 0.01,          # Include more words
  max.words = 100,          # Show more words
  random.order = FALSE,     # Ensure structured layout
  rot.per = 0.3,            # **30% words rotated**
  colors = brewer.pal(8, "Dark2"),  
  fixed.asp = TRUE,        # Allow better layout
  use.r.layout = FALSE      # Use default layout engine
)

# what does this output ----

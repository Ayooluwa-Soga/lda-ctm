library(tm)
# Load the corpus from the RDS file
corpus <- readRDS("csv_corpus_abstracts.rds")
# Inspect the corpus
inspect(corpus[1:3])  # View the first 3 documents
# Extract the content of each document in the corpus
corpus_content <- lapply(corpus, function(doc) doc$content)

# Display the content of the first few documents
print(corpus_content[1:3])  # Display the first 3 documents

# Convert to a data frame
corpus_df <- data.frame(doc_id = seq_along(corpus_content), text = unlist(corpus_content), stringsAsFactors = FALSE)

# View the first few rows
print(corpus_df)

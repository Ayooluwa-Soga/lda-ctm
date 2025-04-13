# Install necessary packages if not already installed
install.packages(c("pdftools", "tm", "tidyverse"))

# Load libraries
library(pdftools)
library(tm)
library(tidyverse)

# Specify the path to your PDF folder
pdf_folder <- "C:/Users/user/Documents/Research/research papers/New"

# Get a list of all PDF files in the folder
pdf_files <- list.files(pdf_folder, pattern = "*.pdf", full.names = TRUE)

# Extract text from each PDF and store it in a list
pdf_texts <- lapply(pdf_files, pdf_text)

# Function to extract abstract from a single PDF
extract_abstract <- function(text) {
  # Collapse text into a single string
  full_text <- paste(text, collapse = " ")
  
  # Use regex to extract text between "Abstract" and the next section (e.g., "Introduction")
  abstract <- str_extract(full_text, "(?<=Abstract)(.*?)(?=(Introduction|1\\.|I\\.|\\n))")
  
  # Return extracted abstract or NA if empty
  return(ifelse(length(abstract) > 0, trimws(abstract), NA))
}

# Apply the function to all extracted texts
abstracts <- sapply(pdf_texts, extract_abstract)

# Convert abstracts to a data frame
abstract_df <- data.frame(doc_id = seq_along(abstracts), text = abstracts, stringsAsFactors = FALSE)

# Remove missing abstracts
abstract_df <- abstract_df %>% drop_na(text)

# Create a text corpus
corpus <- VCorpus(VectorSource(abstract_df$text))

# Preprocess the text corpus
corpus <- tm_map(corpus, content_transformer(tolower))   # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)              # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                  # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))   # Remove stop words
corpus <- tm_map(corpus, stripWhitespace)                # Remove extra whitespace

# Create the Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)

# Check the DTM
inspect(dtm)

# Save DTM and Corpus
saveRDS(dtm, file = "dtm_abstracts.rds")
saveRDS(corpus, file = "corpus_abstracts.rds")

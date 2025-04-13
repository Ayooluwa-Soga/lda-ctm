# Install required libraries if not already installed
install.packages(c("tm", "SnowballC", "topicmodels", "tidyverse"))

# Load libraries
library(tm)
library(SnowballC)
library(topicmodels)
library(tidyverse)

#  Read CSV file
data <- read.csv("C:/Users/user/Desktop/data_r/merged_output.csv", stringsAsFactors = FALSE)

#  Create a text corpus
corpus <- Corpus(VectorSource(data$Content))

#  Clean the text
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%     # Convert to lowercase
  tm_map(removePunctuation) %>%                # Remove punctuation
  tm_map(removeNumbers) %>%                    # Remove numbers
  tm_map(removeWords, stopwords("en")) %>% # Remove common stopwords
  tm_map(stripWhitespace)                  # Remove extra whitespace
  # tm_map(stemDocument)                         # Perform stemming

#  Create a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)


# Save Corpus
saveRDS(corpus, file = "csv_corpus_abstracts.rds")





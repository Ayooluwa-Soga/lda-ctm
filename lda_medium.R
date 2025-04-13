# Load Packages
library(tm)
library(topicmodels)
library(tidyverse)
library(textmineR)
library(slam)
library(wordcloud)
library(RColorBrewer)
library(tidytext)
library(reshape2)


# Load Data
data <- read.csv("C:/Users/user/Desktop/data_r/merged_output.csv")
data <- na.omit(data)
data <- data %>% select(Content, id) %>% head(200)

# Text Cleaning and Tokenization
text_cleaning_tokens <- data %>% 
  tidytext::unnest_tokens(word, Content)

# Remove digits and punctuation
text_cleaning_tokens$word <- gsub('[[:digit:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens$word <- gsub('[[:punct:]]+', '', text_cleaning_tokens$word)

# Filter out single-character words and stop words
text_cleaning_tokens <- text_cleaning_tokens %>%
  filter(!(nchar(word) == 1)) %>%
  anti_join(stop_words)

# Remove empty tokens
tokens <- text_cleaning_tokens %>% 
  filter(!(word == ""))

# Create a document-term matrix
tokens <- tokens %>% 
  group_by(id) %>% 
  mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word)

# Replace NA with empty strings
tokens[is.na(tokens)] <- ""

# Unite the tokens back into a single text column
tokens <- tidyr::unite(tokens, text, -id, sep = " ")

# Trim whitespace
tokens$text <- trimws(tokens$text)

# Creagt dtm
dtm <- CreateDtm(tokens$text,
                 doc_names = tokens$id,
                 ngram_window = c(1,2))

# term freqeuncies
tf <- TermDocFreq(dtm = dtm)
original_tf <- tf %>% select(term, term_freq,doc_freq)
rownames(original_tf) <- 1:nrow(original_tf)

# Eliminate words appearing less than 2 times or in more than half of the
# documents
vocabulary <- tf$term[ tf$term_freq > 1 & tf$doc_freq < nrow(dtm) / 2 ]
dtm = dtm

k_list <- seq(1, 30, by = 1)
model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
if (!dir.exists(model_dir)) dir.create(model_dir)
model_list <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dir, paste0(k, "_topics.rda"))
  
  if (!file.exists(filename)) {
    m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
    m$k <- k
    m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 5)
    save(m, file = filename)
  } else {
    load(filename)
  }
  
  m
}, export=c("dtm", "model_dir")) # export only needed for Windows machines
#model tuning
#choosing the best model
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)
ggplot(coherence_mat, aes(x = k, y = coherence)) +
  geom_point() +
  geom_line(group = 1)+
  ggtitle("Best Topic by Coherence Score") + theme_minimal() +
  scale_x_continuous(breaks = seq(1,20,1)) + ylab("Coherence")

model <- model_list[which.max(coherence_mat$coherence)][[ 1 ]]
model$top_terms <- GetTopTerms(phi = model$phi, M = 20)
top20_wide <- as.data.frame(model$top_terms)

model$topic_linguistic_dist <- CalcHellingerDist(model$phi)
model$hclust <- hclust(as.dist(model$topic_linguistic_dist), "ward.D")
model$hclust$labels <- paste(model$hclust$labels, model$labels[ , 1])
plot(model$hclust)







# Visualising topics of words based on the max value of phi
set.seed(1234)

final_summary_words <- data.frame(top_terms = t(model$top_terms))
final_summary_words$topic <- rownames(final_summary_words)
rownames(final_summary_words) <- 1:nrow(final_summary_words)

# Use melt to reshape the data frame
final_summary_words <- final_summary_words %>%
  melt(id.vars = "topic", value.name = "word") %>%
  select(-variable)

# Group by topic and word, then arrange by the frequency (value)
final_summary_words <- final_summary_words %>%
  group_by(topic, word) %>%
  summarise(frequency = n(), .groups = 'drop') %>%
  arrange(desc(frequency))
  ungroup()

# Separate the topic column into two parts and clean up
final_summary_words <- final_summary_words %>%
  tidyr::separate(topic, into = c("t", "topic"), sep = "\\.") %>%
  select(-t)

# Join with original term frequencies
word_topic_freq <- left_join(final_summary_words, original_tf, by = c("word" = "term"))

# Generate word clouds for each topic
pdf("cluster.pdf")
for(i in 1:length(unique(final_summary_words$topic)))
{  
  wordcloud(words = subset(final_summary_words ,topic == i)$word, 
            freq = subset(final_summary_words ,topic == i)$value, 
            min.freq = 1,
            max.words=200, random.order=FALSE, rot.per=0.35, 
            colors=brewer.pal(8, "Dark2"))
}
dev.off()


# outputs - 
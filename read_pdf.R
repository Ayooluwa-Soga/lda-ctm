require(pdftools)
require(tm)
# install.packages('SnowballC')

#PDF folder
pdf_folder <- "C:/Users/user/Documents/Research/research papers/New"

#PDF files in the folder
pdf_files <- list.files(pdf_folder, pattern = "pdf$", full.names = TRUE)

# Extract text and store it in a list
pdf_texts <- lapply(pdf_files, pdf_text)

length(pdf_texts)

lapply(pdf_texts,length)

#create pdfdb
pdf_db <- VCorpus(URISource(pdf_files), readerControl = list(reader = readPDF))

text.tdm <- TermDocumentMatrix(pdf_db, control=list(removePunctuation = TRUE,
                                                    stopwords = TRUE,
                                                    tolower = TRUE,
                                                    stemming = FALSE,
                                                    removeNumbers = TRUE,
                                                    bounds = list(global = c(3, Inf))))


# Create Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(pdf_db)

# Save DTM
saveRDS(pdf_db, file = "pdf_db.rds")
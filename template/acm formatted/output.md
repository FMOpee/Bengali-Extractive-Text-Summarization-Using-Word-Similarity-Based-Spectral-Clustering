---
abstract: |
  Extractive Text Summarization is the process of picking the best parts
  of a larger text without losing any key information. This is really
  necessary in this day and age to get concise information faster due to
  digital information overflow. Previous attempts at extractive text
  summarization, specially in Bengali, either relied on TF-IDF or used
  naive similarity measures both of these suffers expressing semantic
  relationship correctly. The objective of this paper is to develop an
  extractive text summarization method for Bengali language, that uses
  the latest NLP techniques and extended to other low resource
  languages. We developed a word Similarity-based Spectral Clustering
  (WSbSC) method for Bengali extractive text summarization. It extracts
  key sentences by grouping semantically similar sentences into clusters
  with a novel sentence similarity calculating algorithm. We took the
  geometric means of individual Gaussian similarity values using word
  embedding vectors to get the similarity between two sentences. Then,
  used TF-IDF ranking to pick the best sentence from each cluster. This
  method is tested on four datasets, and it outperformed other recent
  models by 43.2% on average ROUGE scores (ranging from 2.5% to 95.4%).
  The method is also experimented on Turkish, Marathi and Hindi language
  and found that the performance on those languages often exceeded the
  performance of Bengali. In addition, a new high quality dataset is
  provided for text summarization evaluation. We, believe this research
  is a crucial addition to Bengali Natural Language Processing, that can
  easily be extended into other languages.
author:
- Fahim Morshed
- Md. Abdur Rahman
- Sumon Ahmed
bibliography:
- main.bib
title: Extractive Text Summarization Using Word Similarity-based
  Spectral Clustering
---

::: CCSXML
\<ccs2012\> \<concept\>
\<concept_id\>10010147.10010178.10010179.10003352\</concept_id\>
\<concept_desc\>Computing methodologies Information
extraction\</concept_desc\>
\<concept_significance\>500\</concept_significance\> \</concept\>
\</ccs2012\>
:::

# Introduction {#sec:introduction}

Text Summarization is the process of shortening a larger text without
losing any key information to increase the readability and save time for
the reader. But manually summarizing very large texts is a
counter-productive task due to it being more time consuming and tedious.
So, developing an Automatic Text Summarization (ATS) method that can
summarize larger texts reliably is really necessary to alleviate this
manual labour [@Widyassari-2022-rev-ats-tech-met]. Using ATS to
summarize textual data is thus becoming very important in various fields
such as news articles, legal documents, health reports, research papers,
social media contents etc. ATS helps the reader to quickly and
efficiently get the essential information without needing to read
through large amounts of texts
[@wafaa-2021-summary-comprehensive-review]. So, ATS is being utilized in
various fields, from automatic news summarization, content filtering,
and recommendation systems to assisting legal professionals in going
through long documents. And researchers in reviewing academic papers by
condensing vast amount of informations. It can also play a critical role
in personal assistants and chatbots, providing condensed information to
users quickly and efficiently [@tas-2017-rev-text-sum-2].

There are two main types of ATS: extractive and abstractive
[@tas-2017-rev-text-sum-2]. Extractive summarization, which is the focus
of this paper, works by selecting a subset from the source document,
maintaining the original wording and sentence structure
[@moratanch-2017-extractive-review]. In contrast, abstractive
summarization involves synthesising new text that reflects information
from the input document but does not copy from it, similar to how a
human summarizes a text [@Moratanch-2016-abstractive-rev]. Both of the
method has their own advantage. The abstractive summarization can
simulate the human language pattern very well thus increasing the
natural flow and readability of the summary. But the extractive method
requires much less computation than the abstractive method while also
containing more key informations from the input
[@gupta-2010-extractive-rev].

The key approach to extractive summarization is implementing a sentence
selection method to classify which sentences will belong in the summary.
For this purpose, various ranking based methods were used to rank the
sentences and identify the best sentences as the summary. These ranking
methods used indexing [@Baxendale_1958_firstsummarization], statistical
[@edmundson_1969_earlysum] or Term Frequency-Inverse Document Frequency
(TF-IDF) [@das-2022-tfidf; @sarkar-2012-tfidf-2; @sarkar-2012-tfidf]
based techniques to score the sentences and select the best scoring
ones. But these methods fail to capture the semantic relationships
between sentences of the input due to being simplistic in nature. To
capture the semantic relationships between sentences, graph based
extractive methods are effective due to the using of sentence similarity
graph in their workflow [@wafaa-2021-summary-comprehensive-review].
Graph based methods represent the sentences as nodes of a graph, and the
semantic similarity between two sentences as the edge between the nodes
[@moratanch-2017-extractive-review]. Popular graph based algorithms like
LexRank [@Erkan-lexRank-2004] and TextRank [@mihalcea-2004-textrank]
build graphs based on cosine similarity of the bag-of-word vectors.
LexRank uses PageRank [@page-PageRank-1999] method to score the
sentences from the graph while TextRank uses random walk to determine
which sentences are the most important to be in the summary. Graph-based
methods like TextRank and LexRank offer a robust way to capture sentence
importance and relationship, ensuring that the extracted summary covers
the key information while minimizing
redundancy [@wafaa-2021-summary-comprehensive-review].

Clustering-based approaches are a subset of graph-based approach to
extractive text summarization. Here, sentences are grouped into clusters
based on their semantic similarity to divide the document into topics,
and one representative sentence from each cluster is chosen to form the
summary [@Mohan-2022-topic-modeling-rev-clustering]. Clustering reduces
redundancy by ensuring that similar sentences are grouped together and
only the most representative sentence is selected. This method is
effective in summarization of documents with multiple topics or
subtopics by picking sentences from each topic. An example of this
method can be seen with COSUM [@alguliyev-2019-cosum] where the
summarization is achieved using k-means clustering on the sentences and
picking the most salient sentence from each cluster to compile in the
final summary.

Despite the advancements of ATS in other languages, it remains an
under-researched topic for Bengali due to Bengali being a low-resource
language. Early attempts at Bengali text summarization relied on
traditional methods like TF-IDF scoring to select the best scoring
sentences to form the summary
[@Akter-2017-tfidf-3; @das-2022-tfidf; @sarkar-2012-tfidf; @sarkar-2012-tfidf-2].
These TF-IDF based approaches, while simple, faced challenges in
capturing the true meaning of sentences. This is because TF-IDF based
methods treats words as isolated terms resulting in synonyms of words
being regarded as different terms [@tas-2017-rev-text-sum-2]. To solve
this problem, graph-based methods were introduced in Bengali to improve
summarization quality by incorporating sentence similarity but they were
still limited by the quality of word embeddings used for the Bengali
language. With the advent of word embedding models like FastText
[@grave-etal-2018-fasttext], it became possible to represent words in a
vector space model, thus enabling more accurate sentence similarity
calculations. However, existing models that use word embeddings, such as
Sentence Average Similarity-based Spectral Clustering (SASbSC) method
[@roychowdhury-etal-2022-spectral-base], encountered issues with
sentence-similarity calculation when averaging word vectors to represent
the meaning of a sentence with a vector. This method failed in most
similarity calculation cases because words in a sentence are
complementary to each other rather than being similar, leading to
inaccurate sentence representations after averaging these different word
vectors. As a result, word-to-word relationships between sentences get
lost, reducing the effectiveness of the method.

In this paper, we propose a new clustering-based text summarization
approach to address the challenge of calculating sentence similarity
accurately. Our method improves upon previous attempts at graph-based
summarization methods
[@chowdhury-etal-2021-tfidf-clustering; @roychowdhury-etal-2022-spectral-base]
by focusing on the individual similarity between word pairs in sentences
rather than averaging word vectors. We showed that the use of this novel
approach greatly improved the accuracy, coverage and reliability of the
output summaries due to having a deeper understanding of the semantic
similarity between sentences. To calculate sentence similarity, we used
the geometric mean of individual word similarities. The individual word
similarities were achieved using Gaussian kernel function on a pair of
corresponding word vector from each sentence. The word pairs are
selected by finding the word vector with the smallest Euclidean distance
from the target sentence. Thus, we get the semantic similarity between
two sentences which can be used to build an affinity matrix to
graphically represent the relationship between the sentences. This graph
is clustered into groups to divide the document into distinct topics.
One sentence from every cluster is selected to reduce redundancy and
increase topic coverage. This method consistently outperforms other
graph based text summarization methods such as BenSumm
[@chowdhury-etal-2021-tfidf-clustering], LexRank [@Erkan-lexRank-2004],
SASbSC [@roychowdhury-etal-2022-spectral-base] using four datasets on
ROUGE metrics [@lin-2004-rouge] as shown in Figure
[5](#fig:radarchart){reference-type="ref" reference="fig:radarchart"}
and Table [2](#tab:result_comparison-1){reference-type="ref"
reference="tab:result_comparison-1"}. This method performs well in other
low resource languages also such as Hindi, Marathi and Turkish due to
the language independent nature, as shown in the Table
[3](#tab:other_language){reference-type="ref"
reference="tab:other_language"}.

The main contributions of this paper are: (I) Proposed a new way to
calculate similarity between two sentences. (II) Contributes a novel
methodology for extractive text summarization for the Bengali language;
by improving sentence similarity calculations and enhancing clustering
techniques. (III) It offers a generalizable solution for creating less
redundant and information rich summaries across languages. (IV) It
provides a publicly available high quality dataset of 500 human
generated summaries.

The rest of the paper is organized as follows: The Related works and
Methodology are described in section
[2](#sec:literature-review){reference-type="ref"
reference="sec:literature-review"} and
[3](#sec:methodology){reference-type="ref" reference="sec:methodology"}
respectively. Section [4](#sec:result){reference-type="ref"
reference="sec:result"} illustrates the result of the performance
evaluation for this work. Section
[5](#sec:discussion){reference-type="ref" reference="sec:discussion"}
discusses the findings of the paper in more depth, and section
[6](#sec:conclusion){reference-type="ref" reference="sec:conclusion"}
concludes the paper.

# Related Work {#sec:literature-review}

Text summarization has been an important necessity for textual data
consumption for a long time because of its ability to compress a given
input text into a shortened version without losing any key information.
For this reason, automating the text summarization process has been a
research problem for NLP. Thus researchers attempted automatic text
summarization for a long time too. At first, indexing-based text
summarization methods were attempted such as the work by
@Baxendale_1958_firstsummarization [@Baxendale_1958_firstsummarization].
This method scored sentences based on the presence of indexing terms in
the sentence and picked the sentences with the best score. But this type
of method failed to capture the topic and essence of the input text as
we wouldn't have the topic keywords of an unforeseen input document. To
solve this issue, text summarization with statistical methods like
TF-IDF became very popular due to its ability to capture the important
topic words of a document in an unsupervised manner.
@edmundson_1969_earlysum [@edmundson_1969_earlysum] proposed a method
which can focus on the central topic of a document by using the TF-IDF
measure. It uses two metrics, Term Frequency (how many times a term
appears in the input) and Inverse Document Frequency (inverse of how
many documents the term appears in a large text corpus) to calculate the
importance of a term in a document. Using TF-IDF helps to identify the
words that are common in the input text but not as common in the
language in general and thus classifying them as the central topic of
the document. But this method is also error-prone due to its
consideration of every word as a unique isolated term without any
semantic relation with other words. This leads to the method often
missing some central topic if it gets divided into too many synonyms.

The problems faced by topic-based summarization methods were alleviated
by graph-based extractive text summarization methods which brought new
modern breakthroughs. Graph-based methods like LexRank
[@Erkan-lexRank-2004] and TextRank [@mihalcea-2004-textrank] were able
to capture the relationship between sentences more accurately due to use
of sentence similarity graph in their process. LexRank
[@Erkan-lexRank-2004] calculates the similarity graph by using cosine
similarity of bag-of-words vectors between two sentences from the input.
The most important sentences from the graph are classified using the
PageRank [@page-PageRank-1999] algorithm on the graph. PageRank ranks
those sentences higher who are more similar with other high ranked
sentences. Another graph-based method, TextRank
[@mihalcea-2004-textrank] also uses a similar approach while building
the similarity graph. In the graph for every sentence, TextRank
distributed the score of that sentence to its neighbours using a random
walk. This process is repeated over and over until the scores converge.
Then the method picks the sentences with the best scores as the summary.
Although graph-based methods such as LexRank and TextRank models are
Ground-breaking compared to their time, they still lacked fundamental
understanding of the words involved in a sentence due to not using any
vector representation of the semantic relationship between the words
involved.

To solve the problem of representing semantic relationship, a
mathematical abstraction called Word Vector Embedding was conceptualized
by the seminal work of @salton-1975-word-vector
[@salton-1975-word-vector]. Word Vector Embedding uses a word vector
space as a mathematical abstraction of a lexicon where the closer two
words are semantically, the closer they are in the vector space. Using
word vector for graph based text summarization has only been started to
be attempted recently [@Jain-2017-word-vector-embedding-summary] due to
the lack of fully developed word embedding datasets.

Although text summarization have been a forefront of NLP research, text
summarization research in Bengali is a more recent development than in
other high resource languages. So, a lot of sophisticated approaches
from other languages haven't been attempted at Bengali yet. Earlier
bengali extractive methods have been focused on some derivative of
TF-IDF based text summarization such as the methods developed by
@chowdhury-etal-2021-tfidf-clustering
[@chowdhury-etal-2021-tfidf-clustering], @das-2022-tfidf
[@das-2022-tfidf], @sarkar-2012-tfidf [@sarkar-2012-tfidf] etc.
@sarkar-2012-tfidf [@sarkar-2012-tfidf] used a simple TF-IDF score of
each sentence to rank them and pick the best sentences to generate the
output summary. @das-2022-tfidf [@das-2022-tfidf] used weighted TF-IDF
along with some other sentence features like sentence position to rank
the sentences. @chowdhury-etal-2021-tfidf-clustering
[@chowdhury-etal-2021-tfidf-clustering] however, used TF-IDF matrix of a
document to build a graph and perform hierarchical clustering to group
sentences together and pick one sentence from each group. One
shortcoming of this method is that TF-IDF matrix is not semantically
equivalent to the actual sentences due to the fundamental issues with
TF-IDF. So, the TF-IDF doesn't perfectly represent the semantic
relationship between the sentences in the generated graph. Using word
vector embedding for Bengali has solved this problem of semantic
representation. Word embedding datasets such as FastText
[@grave-etal-2018-fasttext] dataset[^1] with word vector embeddings in
157 languages, including Bengali. Using this dataset, SASbSC
[@roychowdhury-etal-2022-spectral-base] proposed a model where they
replaced all the words from the input with their respective vector, then
averaged the word vectors in a sentence to get a vector representation
for the sentence. The sentence average vectors are then used to get the
similarity between two sentences using the Gaussian similarity function
to build an affinity matrix. This affinity matrix is used to cluster the
sentences using spectral clustering to group sentences into distinct
topics. The summary is generated after picking one sentence from each
cluster to reduce redundancy and increase coverage.

But the sentence average method suffers critically due to its inability
in capturing accurate relationship between sentences. This happens due
to words in a sentence generally not having similar meaning with each
other, instead they express different parts of one whole meaning of a
sentence. This makes the words more complementary instead of being
similar leading to word vectors being scattered throughout the word
vector space. This characteristics makes the sentence average vectors
always tending towards the centre and not representing the semantic
similarity accurately. An example of this happening is shown in Figure
[1](#fig:sarkar-problem){reference-type="ref"
reference="fig:sarkar-problem"} where the distance between the sentence
average vectors are being misleading. In the figure, scenario (a) shows
two sentence with word vectors very closely corresponding with each
other. On the other hand, scenario (b) shows two sentences without any
significant word correspondence. But the scenario (a) has a larger
distance between sentence average vectors than scenario (b) despite
having more word correspondence. This larger distance makes the Gaussian
similarity between the sentences lower due to the inverse exponential
nature of the function. The lower similarity leads to the graphical
representation being less accurate and thus failing to capture the true
semantic relationship within the sentences. This shortcoming of the
method has been one of the key motivations for this research.

![A scenario where sentence averaging method fails. (a) shows a scenario
where the distance between sentence average vectors are larger than (b)
despite the word vectors from (a) being more closely related than
(b).](figs/sarkar_problem-edited){#fig:sarkar-problem width="80%"}

# Methodology {#sec:methodology}

The summarization process followed here can be boiled down as two basic
steps, grouping all the close sentences together based on their meaning
to minimize redundancy and picking one sentence from each group to
maximize sentence coverage. To group semantically similar sentences into
clusters, we build a sentence similarity graph and perform spectral
clustering on it [@roychowdhury-etal-2022-spectral-base]. The sentence
similarity graph is produced using a novel sentence similarity
calculation algorithm that uses geometric mean of Gaussian similarity
between individual word pairs from the two sentences. The Gaussian
similarity is calculated using the vector embedding representations of
the words. On the other hand, we used TF-IDF scores to pick the highest
ranked sentences from a cluster
[@Akter-2017-tfidf-3; @das-2022-tfidf; @sarkar-2012-tfidf; @sarkar-2012-tfidf-2].
The summarization process followed here involves 4 steps. These are,
Pre-processing, Sentence similarity calculation, Clustering and Summary
generation. These steps are illustrated in Figure
[2](#fig:process-flow-diagram){reference-type="ref"
reference="fig:process-flow-diagram"} and further discussed in the
following subsections.

![Process Flow
Diagram](figs/process-flow-diagram-new){#fig:process-flow-diagram
width="80%"}

## Preprocessing {#subsec:preprocessing}

Preprocessing is the standard process of NLP that transforms raw human
language inputs into a format that can be used by a computer algorithm.
In this step, the document is transformed into a few set of vectors
where each word is represented with a vector, each sentence is
represented as a set of vectors and the whole document as a list
containing those sets. To achieve this representation, the preprocessing
follows three steps. These are tokenization, stop word removal, and word
embedding. A very common step in preprocessing, word stemming, isn't
used here as the word embedding dataset works best for the whole word
instead of the root word. These steps are further discussed below.

Tokenization is the step of dividing an input document into sentences
and words to transform it into a more usable format. Here, the input
document is represented as a list of sentence and the sentences are
represented as a list of words. Stop words, such as prepositions and
conjunctions, add sentence fluidity but don't carry significant meaning.
Removing these words allows the algorithm to focus on more impactful
words. Word Embedding is the process of representing words as vector in
a vector space. In this vector space, semantically similar words are
placed closer together so that the similarity relation between words can
be expressed in an abstract mathematical way. Each word from the
tokenized and filtered list is replaced with their corresponding
vectors. If some words aren't present in the dataset, they are
considered too rare and thus ignored. Following these steps, the input
document is transformed into a set of vectors to be used in sentence
similarity calculation.

## Sentence Similarity Calculation {#subsec:sentence-similarity-calculation}

Sentence similarity is the key criteria to build a graphical
representation of the semantic relationship in the input document. This
graphical representation is expressed via an affinity matrix to cluster
the semantically similar sentences. The nodes in the affinity matrix
represents the sentences of the input and the edges of the matrix
represents the similarity between two sentence. Here, we proposed a
novel sentence similarity calculation technique using individual
Gaussian similarity of word-pairs to construct an affinity matrix. To
calculate the sentence similarity between two sentences, we adhere to
the following steps.

Firstly, for every word of a sentence, we find its closest counterpart
from the other sentence to build a word pair. The Euclidean distance
between the vector representation of the word-pairs is defined as the
Most Similar Word Distance ($D_{msw}$). The process of calculating the
$D_{msw}$ is shown in the Equation
[\[eq:msd\]](#eq:msd){reference-type="ref" reference="eq:msd"}. In this
equation, for every word vector $x$, in a sentence $X$, we find the
Euclidean distance ( $d(x,y_i)$ ) between the word vectors $x$ and $y_i$
where $y_i$ is a word vector from the sentence $Y$. The lowest possible
distance in this set of Euclidean distance is the $D_{msw}$.
$$\label{eq:msd}
    D_{msw}(x,Y) = min(\{d(x,y_i) : y_i \in Y \})$$ Then, we calculate
the $D_{msw}$ for every word of the two sentences $X$ and $Y$ to make
the sentence similarity calculation symmetric. This process is shown in
the Equation [\[eq:mswdset\]](#eq:mswdset){reference-type="ref"
reference="eq:mswdset"} where $D$ is a set containing all the $D_{msw}$
from both $X$ and $Y$ that would be used in the later steps.
$$D = \{D_{msw}(x,Y) : x \in X\} \cup \{D_{msw}(y,X) : y \in Y\}
    \label{eq:mswdset}$$ After this, the word similarity between the
word pairs is calculated to get the degree of correspondence between the
two sentences. Here, the word similarity is calculated using Gaussian
kernel function for the elements of the set $D$; Gaussian kernel
functions provides a smooth, flexible and most representative similarity
between two vectors [@babud-1986-gaussian]. The process of calculating
word similarity ($W_{sim}$) is given in the Equation
[\[eq:wsim\]](#eq:wsim){reference-type="ref" reference="eq:wsim"}. In
this equation, for every element $D_i$ in set $D$, we calculate the
Gaussian similarity to obtain word similarity. In the formula for
Gaussian similarity, $\sigma$ denotes the standard deviation that can be
used as a control variable. The standard deviation represents the
blurring effect of the kernel function. A lower value for $\sigma$
represents a high noise sensitivity of the function
[@babud-1986-gaussian]. The value of sigma was fine-tuned to be
$5\times10^{-11}$ which gives the best similarity measurement. The
process of fine-tuning is described in the experimentation section
(section [4.4.1](#subsubsec:sigma){reference-type="ref"
reference="subsubsec:sigma"}). $$\label{eq:wsim}
    W_{sim} = \{ exp\left(\frac{-D_i^2}{2\sigma^2}\right) : D_i \in D\}$$
Finally, the sentence similarity between the two sentences $Sim(X,Y)$ is
calculated using geometric mean of the word similarities to construct an
affinity matrix. The geometric mean makes the similarity value less
prone to effects of outliers thus it makes the calculation more
reliable. This process is shown in the Equation
[\[eq:sent_sim\]](#eq:sent_sim){reference-type="ref"
reference="eq:sent_sim"}; the geometric mean of each $w_{sim}$ value for
the two sentences is simplified in the Equation
[\[eq:sent_sim\]](#eq:sent_sim){reference-type="ref"
reference="eq:sent_sim"} to make the calculation process more
computation friendly. $$\label{eq:sent_sim}
    \begin{split}
        Sim(X,Y)
        &=  \left(
            \prod_{i=1}^nW_{Sim_i}
        \right)^{1/n}\\
        &=  \left(
            exp\left(\frac{-D_1^2}{2\sigma^2}\right)\cdot
            exp\left(\frac{-D_2^2}{2\sigma^2}\right)\cdot
                \ldots \cdot
            exp\left(\frac{-D_n^2}{2\sigma^2}\right)
        \right)^{1/n}\\
        &=  exp\left(
            -\frac{D_1^2+D_2^2+\ldots+D_n^2}{2n\sigma^2}
            \right)\\
        &=  exp\left(
            -\frac{\sum_{i=1}^nD_i^2}{2n\sigma^2}
        \right)
    \end{split}$$ By following steps described above, we get a
similarity value for two sentences. This value solves the lack of local
word correspondence problem faced by the word averaging based similarity
calculation method [@roychowdhury-etal-2022-spectral-base]. Figure
[3](#fig:msd){reference-type="ref" reference="fig:msd"} demonstrates the
merit of the method claimed above. Figure
[3](#fig:msd){reference-type="ref" reference="fig:msd"} shows that the
proposed method can solve the local word correspondence problem faced by
word averaging method. In the figure, the scenario
[3](#fig:msd){reference-type="ref" reference="fig:msd"}(a) has a set of
smaller $D_{msw}$ than the scenario [3](#fig:msd){reference-type="ref"
reference="fig:msd"}(b). Having smaller $D_{msw}$ makes the individual
word similarities $W_{sim}$ larger due to the nature of Gaussian kernel
function. These values would result in a higher sentence similarity for
the sentences in the scenario [3](#fig:msd){reference-type="ref"
reference="fig:msd"}(a) than in the scenario
[3](#fig:msd){reference-type="ref" reference="fig:msd"}(b). This solved
the problem showed in the Figure
[1](#fig:sarkar-problem){reference-type="ref"
reference="fig:sarkar-problem"} where the scenario
[1](#fig:sarkar-problem){reference-type="ref"
reference="fig:sarkar-problem"}(a) has a larger sentence average
distance than [1](#fig:sarkar-problem){reference-type="ref"
reference="fig:sarkar-problem"}(b) resulting in
[1](#fig:sarkar-problem){reference-type="ref"
reference="fig:sarkar-problem"}(a) having a smaller sentence similarity
than [1](#fig:sarkar-problem){reference-type="ref"
reference="fig:sarkar-problem"}(b).

![Emphasis of local word correspondence in $D_{msw}$ method. Here,
scenario (a) has a larger similarity value due to having a set of
smaller $D_{msw}$ than scenario (b)](figs/msd-edited){#fig:msd
width="80%"}

The whole process of sentence similarity calculation is shown in the
Algorithm [\[alg:similarity\]](#alg:similarity){reference-type="ref"
reference="alg:similarity"}. In this algorithm, we calculate an affinity
matrix using the input word vector list. We took the most similar word
distance $D_{msw}$ in line 8--13 and 18--23 for each word (line 7 and
17) of a sentence pair (line 3 and 6). Sum of $D^2$ from Equation
[\[eq:sent_sim\]](#eq:sent_sim){reference-type="ref"
reference="eq:sent_sim"} is calculated in the lines 14 and 24 to be used
in the calculation of sentence similarity (line 27). The similarity is
used to construct an affinity matrix $A$ (line 28).

::: algorithm
:::

## Clustering {#subsec:clustering}

Clustering is a key corner stone of the proposed method where we aim to
cluster semantically similar sentences together to divide the input
document into multiple topics. Clustering the document minimizes
redundancy in the output summary by ignoring multiple sentences from the
same topic. For clustering, spectral and DBSCAN methods were considered
due to their capability of being able to cluster irregular shapes.
However, spectral clustering was found to perform better than DBSCAN
because, smaller input documents have lower density which hinders DBSCAN
[@roychowdhury-etal-2022-spectral-base].

On the contrary, spectral clustering takes the affinity matrix of a
graph as input and returns the grouping of graph nodes by transforming
the graph into its eigenspace [@vonLuxburg-2007-spectral-tutorial]. The
following Equation [\[eq:affinity\]](#eq:affinity){reference-type="ref"
reference="eq:affinity"} shows the process of building an affinity
matrix. Here, for every sentence pair $S_i$ and $S_j$, we calculate
their sentence similarity and place it in both $A_{ij}$ and $A_{ji}$ of
the affinity matrix $A$. $$\label{eq:affinity}
    A_{ij}=A_{ji}=Sim(S_i,S_j)$$ The affinity matrix is clustered into a
reasonable, $k=\lceil N/5 \rceil$ groups to achieve an output summary.
This summary, is short while the sentence groups resulting from
clustering is also not too broad.

## Summary Generation {#subsec:summary-generation}

Output summary is generated by selecting one sentence from each cluster
achieved in the previous step to minimize topic redundancy and maximize
topic coverage. To select one sentence from a cluster, we perform TF-IDF
ranking on the sentences inside a cluster and pick the sentence with the
highest TF-IDF score. To get the TF-IDF score of a sentence, we take the
sum of all TF-IDF values for the words in that sentence. The TF-IDF
value for a word is achieved by multiplying how many time the word
appeared in the input document (Term Frequency, TF) and the inverse of
how many document does the word appear in a corpus (Inverse Document
Frequency, IDF). The process of scoring sentences are shown in the
Equation [\[eq:tfidf\]](#eq:tfidf){reference-type="ref"
reference="eq:tfidf"} where, for each word $W_i$ in a sentence $S$ and a
corpus $C$, we calculate the TF-IDF score of a sentence.
$$\label{eq:tfidf}
    \text{TFIDF}(S) = \sum_{i=1}^{\text{length}(S)}\text{TF}(W_i) \times \text{IDF}(W_i,C)$$
The sentences with the best TF-IDF score from each clusters are then
compiled as the output summary in their order of appearance in the input
document to preserve the original flow of information. The process of
generating output summary is further expanded in the Algorithm
[\[alg:summary\]](#alg:summary){reference-type="ref"
reference="alg:summary"}. After the clustering step (line 2), we took
the TF-IDF score (line 7) of each sentence in a cluster (line 6). For
each cluster (line 4), we pick the best scoring sentence (line 9). These
sentences are then ordered (line 11) and concatenated (line 13--15) to
generate the output summary.

::: algorithm
$k \gets \lceil$ length($A$) / 5 $\rceil$ clusters $\gets$
`spectral_clustering(adjacency = `$A$`, `$k$`)` indexes $\gets \{\}$
`sort(indexes)` $S \gets `` "$ $S$
:::

# Result {#sec:result}

The performance of the proposed method has been compared against three
Bengali text summarization methods to evaluate the correctness of
generated summaries. The three methods, which have been used as a
benchmark, are BenSumm [@chowdhury-etal-2021-tfidf-clustering], LexRank
[@Erkan-lexRank-2004] and SASbSC
[@roychowdhury-etal-2022-spectral-base]. All of these methods have been
evaluated using four datasets to test the robustness of the model for
Bengali text summarization for input from different sources. For
evaluation, the Recall-Oriented Understudy for Gisting Evaluation
(ROUGE) [@lin-2004-rouge] metric has been used. Details about the
models, datasets and evaluation metrics are provided in the following
sections.

## Text Summarization Models {#subsec:text-summarization-models}

We implemented Bensumm [@chowdhury-etal-2021-tfidf-clustering] and
SASbSC [@roychowdhury-etal-2022-spectral-base], two recent Bengali
extractive models, and LexRank [@Erkan-lexRank-2004], a popular
benchmarking model for extractive text summarization to evaluate the
effectiveness of the proposed WSbSC method. These methods are further
discussed in the following section.

**WSbSC** is the proposed model for this research. We find the Gaussian
similarity for word-pairs from two sentences and take their geometric
mean to get the similarity between two sentences. We use the similarity
value to perfom spectral clustering to group sentences into groups and
extract a representative sentence using TF-IDF score. The extracted
sentences are used to generate the output summary which minimizes
redundancy and maximizes coverage.

**SASbSC** [@roychowdhury-etal-2022-spectral-base] is the first method
that introduced the approach of clustering sentences using sentence
similarity. However, it uses the average of word vectors in a sentence
for calculating similarity. After clustering the sentences based on
their similarity, cosine similarity between the average vectors are used
to pick the best sentence from a cluster.

**BenSumm** [@chowdhury-etal-2021-tfidf-clustering] is another recent
research that describes an extractive and an abstractive text
summarization technique. We compared the extractive technique with our
model to ensure a fair and balanced comparison. Here, similarity matrix
is built using TF-IDF which groups the sentences using agglomerative
clustering. A Github implementation[^2] provided by the authors is used
in the comparison process.

**LexRank** [@Erkan-lexRank-2004] uses a TF-IDF based matrix and Googles
PageRank algorithm [@page-PageRank-1999] to rank sentences. Then the top
ranked sentences are selected and arranged into summary. An implemented
version of this method is available as lexrank[^3] which is used in the
comparison process using a large Bengali wikipedia corpus[^4].

## Evaluation Datasets {#subsec:evaluation-datasets}

We used four evaluation dataset with varying quality, size and source to
examine the robustness of the methods being tested. The first dataset is
a **self-curated** extractive dataset that we developed to evaluate the
performance of our proposed method. An expert linguistic team of ten
members summarized 250 news articles of varying sizes to diversify the
dataset. Each article is summarized twice by two different experts to
minimize human bias in the summary. In total, 500 different
document-summary pairs are present in this dataset. The dataset is
publicly available on Github[^5] for reproducibility.

The second dataset is collected from **Kaggle** dataset[^6] which is a
collection of summary-article pair from "The Daily Prothom Alo\"
newspaper. The dataset is vast in size however the quality of the
summaries are poor. All the articles smaller than 50 characters were
discarded from the dataset. The articles with unrelated summaries were
also removed from the dataset to improve the quality. After filtering,
total 10,204 articles remained, each with two summaries in the dataset.

The third dataset we used for evaluation is **BNLPC** which is a
collection of news article summaries [@Hque-2015-BNLPC-Dataset]. This
was collected from GitHub[^7] for experimentation that contains one
hundred articles with three different summaries for each article.

The fourth dataset is collected from **Github**[^8]. The dataset
contains 200 documents each with two human generated summaries. These
documents were collected from several different Bengali news portals.
The summaries were generated by linguistic experts which ensures the
high quality of the dataset.

## Evaluation Metrics {#subsec:evaluation-metrics}

To evaluate the correctness of generated summaries against human written
summaries, ROUGE metric [@lin-2004-rouge] is used. The method compares a
reference summary with a machine generated summary to evaluate alignment
between the two. It uses N-gram-based overlapping to calculate a
precision, recall and F-1 score of each summary. The Rouge package[^9]
is used to evaluate the proposed models against human generated
summaries. The package has three different metrics for comparison of
summaries. These are are:

1.  **ROUGE-1** uses unigram matching to find how similar two summaries
    are. It calculates total common characters between the summaries to
    evaluate the performance. But using this metric also can be
    misleading as very large texts will share a very high proportion of
    uni-grams between them.

2.  **ROUGE-2** uses bi-gram matching to find how much similar the two
    summaries are in a word level. Having more common bigrams between
    two summaries indicates a deeper syntactic similarity between them.
    Using this in combination with the ROUGE-1 is a standard practice to
    evaluate machine generated summaries
    [@wafaa-2021-summary-comprehensive-review].

3.  **ROUGE-LCS** finds the longest common sub-sequence between two
    summaries to calculate the rouge scores. It focuses on finding
    similarity in the flow of information in the sentence level between
    two summaries.

In this study, we compared the F-1 scores from each of these metrics for
the four models.

## Experimentation {#subsec:experimentation}

The proposed model contains three main steps, similarity calculation,
clustering and sentence extraction. Each of these steps have different
techniques or values that can be experimented on to find the most suited
strategy. For sentence similarity calculation, different values for
standard deviation ($\sigma$) can be fixed to get the most
representative semantic similarity value. Spectral clustering was found
to work best for the clustering step through experimentations done by
other researchers [@roychowdhury-etal-2022-spectral-base]. To extract
the best sentence from a cluster, lead extraction and TF-IDF ranking
strategies was considered. Thus, experimentations on finding the most
suited standard deviation value and sentence extraction was conducted.
These experimentations are described in the following sections.

### Fine-tuning Standard Deviation ($\sigma$):  {#subsubsec:sigma}

Standard deviation ($\sigma$) plays a crucial role for sentence
similarity calculation (Equation
[\[eq:sent_sim\]](#eq:sent_sim){reference-type="ref"
reference="eq:sent_sim"}). Therefore, to fix the most suited $\sigma$
value sixty-three different values were experimented on. These values
ranged from $10^{-12}$ to $10$ on regular intervals. After
experimentation, $5\times10^{-11}$ was fixed as the value for $\sigma$
that gives the most representative semantic relation between sentences.
The result for fine-tuning process is shown in Figure
[4](#fig:sigma-fine-tuning){reference-type="ref"
reference="fig:sigma-fine-tuning"}.

![Fine-tuning for different standard deviation ($\sigma$)
values](figs/fine-tuning-edited){#fig:sigma-fine-tuning width="80%"}

### Different Sentence Extraction Techniques From Clusters:  {#subsubsec:different-ranking-techniques-inside-clusters}

We implemented two sentence extraction methods to pick the most
representative sentence from each cluster. Firstly, the lead extraction
method is used to select the sentence that appears first in the input
document. Because, generally the earlier sentences in an input contain
more information on the context of the input document. Secondly,
extracting sentences based on their TF-IDF score was also experimented
on. In Table [1](#tab:ranking){reference-type="ref"
reference="tab:ranking"}, the TF-IDF ranking is shown to performs better
than the lead extraction method.

::: {#tab:ranking}
  Method             Rouge-1    Rouge-2    Rouge-LCS
  ----------------- ---------- ---------- -----------
  Lead extraction      0.47       0.36       0.43
  TF-IDF ranking     **0.50**   **0.40**   **0.46**

  : Comparison of Result of different ranking techniques
:::

## Comparison {#subsec:comparison}

The performance of the proposed method (WSbSC) is compared with BenSumm
[@chowdhury-etal-2021-tfidf-clustering], SASbSC
[@roychowdhury-etal-2022-spectral-base] and LexRank
[@Erkan-lexRank-2004] on four datasets (Self-Curated (SC), Kaggle,
BNLPC, Github) using the average F-1 score for three ROUGE metrics
(Rouge-1, Rouge-2, Rouge-LCS). The comparative results of this
evaluation are shown in Table
[2](#tab:result_comparison-1){reference-type="ref"
reference="tab:result_comparison-1"} where our proposed model performs
11.9%, 24.1% and 16.2% better than SASbSC in Rouge-1, Rouge-2 and
Rouge-LCS respectively on the self-curated dataset. It performs 68.9%,
95.4% and 84.6% better in Rouge-1, Rouge-2 and Rouge-LCS respectively
than BenSumm on the Kaggle dataset. It also performs 3% and 2.6% better
in Rouge-2 and Rouge-LCS respectively and ties in Rouge-1 with SASbSC
using the BNLPC dataset. It performs 58%, 86.4%, and 67.9% better in
Rouge-1, Rouge-2 and Rouge-LCS respectively than BenSumm on the Github
dataset.

::: {#tab:result_comparison-1}
  Dataset        Model                                              Rouge-1    Rouge-2    Rouge-LCS
  -------------- ------------------------------------------------- ---------- ---------- -----------
  Self-curated   WSbSC (Proposed)                                   **0.47**   **0.36**   **0.43**
                 BenSumm [@chowdhury-etal-2021-tfidf-clustering]      0.41       0.29       0.36
                 SASbSC [@roychowdhury-etal-2022-spectral-base]       0.42       0.29       0.37
                 LexRank [@Erkan-lexRank-2004]                        0.22       0.14       0.20
  Kaggle         WSbSC (Proposed)                                   **0.49**   **0.43**   **0.48**
                 BenSumm [@chowdhury-etal-2021-tfidf-clustering]      0.29       0.22       0.26
                 SASbSC [@roychowdhury-etal-2022-spectral-base]       0.23       0.12       0.18
                 LexRank [@Erkan-lexRank-2004]                        0.24       0.16       0.22
  BNLPC          WSbSC (Proposed)                                   **0.41**   **0.34**   **0.40**
                 BenSumm [@chowdhury-etal-2021-tfidf-clustering]      0.36       0.28       0.34
                 SASbSC [@roychowdhury-etal-2022-spectral-base]     **0.41**     0.33       0.39
                 LexRank [@Erkan-lexRank-2004]                        0.26       0.19       0.24
  Github         WSbSC (Proposed)                                   **0.49**   **0.41**   **0.47**
                 BenSumm [@chowdhury-etal-2021-tfidf-clustering]      0.31       0.22       0.28
                 SASbSC [@roychowdhury-etal-2022-spectral-base]       0.30       0.18       0.24
                 LexRank [@Erkan-lexRank-2004]                        0.22       0.14       0.20

  : Comparison of average Rouge scores between graph based extractive
  summarization models on 4 datasets
:::

These results are further visualized into three radar charts in Figure
[5](#fig:radarchart){reference-type="ref" reference="fig:radarchart"} to
compare the performance of the models on different Rouge metrics. As
stated in the charts, the proposed method performed consistently and
uniformly across all the datasets regardless of the quality of the
dataset. But other models, such as BenSumm performs well in three
datasets (SC, GitHub, BNLPC) but fails in the Kaggle dataset. Similarly,
SASbSC performs well in SC and BNLPC datasets but its performance
decreases sharply in Kaggle and GitHub datasets. LexRank although
performs consistently across all datasets but is far lower on average.
According to the result analysis in Table
[2](#tab:result_comparison-1){reference-type="ref"
reference="tab:result_comparison-1"} and Figure
[5](#fig:radarchart){reference-type="ref" reference="fig:radarchart"},
WSbSC is the most accurate and reliable Bengali extractive text
summarization model.

![The Radar chart of the models of being compared on four datasets at
once](figs/radar-chart-edited){#fig:radarchart width="80%"}

## Implementation Into Other Languages {#subsec:implementation-into-other-languages}

The proposed model is language-independent thus, it can be extended into
other languages too. For this, only a language-specific tokenizer, a
stop-word list and a word embedding dataset is required. We implemented
this model on three non-bengali datasets to show the language
independent nature of the model. To evaluate the quality of the sentence
extraction, we tried to find evaluation datasets for summarization on
other low resource languages. But could only find relevant datasets in
three other languages i.e. Hindi, Marathi and Turkish. We adopted the
proposed model into these three low resource languages to check how well
it performs.

::: {#tab:other_language}
  Language                  Rouge-1   Rouge-2   Rouge-LCS
  ------------------------ --------- --------- -----------
  Bengali (Self-curated)     0.47      0.36       0.43
  Bengali (Kaggle)           0.49      0.43       0.48
  Bengali (BNLPC)            0.41      0.34       0.40
  Bengali (Github)           0.49      0.41       0.47
  Bengali (Average)          0.47      0.38       0.44
  Hindi                      0.40      0.26       0.36
  Marathi                    0.50      0.42       0.50
  Turkish                    0.48      0.39       0.47

  : Comparison of Result of proposed summarization method in other
  low-resource languages
:::

The Table [3](#tab:other_language){reference-type="ref"
reference="tab:other_language"} shows the result of the proposed WSbSC
method for extractive summarization in other low resource languages. In
this table, we can see that the results on Marathi and Turkish are
slightly better than the result on Bengali. Although it performs
slightly lower on Hindi, the score is still similar to Bengali. To
evaluate the models performance on Hindi, we used a Kaggle dataset[^10]
produced by Gaurav Arora. For the Marathi, we used another Kaggle
dataset[^11] produced by Ketki Nirantar. For the Turkish language, we
used a GitHub dataset[^12] produced by the XTINGE
[@Demir-2024-xtinge_turkish_extractive] team for evaluation.

# Discussion {#sec:discussion}

The results presented at Table
[2](#tab:result_comparison-1){reference-type="ref"
reference="tab:result_comparison-1"},
[3](#tab:other_language){reference-type="ref"
reference="tab:other_language"} and Figure
[5](#fig:radarchart){reference-type="ref" reference="fig:radarchart"}
highlights the effectiveness of proposed WSbSC model for extractive text
summarization in Bengali, as well as its adaptability to other
low-resource languages. This section analyses the comparative results,
the strengths and limitations of the proposed method, and potential
areas for further research.

As evidenced by the results shown in
Table [2](#tab:result_comparison-1){reference-type="ref"
reference="tab:result_comparison-1"} and
Figure [5](#fig:radarchart){reference-type="ref"
reference="fig:radarchart"}, the WSbSC model consistently outperforms
other graph-based extractive text summarization models, namely BenSumm
[@das-2022-tfidf], LexRank [@Erkan-lexRank-2004], and SASbSC
[@roychowdhury-etal-2022-spectral-base], across four datasets. The
proposed model shows better performance compared to other three methods
on Rouge-1, Rouge-2, Rouge-LCS metrics. This performance improvement can
largely be attributed to the novel approach of calculating sentence
similarity. While calculating sentence similarity, taking the geometric
mean of individual similarity between word pairs overcomes the lack of
local word correspondence faced by the averaging vector method
[@roychowdhury-etal-2022-spectral-base]. The Gaussian kernel-based word
similarity provides a precise semantic relationships between sentences
which further contribute in the performance improvement. Another reason
for performance improvement is the usage of spectral clustering which is
very effective in capturing irregular cluster shapes.

WSbSC includes a novel strategy for calculating sentence similarity
despite the existence of other popular methods for comparing two sets of
vectors. Our proposed strategy is more suited for similarity calculation
in the context of language than other strategies such as Earth Movers
Distance (EMD) [@Rubner-19998-emd], Hausdorff Distance
[@hausdorff-1914-hausdorff-distance], Procrustes Analysis
[@Gower-1975-procrustes-distance]. EMD [@Rubner-19998-emd] and
Procrustes Analysis [@Gower-1975-procrustes-distance] are two very
computationally expensive method who also involve scaling or rotating
vectors; irrelevant for word vectors due to not holding any semantic
meaning. Another method, Hausdorff distance
[@hausdorff-1914-hausdorff-distance] calculates the highest possible
distance between vectors from two set. Although similarly expensive as
WSbSC, it is easily influenced by outlier words due to only considering
the worst case scenario.

On the other hand, the proposed method focuses on local vector
similarity between two sets which is more important for words. The
Gaussian similarity function captures the proximity of points smoothly,
providing a continuous value for similarity between two words in a
normalized way. Gaussian similarity is also robust against small
outliers due to being a soft similarity measure. Taking geometric mean
also helps smooth over similarity values for outlier words.

One of the key strengths of this proposed method is the reduction of
redundancy, a common challenge in extractive summarization methods, by
grouping semantically similar sentences together. The use of spectral
clustering for the grouping task improves the performance by not
assuming a specific cluster shape. Another key strength of WSbSC is the
improved sentence similarity calculation technique over word averaging
method [@roychowdhury-etal-2022-spectral-base], which dilutes the
semantic meaning of a sentence. Another key strength is the scalability
of the method across languages by requiring very few language-specific
resources. This scalability is demonstrated in the experiments with
Hindi, Marathi, and Turkish languages (Table
[3](#tab:other_language){reference-type="ref"
reference="tab:other_language"}).

Despite its advantages, the WSbSC model does face some challenges. The
model heavily relies on pre-trained word embeddings, which may not
always capture the full details of certain domains or newly coined
terms. The FastText [@grave-etal-2018-fasttext] dataset used here is
heavily reliant on wikipedia for training which could introduce some
small unforeseen biases. Where the word embeddings do not have some
words of a given document, the model's performance could degrade as it
leaves those words out of the calculation process. The model also does
not take into account the order in which words appear in a sentence or
when they form special noun or verb groups. So it can be a little naive
in some highly specialized fields.

The proposed WSbSC model represents a significant advancement in Bengali
extractive text summarization due to its ability to accurately capture
semantic similarity, reduce redundancy, maximize coverage and generalize
across languages. While there are still challenges to be addressed, the
results of this study demonstrate the robustness and adaptability of the
WSbSC model, offering a promising direction for future research in
multilingual extractive summarization.

# Conclusion {#sec:conclusion}

In this study, we proposed WSbSC method for Bengali extractive text
summarization which is also extended to other low resource languages.
The method uses geometric mean of Gaussian similarities between
individual word pairs to identify deeper semantic relationship within
two sentences. This sentence similarity calculation method leads to
improved performance than recent graph based extractive text
summarization methods. The performance is also improved by using of
spectral clustering which improve coherence and relevance of the
generated summaries by minimizing redundancy and maximizing topic
coverage. High performance across three ROUGE metrics on four datasets
prove the versatility and robustness of the proposed method. WSbSC can
also be extended into other languages as shown through the results on
Hindi, Marathi and Turkish languages. It addresses the need for an
effective summarization technique in Bengali language.

The results showed the strengths of the proposed WSbSC method compared
to several baseline techniques. Despite the promising results, WSbSC may
face limitations on highly specialized or domain-specific texts, where
deeper linguistic features beyond word similarity could be considered.
The lack of consideration for word order in a sentence is also a key
limitation which could be explored in the future. Future works could
also explore hybrid models that integrate modern post-processing
techniques to improve the flow of the output.

In conclusion, this work contributes to the growing body of
computational linguistics research focused on low-resource languages
like Bengali. The WSbSC method offers a novel approach for extractive
summarization by using a new algorithm for calculating similarity
between two sentences and sets the stage for further advancements in
both Bengali text processing and multilingual summarization techniques.

[^1]: *https://fasttext.cc/docs/en/crawl-vectors.html*

[^2]: *https://github.com/tafseer-nayeem/BengaliSummarization*

[^3]: *https://pypi.org/project/lexrank/*

[^4]: *https://www.kaggle.com/datasets/shazol/bangla-wikipedia-corpus*

[^5]: *dataset link*

[^6]: *https://www.kaggle.com/datasets/towhidahmedfoysal/bangla-summarization-datasetprothom-alo*

[^7]: *https://github.com/tafseer-nayeem/BengaliSummarization/tree/main/Dataset/BNLPC/Dataset2*

[^8]: *https://github.com/Abid-Mahadi/Bangla-Text-summarization-Dataset*

[^9]: *https://pypi.org/project/rouge/*

[^10]: *https://www.kaggle.com/datasets/disisbig/hindi-text-short-and-large-summarization-corpus/*

[^11]: *https://www.kaggle.com/datasets/ketki19/marathi*

[^12]: *https://github.com/xtinge/turkish-extractive-summarization-dataset/blob/main/dataset/XTINGE-SUM_TR_EXT/xtinge-sum_tr_ext.json*

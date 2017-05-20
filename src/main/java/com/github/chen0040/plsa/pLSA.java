package com.github.chen0040.plsa;

import com.github.chen0040.data.text.*;
import com.github.chen0040.data.utils.TupleTwo;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;


/**
 * Created by xschen on 9/16/15.
 * probabilistic Latent Semantic Analysis
 */
@Getter
@Setter
public class pLSA {

    private static final Logger logger = LoggerFactory.getLogger(pLSA.class);

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private SparseMatrix probability_topic_given_doc_and_word = null;
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private SparseMatrix probability_topic = null;
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private SparseMatrix probability_doc_given_topic = null;
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private SparseMatrix probability_word_given_topic = null;

    private int topicCount = 20;

    @Setter(AccessLevel.NONE)
    private int docCount = -1;
    @Setter(AccessLevel.NONE)
    private int wordCount = -1;

    private int maxIters = 10;
    @Setter(AccessLevel.NONE)
    private double loglikelihood = Double.NEGATIVE_INFINITY;
    private int maxVocabularySize = 100;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private Random random = new Random();
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private Vocabulary vocabulary;


    private boolean removeNumbers = true;
    private boolean removeIpAddress = true;
    private boolean stemmerEnabled = false;

    private List<Document> documents;

    public pLSA(){

    }

    public String wordAtIndex(int word){
        return vocabulary.get(word);
    }

    public pLSA makeCopy() {
        pLSA clone = new pLSA();
        clone.copy(this);

        return clone;
    }

    public void copy(pLSA that){
        this.probability_topic_given_doc_and_word = that.probability_topic_given_doc_and_word.makeCopy();
        this.probability_topic = that.probability_topic.makeCopy();
        this.probability_doc_given_topic = probability_doc_given_topic.makeCopy();
        this.probability_word_given_topic = that.probability_word_given_topic.makeCopy();
        this.topicCount = that.topicCount;
        this.docCount = that.docCount;
        this.wordCount = that.wordCount;
        this.vocabulary = that.vocabulary.makeCopy();
        this.maxIters = that.maxIters;
        this.loglikelihood = that.loglikelihood;
        this.stemmerEnabled = that.stemmerEnabled;
        this.removeIpAddress = that.removeIpAddress;
        this.removeNumbers = that.removeNumbers;
    }


    private List<Document> buildDocuments(List<String> docs){
        final StopWordRemoval stopWordRemoval = new StopWordRemoval();
        final LowerCase lowerCase = new LowerCase();
        final PorterStemmer stemmer = new PorterStemmer();

        stopWordRemoval.setRemoveIPAddress(removeIpAddress);
        stopWordRemoval.setRemoveNumbers(removeNumbers);

        List<Map<String, Integer>> wordCountMap = docs.parallelStream().map(text -> {

            List<String> words = BasicTokenizer.doTokenize(text);

            words = lowerCase.filter(words);
            words = stopWordRemoval.filter(words);

            if(stemmerEnabled) {
                words = stemmer.filter(words);
            }

            Map<String, Integer> wordCounts = new HashMap<>();
            for(String word : words){
                wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
            }

            return wordCounts;

        }).collect(Collectors.toList());

        vocabulary = new BasicVocabulary();

        int m = wordCountMap.size();
        Map<String, Integer> uniqueWords = new HashMap<>();
        for(int i=0; i < m; ++i){
            Map<String, Integer> doc = wordCountMap.get(i);
            for(Map.Entry<String, Integer> entry : doc.entrySet()){
                uniqueWords.put(entry.getKey(), uniqueWords.getOrDefault(entry.getKey(), 0) + entry.getValue());
            }
        }

        List<TupleTwo<String,Integer>> words = uniqueWords.entrySet().stream().map(entry -> new TupleTwo<>(entry.getKey(), entry.getValue())).collect(Collectors.toList());

        words.sort((a, b) -> -Integer.compare(a._2(), b._2()));

        List<String> candidates = new ArrayList<>();
        Map<String, Integer> positions = new HashMap<>();

        for(int i=0; candidates.size() < maxVocabularySize && i < words.size(); ++i){
            String word = words.get(i)._1();
            candidates.add(word);
            positions.put(word, candidates.size()-1);
        }

        vocabulary.setWords(candidates);

        List<Document> result = new ArrayList<>();
        for(int i=0; i < m; ++i){
            Map<String, Integer> doc = wordCountMap.get(i);
            Map<Integer, Integer> wordCount = new HashMap<>();
            for(Map.Entry<String, Integer> entry : doc.entrySet()){
                String word = entry.getKey();
                if(positions.containsKey(word)){
                    wordCount.put(positions.get(word), entry.getValue());
                }
            }
            result.add(new BasicDocument(wordCount, docs.get(i), i));
        }

        return result;

    }

    public List<TupleTwo<Integer, Double>> getTopRankingTopics4Doc(int doc, int limits){
        final double[] probs = new double[topicCount];
        List<Integer> topic_orders = new ArrayList<Integer>();
        for(int topic = 0; topic < topicCount; ++topic){
            probs[topic] = probability_topic.get(topic) * probability_doc_given_topic.get(topic, doc);
            topic_orders.add(topic);
        }

        topic_orders.sort((t1, t2) -> Double.compare(probs[t2], probs[t1]));

        List<TupleTwo<Integer, Double>> topRankedTopics = new ArrayList<>();
        limits = Math.min(limits, topicCount);
        for(int i = 0; i < limits; ++i){
            int topic = topic_orders.get(i);
            topRankedTopics.add(new TupleTwo<>(topic, probs[topic]));
        }
        return topRankedTopics;
    }

    public List<TupleTwo<Document, Double>> getTopRankingDocs4Topic(int topic, int limits){
        final double[] probs = new double[docCount];
        List<Document> doc_orders = new ArrayList<>();
        for(int docIndex = 0; docIndex < docCount; ++docIndex){
            probs[docIndex] = probability_doc_given_topic.get(topic, docIndex);
            doc_orders.add(documents.get(docIndex));
        }

        doc_orders.sort((a, b) -> -Double.compare(probs[a.docIndex()], probs[b.docIndex()]));

        List<TupleTwo<Document, Double>> topRankedDocs = new ArrayList<>();
        limits = Math.min(limits, docCount);
        for(int i = 0; i < limits; ++i){
            Document doc = doc_orders.get(i);
            topRankedDocs.add(new TupleTwo<>(doc, probs[doc.docIndex()]));
        }
        return topRankedDocs;
    }

    public List<TupleTwo<String, Double>> getTopRankingWords4Topic(int topic, int limits){
        final double[] probs = new double[wordCount];
        List<String> word_orders = new ArrayList<>();
        for(int wordIndex = 0; wordIndex < wordCount; ++wordIndex){
            probs[wordIndex] = probability_word_given_topic.get(topic, wordIndex);
            word_orders.add(wordAtIndex(wordIndex));
        }

        word_orders.sort((t1, t2) -> Double.compare(probs[vocabulary.indexOf(t2)], probs[vocabulary.indexOf(t1)]));

        List<TupleTwo<String, Double>> topRankedWords = new ArrayList<>();
        limits = Math.min(limits, wordCount);
        for(int i = 0; i < limits; ++i){
            String word = word_orders.get(i);
            topRankedWords.add(new TupleTwo<>(word, probs[vocabulary.indexOf(word)]));
        }
        return topRankedWords;

    }

    public void fit(List<String> docs){
        documents = buildDocuments(docs);

        docCount = documents.size();
        wordCount = vocabulary.getLength();

        probability_topic = new SparseMatrix(topicCount);
        probability_doc_given_topic = new SparseMatrix(topicCount, docCount);
        probability_word_given_topic = new SparseMatrix(topicCount, wordCount);
        probability_topic_given_doc_and_word = new SparseMatrix(docCount, wordCount, topicCount);

        for(int topic = 0; topic < topicCount; ++topic) {
            probability_topic.set(topic, 1.0 / topicCount);

            for(int doc = 0; doc < docCount; ++doc){
                probability_doc_given_topic.set(topic, doc, random.nextDouble());
            }
            probability_doc_given_topic.normalize(topic);

            for(int word = 0; word < wordCount; ++word){
                probability_word_given_topic.set(topic, word, random.nextDouble());
            }
            probability_word_given_topic.normalize(topic);
        }


        for(int iter = 0; iter < maxIters; ++iter){

            // E-step
            for(int doc = 0; doc < docCount; ++doc){

                List<Integer> words = documents.get(doc).wordIndices();
                for(Integer word : words) {
                    for(int topic = 0; topic < topicCount; ++topic) {
                        double probability_of_topic_and_doc_and_word = probability_topic.get(topic)
                                * probability_doc_given_topic.get(topic,doc)
                                * probability_word_given_topic.get(topic, word);
                        probability_topic_given_doc_and_word.set(doc, word, topic, probability_of_topic_and_doc_and_word);
                    }

                    probability_topic_given_doc_and_word.normalize(doc, word);
                }
            }


            // M-step
            for(int topic = 0; topic < topicCount; ++topic){
                for(int word = 0; word < wordCount; ++word) {

                    // update P (word | topic) /prop sum_{doc} (P(topic | word, doc) * count(word in doc))
                    double sum = 0;
                    for (int doc = 0; doc < docCount; ++doc) {
                        Document basicDocument = documents.get(doc);
                        Map<Integer, Integer> wordCounts = basicDocument.indexedWordCount();

                        sum += probability_topic_given_doc_and_word.get(doc, word, topic) * wordCounts.getOrDefault(word, 0);
                    }
                    probability_word_given_topic.set(topic, word, sum);
                }
                probability_word_given_topic.normalize(topic);

                for(int doc = 0; doc < docCount; ++doc){
                    // update P (doc | topic) /prop sum_{word} (P(topic | word, doc) * count(word in doc))
                    double sum = 0;
                    for(Map.Entry<Integer, Integer> entry : documents.get(doc).indexedWordCount().entrySet()){
                        int word = entry.getKey();
                        sum += probability_topic_given_doc_and_word.get(doc, word, topic) * entry.getValue();
                    }

                    probability_doc_given_topic.set(topic, doc, sum);
                }
                probability_doc_given_topic.normalize(topic);

                double sum = 0;
                for(int doc = 0; doc < docCount; ++doc){
                    Document basicDocument = documents.get(doc);
                    Map<Integer, Integer> wordCounts = basicDocument.indexedWordCount();

                    for(Map.Entry<Integer, Integer> entry : wordCounts.entrySet()){
                        int word = entry.getKey();
                        sum += probability_topic_given_doc_and_word.get(doc, word, topic) * entry.getValue();
                    }
                }
                probability_topic.set(topic, sum);

            }

            // Normalize
            probability_topic.normalize();

            loglikelihood = calcLogLikelihood(documents);

            logger.info("#: {} log-likelihood: {}", iter, loglikelihood);
        }
    }

   private double calcLogLikelihood(List<Document> batch){
       int m = batch.size();

       double L = 0.0;

       for(int doc = 0; doc < m; ++doc){
           Document basicDocument = batch.get(doc);
           Map<Integer, Integer> wordCounts = basicDocument.indexedWordCount();

           for(Map.Entry<Integer, Integer> entry : wordCounts.entrySet()) {
               int word = entry.getKey();

               double sum = 0;

               for(int topic = 0; topic < topicCount; ++topic) {
                   double value = probability_topic.get(topic)
                           * probability_doc_given_topic.get(topic,doc)
                           * probability_word_given_topic.get(topic, word);
                   sum += value;
               }


               double logSum = Math.log(sum);
               if(Double.isNaN(logSum)) continue;
               L += entry.getValue() * logSum;
           }
       }

       return L;

   }






}

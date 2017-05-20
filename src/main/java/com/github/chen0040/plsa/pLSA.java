package com.github.chen0040.plsa;

import com.github.chen0040.data.text.BasicVocabulary;
import com.github.chen0040.data.text.LowerCase;
import com.github.chen0040.data.text.PorterStemmer;
import com.github.chen0040.data.text.Vocabulary;
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

    private double[][] clone(double[][] rhs){
        if(rhs==null) return null;
        int m = rhs.length;
        double[][] clone = new double[m][];
        for(int i=0; i < m; ++i){
            clone[i] = clone(rhs[i]);
        }
        return clone;
    }

    private double[] clone(double[] rhs){
        if(rhs == null) return null;
        int m = rhs.length;
        double[] clone = new double[m];
        for(int i=0; i < m; ++i){
            clone[i] = rhs[i];
        }
        return clone;
    }

    private List<Document> buildVocab(List<Map<String, Integer>> documents){
        vocabulary = new BasicVocabulary();

        int m = documents.size();
        Map<String, Integer> uniqueWords = new HashMap<>();
        for(int i=0; i < m; ++i){
            Map<String, Integer> doc = documents.get(i);
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
            Map<String, Integer> doc = documents.get(i);
            Map<Integer, Integer> wordCount = new HashMap<>();
            for(Map.Entry<String, Integer> entry : doc.entrySet()){
                String word = entry.getKey();
                if(positions.containsKey(word)){
                    wordCount.put(positions.get(word), entry.getValue());
                }
            }
            result.add(new BasicDocument(wordCount));
        }

        return result;

    }

    public List<Map.Entry<Integer, Double>> getTopRankingTopics4Doc(int doc, int limits){
        final double[] probs = new double[topicCount];
        List<Integer> topic_orders = new ArrayList<Integer>();
        for(int topic = 0; topic < topicCount; ++topic){
            probs[topic] = probability_topic.get(topic) * probability_doc_given_topic.get(topic, doc);
            topic_orders.add(topic);
        }

        Collections.sort(topic_orders, (t1, t2) -> Double.compare(probs[t2], probs[t1]));

        List<Map.Entry<Integer, Double>> topRankedTopics = new ArrayList<Map.Entry<Integer, Double>>();
        limits = Math.min(limits, topicCount);
        for(int i = 0; i < limits; ++i){
            int topic = topic_orders.get(i);
            topRankedTopics.add(new AbstractMap.SimpleEntry<>(topic, probs[topic]));
        }
        return topRankedTopics;
    }

    public List<Map.Entry<Integer, Double>> getTopRankingDocs4Topic(int topic, int limits){
        final double[] probs = new double[docCount];
        List<Integer> doc_orders = new ArrayList<>();
        for(int doc = 0; doc < docCount; ++doc){
            probs[doc] = probability_doc_given_topic.get(topic, doc);
            doc_orders.add(doc);
        }

        doc_orders.sort((a, b) -> -Double.compare(probs[a], probs[b]));

        List<Map.Entry<Integer, Double>> topRankedDocs = new ArrayList<>();
        limits = Math.min(limits, docCount);
        for(int i = 0; i < limits; ++i){
            int doc = doc_orders.get(i);
            topRankedDocs.add(new AbstractMap.SimpleEntry<>(doc, probs[doc]));
        }
        return topRankedDocs;
    }

    public List<Map.Entry<Integer, Double>> getTopRankingWords4Topic(int topic, int limits){
        final double[] probs = new double[wordCount];
        List<Integer> word_orders = new ArrayList<Integer>();
        for(int word = 0; word < wordCount; ++word){
            probs[word] = probability_word_given_topic.get(topic, word);
            word_orders.add(word);
        }

        Collections.sort(word_orders, (t1, t2) -> Double.compare(probs[t2], probs[t1]));

        List<Map.Entry<Integer, Double>> topRankedWords = new ArrayList<>();
        limits = Math.min(limits, wordCount);
        for(int i = 0; i < limits; ++i){
            int word = word_orders.get(i);
            topRankedWords.add(new AbstractMap.SimpleEntry<>(word, probs[word]));
        }
        return topRankedWords;

    }

    public void fit(List<String> docs){

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

        List<Document> documents = buildVocab(wordCountMap);

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

                List<Integer> words = documents.get(doc).words();
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
                        Map<Integer, Integer> wordCounts = basicDocument.getWordCounts();

                        sum += probability_topic_given_doc_and_word.get(doc, word, topic) * wordCounts.getOrDefault(word, 0);
                    }
                    probability_word_given_topic.set(topic, word, sum);
                }
                probability_word_given_topic.normalize(topic);

                for(int doc = 0; doc < docCount; ++doc){
                    // update P (doc | topic) /prop sum_{word} (P(topic | word, doc) * count(word in doc))
                    double sum = 0;
                    for(Map.Entry<Integer, Integer> entry : documents.get(doc).getWordCounts().entrySet()){
                        int word = entry.getKey();
                        sum += probability_topic_given_doc_and_word.get(doc, word, topic) * entry.getValue();
                    }

                    probability_doc_given_topic.set(topic, doc, sum);
                }
                probability_doc_given_topic.normalize(topic);

                double sum = 0;
                for(int doc = 0; doc < docCount; ++doc){
                    Document basicDocument = documents.get(doc);
                    Map<Integer, Integer> wordCounts = basicDocument.getWordCounts();

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
           Map<Integer, Integer> wordCounts = basicDocument.getWordCounts();

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
               if(!Double.isNaN(logSum)) continue;
               L += entry.getValue() * logSum;
           }
       }

       return L;

   }






}

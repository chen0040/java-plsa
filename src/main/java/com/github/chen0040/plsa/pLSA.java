package com.github.chen0040.plsa;

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

    private double[][][] probability_topic_given_doc_and_word = null;
    private double[] probability_topic = null;
    private double[][] probability_doc_given_topic = null;
    private double[][] probability_word_given_topic = null;
    private int topicCount = 20;
    private int docCount = -1;
    private int wordCount = -1;
    private Vocabulary vocabulary;
    private StopWordRemoval stopWordRemoval;
    private int maxIters = 100;
    private Random random = new Random();
    private double loglikelihood = Double.NEGATIVE_INFINITY;
    private int maxVocabularySize = 100;

    public pLSA(){
        stopWordRemoval = new StopWordRemoval();
    }

    public String wordAtIndex(int word){
        return vocabulary.get(word);
    }

    public int getDocCount() {
        return docCount;
    }



    public pLSA makeCopy() {
        pLSA clone = new pLSA();
        clone.copy(this);

        return clone;
    }

    public void copy(pLSA that){
        this.probability_topic_given_doc_and_word = clone(that.probability_topic_given_doc_and_word);
        this.probability_topic = clone(that.probability_topic);
        this.probability_doc_given_topic = clone(that.probability_doc_given_topic);
        this.probability_word_given_topic = clone(that.probability_word_given_topic);
        this.topicCount = that.topicCount;
        this.docCount = that.docCount;
        this.wordCount = that.wordCount;
        this.vocabulary = that.vocabulary.makeCopy();
        this.stopWordRemoval = that.stopWordRemoval.makeCopy();
        this.maxIters = that.maxIters;
        this.loglikelihood = that.loglikelihood;
    }

    private double[][][] clone(double[][][] rhs){
        if(rhs==null) return null;
        int m = rhs.length;
        double[][][] clone = new double[m][][];
        for(int i=0; i < m; ++i){
            clone[i] = clone(rhs[i]);
        }
        return clone;
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

    private void buildVocab(List<Document> documents){
        vocabulary = new BasicVocabulary();

        int m = documents.size();
        Map<String, Integer> uniqueWords = new HashMap<>();
        for(int i=0; i < m; ++i){
            Document doc = documents.get(i);
            for(Map.Entry<String, Integer> entry : doc.getWordCounts().entrySet()){
                uniqueWords.put(entry.getKey(), uniqueWords.getOrDefault(entry.getKey(), 0) + entry.getValue());
            }
        }

        List<TupleTwo<String,Integer>> words = uniqueWords.entrySet().stream().map(entry -> new TupleTwo<>(entry.getKey(), entry.getValue())).collect(Collectors.toList());

        words.sort((a, b) -> -Integer.compare(a._2(), b._2()));

        List<String> candidates = new ArrayList<>();

        for(int i=0; candidates.size() < maxVocabularySize && i < words.size(); ++i){
            if(stopWordRemoval.accept(words.get(i)._1())) {
                candidates.add(words.get(i)._1());
            }
        }

        candidates = stopWordRemoval.filter(candidates);
        vocabulary.setWords(candidates);
    }

    public List<Map.Entry<Integer, Double>> getTopRankingTopics4Doc(int doc, int limits){
        final double[] probs = new double[topicCount];
        List<Integer> topic_orders = new ArrayList<Integer>();
        for(int topic = 0; topic < topicCount; ++topic){
            probs[topic] = probability_topic[topic] * probability_doc_given_topic[topic][doc];
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
            probs[doc] = probability_doc_given_topic[topic][doc];
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
            probs[word] = probability_word_given_topic[topic][word];
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

    public void fit(List<Document> documents){

        if(vocabulary ==null) {
            buildVocab(documents);
        }

        docCount = documents.size();
        wordCount = vocabulary.getLength();

        probability_topic = new double[topicCount];
        probability_doc_given_topic = new double[topicCount][];
        probability_word_given_topic = new double[topicCount][];
        probability_topic_given_doc_and_word = new double[docCount][][];

        for(int topic = 0; topic < topicCount; ++topic) {
            probability_doc_given_topic[topic] = new double[docCount];
            probability_topic[topic] = 1.0 / topicCount;

            for(int doc = 0; doc < docCount; ++doc){
                probability_doc_given_topic[topic][doc] = random.nextDouble();
            }
            normalize(probability_doc_given_topic[topic]);

            probability_word_given_topic[topic] = new double[wordCount];

            for(int word = 0; word < wordCount; ++word){
                probability_word_given_topic[topic][word] = random.nextDouble();
            }
            normalize(probability_word_given_topic[topic]);
        }



        for(int doc = 0; doc < docCount; ++doc){
            probability_topic_given_doc_and_word[doc] = new double[wordCount][];

            for(int word = 0; word < wordCount; ++word){
                probability_topic_given_doc_and_word[doc][word] = new double[topicCount];
            }
        }

        for(int iters = 0; iters < maxIters; ++iters){

            // E-step
            for(int doc = 0; doc < docCount; ++doc){
                for(int word = 0; word < wordCount; ++word) {
                    for(int topic = 0; topic < topicCount; ++topic) {
                        probability_topic_given_doc_and_word[doc][word][topic] = probability_topic[topic]
                                * probability_doc_given_topic[topic][doc]
                                * probability_word_given_topic[topic][word];
                    }

                    normalize(probability_topic_given_doc_and_word[doc][word]);
                }

            }


            // M-step
            for(int topic = 0; topic < topicCount; ++topic){

                for(int word = 0; word < wordCount; ++word) {

                    // update P (word | topic) /prop sum_{doc} (P(topic | word, doc) * count(word in doc))
                    double sum = 0;
                    for (int doc = 0; doc < docCount; ++doc) {
                        Document basicDocument = documents.get(doc);
                        Map<String, Integer> wordCounts = basicDocument.getWordCounts();

                        sum += probability_topic_given_doc_and_word[doc][word][topic] * wordCounts.getOrDefault(vocabulary.get(word), 0);
                    }
                    probability_word_given_topic[topic][word] = sum;
                }
                normalize(probability_word_given_topic[topic]);

                for(int doc = 0; doc < docCount; ++doc){
                    Document basicDocument = documents.get(doc);
                    Map<String, Integer> wordCounts = basicDocument.getWordCounts();

                    // update P (doc | topic) /prop sum_{word} (P(topic | word, doc) * count(word in doc))
                    double sum = 0;
                    for(int word = 0; word < wordCount; ++word){
                        sum += probability_topic_given_doc_and_word[doc][word][topic] * wordCounts.getOrDefault(vocabulary.get(word), 0);
                    }

                    probability_doc_given_topic[topic][doc] = sum;
                }
                normalize(probability_doc_given_topic[topic]);

                double sum = 0;
                for(int doc = 0; doc < docCount; ++doc){
                    Document basicDocument = documents.get(doc);
                    Map<String, Integer> wordCounts = basicDocument.getWordCounts();

                    for(int word = 0; word < wordCount; ++word){
                        sum += probability_topic_given_doc_and_word[doc][word][topic] * wordCounts.getOrDefault(vocabulary.get(word), 0);
                    }
                }
                probability_topic[topic] = sum;

            }

            // Normalize
            normalize(probability_topic);

            loglikelihood = calcLogLikelihood(documents);

            logger.info("#: {} log-likelihood: {}", iters, loglikelihood);
        }
    }

   private double calcLogLikelihood(List<Document> batch){
       int m = batch.size();
       int N = vocabulary.getLength();

       double L = 0.0;

       for(int doc = 0; doc < m; ++doc){
           Document basicDocument = batch.get(doc);
           Map<String, Integer> wordCounts = basicDocument.getWordCounts();

           for(int word = 0; word < N; ++word) {
               double sum = 0;

               for(int topic = 0; topic < topicCount; ++topic) {
                   double value = probability_topic[topic]
                           * probability_doc_given_topic[topic][doc]
                           * probability_word_given_topic[topic][word];
                   sum += value;
               }

               L += wordCounts.getOrDefault(vocabulary.get(word), 0) * Math.log(sum);
           }
       }

       return L;

   }


    private void normalize(double[] values){
        int m = values.length;
        double sum = sum(values);
        if(sum > 0) {
            for (int i = 0; i < m; ++i) {
                values[i] /= sum;
            }
        }
    }

    private double sum(double[] values){
        double sum = 0;
        for(int i=0; i < values.length; ++i){
            sum += values[i];
        }
        return sum;
    }




}

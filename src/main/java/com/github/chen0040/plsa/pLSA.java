package com.github.chen0040.plsa;

import java.util.*;
import java.util.stream.Collectors;


/**
 * Created by xschen on 9/16/15.
 * probabilistic Latent Semantic Analysis
 */
public class pLSA {
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

    public pLSA(){
        stopWordRemoval = new StopWordRemoval();
    }

    public String wordAtIndex(int word){
        return vocabulary.get(word);
    }

    public int getDocCount() {
        return docCount;
    }

    public void setDocCount(int docCount) {
        this.docCount = docCount;
    }

    public int getWordCount() {
        return wordCount;
    }

    public void setWordCount(int wordCount) {
        this.wordCount = wordCount;
    }

    public double[][][] getProbability_topic_given_doc_and_word() {
        return probability_topic_given_doc_and_word;
    }

    public void setProbability_topic_given_doc_and_word(double[][][] probability_topic_given_doc_and_word) {
        this.probability_topic_given_doc_and_word = probability_topic_given_doc_and_word;
    }

    public double[] getProbability_topic() {
        return probability_topic;
    }

    public void setProbability_topic(double[] probability_topic) {
        this.probability_topic = probability_topic;
    }

    public double[][] getProbability_doc_given_topic() {
        return probability_doc_given_topic;
    }

    public void setProbability_doc_given_topic(double[][] probability_doc_given_topic) {
        this.probability_doc_given_topic = probability_doc_given_topic;
    }

    public double[][] getProbability_word_given_topic() {
        return probability_word_given_topic;
    }

    public void setProbability_word_given_topic(double[][] probability_word_given_topic) {
        this.probability_word_given_topic = probability_word_given_topic;
    }

    public StopWordRemoval getStopWordRemoval() {
        return stopWordRemoval;
    }

    public void setStopWordRemoval(StopWordRemoval stopWordRemoval) {
        this.stopWordRemoval = stopWordRemoval;
    }

    public double getLoglikelihood() {
        return loglikelihood;
    }

    public void setLoglikelihood(double loglikelihood) {
        this.loglikelihood = loglikelihood;
    }

    public int getTopicCount(){
        return topicCount;
    }

    public void setTopicCount(int K){
        this.topicCount = K;
    }

    public int getMaxIters(){
        return maxIters;
    }

    public void setMaxIters(int maxIters){
        this.maxIters = maxIters;
    }

    public Vocabulary getVocabulary(){
        return vocabulary;
    }

    public void setVocabulary(Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
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

    private void buildVocab(List<Document> batch){
        vocabulary = new BasicVocabulary();

        int m = batch.size();
        Set<String> uniqueWords = new HashSet<>();
        for(int i=0; i < m; ++i){
            Document doc = batch.get(i);
            uniqueWords.addAll(doc.getWordCounts().keySet().stream().collect(Collectors.toList()));
        }

        List<String> candidates = new ArrayList<>();
        for(String word : uniqueWords){
            candidates.add(word);
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

        Collections.sort(topic_orders, new Comparator<Integer>() {
            public int compare(Integer t1, Integer t2) {
                return Double.compare(probs[t2], probs[t1]);
            }
        });

        List<Map.Entry<Integer, Double>> topRankedTopics = new ArrayList<Map.Entry<Integer, Double>>();
        limits = Math.min(limits, topicCount);
        for(int i = 0; i < limits; ++i){
            int topic = topic_orders.get(i);
            topRankedTopics.add(new AbstractMap.SimpleEntry<Integer, Double>(topic, probs[topic]));
        }
        return topRankedTopics;
    }

    public List<Map.Entry<Integer, Double>> getTopRankingDocs4Topic(int topic, int limits){
        final double[] probs = new double[docCount];
        List<Integer> doc_orders = new ArrayList<Integer>();
        for(int doc = 0; doc < docCount; ++doc){
            probs[doc] = probability_doc_given_topic[topic][doc];
            doc_orders.add(doc);
        }

        Collections.sort(doc_orders, new Comparator<Integer>() {
            public int compare(Integer t1, Integer t2) {
                return Double.compare(probs[t2], probs[t1]);
            }
        });

        List<Map.Entry<Integer, Double>> topRankedDocs = new ArrayList<Map.Entry<Integer, Double>>();
        limits = Math.min(limits, docCount);
        for(int i = 0; i < limits; ++i){
            int doc = doc_orders.get(i);
            topRankedDocs.add(new AbstractMap.SimpleEntry<Integer, Double>(doc, probs[doc]));
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

        Collections.sort(word_orders, new Comparator<Integer>() {
            public int compare(Integer t1, Integer t2) {
                return Double.compare(probs[t2], probs[t1]);
            }
        });

        List<Map.Entry<Integer, Double>> topRankedWords = new ArrayList<Map.Entry<Integer, Double>>();
        limits = Math.min(limits, wordCount);
        for(int i = 0; i < limits; ++i){
            int word = word_orders.get(i);
            topRankedWords.add(new AbstractMap.SimpleEntry<Integer, Double>(word, probs[word]));
        }
        return topRankedWords;

    }

    public void fit(List<Document> batch){

        if(vocabulary ==null) {
            buildVocab(batch);
        }

        docCount = batch.size();
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
                        Document basicDocument = batch.get(doc);
                        HashMap<String, Integer> wordCounts = basicDocument.getWordCounts();

                        sum += probability_topic_given_doc_and_word[doc][word][topic] * wordCounts.getOrDefault(vocabulary.get(word), 0);
                    }
                    probability_word_given_topic[topic][word] = sum;
                }
                normalize(probability_word_given_topic[topic]);

                for(int doc = 0; doc < docCount; ++doc){
                    Document basicDocument = batch.get(doc);
                    HashMap<String, Integer> wordCounts = basicDocument.getWordCounts();

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
                    Document basicDocument = batch.get(doc);
                    HashMap<String, Integer> wordCounts = basicDocument.getWordCounts();

                    for(int word = 0; word < wordCount; ++word){
                        sum += probability_topic_given_doc_and_word[doc][word][topic] * wordCounts.getOrDefault(vocabulary.get(word), 0);
                    }
                }
                probability_topic[topic] = sum;

            }

            // Normalize
            normalize(probability_topic);

            loglikelihood = calcLogLikelihood(batch);
        }
    }

    private void isNormalized(double[] values){
        double sum = sum(values);
        if(sum != 1){
            System.out.println("normalized sum should be one: sum = " + sum);
        }
    }

   private double calcLogLikelihood(List<Document> batch){
       int m = batch.size();
       int N = vocabulary.getLength();

       double L = 0.0;

       for(int doc = 0; doc < m; ++doc){
           Document basicDocument = batch.get(doc);
           HashMap<String, Integer> wordCounts = basicDocument.getWordCounts();

           for(int word = 0; word < N; ++word) {
               double[] values = new double[topicCount];
               double sum = 0;

               for(int topic = 0; topic < topicCount; ++topic) {
                   double value = probability_topic[topic]
                           * probability_doc_given_topic[topic][doc]
                           * probability_word_given_topic[topic][word];



                   values[topic] = value;
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

    private void checkNaN(double[] values, String location){
        for(int i=0; i < values.length; ++i){
            if(Double.isNaN(values[i])){
                System.out.println(location + " produce NaN at " + i);
                System.exit(0);
            }
        }
    }



}

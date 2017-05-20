# java-plsa
Package provides the java implementation of probabilistic latent semantic analysis (pLSA)

[![Build Status](https://travis-ci.org/chen0040/java-plsa.svg?branch=master)](https://travis-ci.org/chen0040/java-plsa) [![Coverage Status](https://coveralls.io/repos/github/chen0040/java-plsa/badge.svg?branch=master)](https://coveralls.io/github/chen0040/java-plsa?branch=master)

# Usage
 
 The sample code belows illustrates how to perform topic modelling using pLSA
 
 ```java
 List<String> docs = Arrays.asList("[doc-1-content]", "[doc-2-content]", ...);
 
pLSA method = new pLSA();
method.setStemmerEnabled(true);

method.setMaxIters(10);
method.setMaxVocabularySize(1000);
method.fit(docs);

for(int topic = 0; topic < method.getTopicCount(); ++topic){
  List<TupleTwo<Document, Double>> topRankedDocs = method.getTopRankingDocs4Topic(topic, 3);
  List<TupleTwo<String, Double>> topRankedWords = method.getTopRankingWords4Topic(topic, 3);

  System.out.println("Topic "+topic+": ");

  System.out.println("Top Ranked Document:");
  for(TupleTwo<Document, Double> entry : topRankedDocs){
     Document doc = entry._1();
     double prob = entry._2();
     System.out.print(doc.docIndex()+"(" + prob +"), ");
     System.out.println(doc.content());
  }
  System.out.println();

  System.out.println("Top Ranked Words:");
  for(TupleTwo<String, Double> entry : topRankedWords){
     String word = entry._1();
     double prob = entry._2();
     System.out.print(word+"(" + prob +"), ");
  }
  System.out.println();
}

System.out.println("// ============================================= //");

for(int doc = 0; doc < method.getDocCount(); ++doc){
  List<TupleTwo<Integer, Double>> topRankedTopics = method.getTopRankingTopics4Doc(doc, 3);
  System.out.print("Doc "+doc+": ");
  for(TupleTwo<Integer, Double> entry : topRankedTopics){
     int topic = entry._1();
     double prob = entry._2();
     System.out.print(topic+"(" + prob +"), ");
  }
  System.out.println();
}
 ```

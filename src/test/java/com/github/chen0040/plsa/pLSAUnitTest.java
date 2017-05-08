package com.github.chen0040.plsa;


import com.github.chen0040.plsa.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.testng.Assert.*;


/**
 * Created by xschen on 9/5/2017.
 */
public class pLSAUnitTest {

   private static final Logger logger = LoggerFactory.getLogger(pLSAUnitTest.class);

   private List<Document> getDocs() throws IOException {

      InputStream inputStream = FileUtils.getResource("documents.txt");

      List<Document> docs = new ArrayList<>();

      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
      reader.lines().forEach(line->{
         line = line.trim();
         if(line.equals("")) return;

         String[] fields = line.split("\t");
         String text = fields[0];
         if (fields.length == 3) {
            text = fields[2];
         }

         Document document = new BasicDocument(text);
         docs.add(document);
      });
      reader.close();

      return docs;
   }

   @Test
   public void testPLSA() throws IOException {
      List<Document> docs = getDocs();

      pLSA method = new pLSA();
      method.setMaxIters(10);
      method.setMaxVocabularySize(100);
      method.fit(docs);

      for(int topic = 0; topic < method.getTopicCount(); ++topic){
         List<Map.Entry<Integer, Double>> topRankedDocs = method.getTopRankingDocs4Topic(topic, 3);
         List<Map.Entry<Integer, Double>> topRankedWords = method.getTopRankingWords4Topic(topic, 3);

         System.out.println("Topic "+topic+": ");

         System.out.println("Top Ranked Document:");
         for(Map.Entry<Integer, Double> entry : topRankedDocs){
            int doc = entry.getKey();
            double prob = entry.getValue();
            System.out.print(doc+"(" + prob +"), ");
         }
         System.out.println();

         System.out.println("Top Ranked Words:");
         for(Map.Entry<Integer, Double> entry : topRankedWords){
            int word = entry.getKey();
            double prob = entry.getValue();
            System.out.print(method.wordAtIndex(word)+"(" + prob +"), ");
         }
         System.out.println();
      }

      for(int doc = 0; doc < method.getDocCount(); ++doc){
         List<Map.Entry<Integer, Double>> topRankedTopics = method.getTopRankingTopics4Doc(doc, 3);
         System.out.print("Doc "+doc+": ");
         for(Map.Entry<Integer, Double> entry : topRankedTopics){
            int topic = entry.getKey();
            double prob = entry.getValue();
            System.out.print(topic+"(" + prob +"), ");
         }
         System.out.println();
      }
   }
}

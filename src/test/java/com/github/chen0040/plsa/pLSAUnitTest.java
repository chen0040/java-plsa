package com.github.chen0040.plsa;


import com.github.chen0040.data.utils.TupleTwo;
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

   private List<String> getDocs() throws IOException {

      InputStream inputStream = FileUtils.getResource("documents.txt");

      List<String> docs = new ArrayList<>();

      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
      reader.lines().forEach(line->{
         line = line.trim();
         if(line.equals("")) return;

         String[] fields = line.split("\t");
         String text = fields[0];
         if (fields.length == 3) {
            text = fields[2];
         }

         docs.add(text);
      });
      reader.close();

      return docs;
   }

   @Test
   public void testPLSA() throws IOException {
      List<String> docs = getDocs();

      pLSA method = new pLSA();
      method.setStemmerEnabled(true);

      method.setMaxIters(1);
      method.setMaxVocabularySize(10);
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
   }
}

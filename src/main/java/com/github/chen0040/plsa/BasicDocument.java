package com.github.chen0040.plsa;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by xschen on 9/9/15.
 */
public class BasicDocument implements Document {

    private final String text;
    private final Map<String, Integer> wordCounts = new HashMap<>();
    public BasicDocument(String text){
        this.text = text;

        List<String> words = BasicTokenizer.doTokenize(text);

        for(String word : words) {
            wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
        }
    }

    public String getText(){
        return text;
    }

    public Map<String, Integer> getWordCounts(){
        return wordCounts;
    }
}

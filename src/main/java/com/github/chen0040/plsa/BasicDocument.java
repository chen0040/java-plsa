package com.github.chen0040.plsa;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


/**
 * Created by xschen on 9/9/15.
 */
public class BasicDocument implements Document {

    private final Map<Integer, Integer> wordCounts;
    private final List<Integer> words;
    private String text;

    public BasicDocument(Map<Integer, Integer> wordCounts, String text){
        this.wordCounts = wordCounts;
        this.text = text;

        this.words = new ArrayList<>(wordCounts.keySet());
    }

    public Map<Integer, Integer> indexedWordCount(){
        return wordCounts;
    }


    @Override public List<Integer> wordIndices() {
        return words;
    }

    public String getContent(){
        return text;
    }
}

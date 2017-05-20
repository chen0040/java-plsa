package com.github.chen0040.plsa;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by xschen on 9/9/15.
 */
public interface Document {

    Map<Integer, Integer> getWordCounts();
    List<Integer> words();
}

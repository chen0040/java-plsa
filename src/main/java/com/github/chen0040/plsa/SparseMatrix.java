package com.github.chen0040.plsa;


import java.util.HashMap;
import java.util.Map;


/**
 * Created by xschen on 9/5/2017.
 */
public class SparseMatrix {
   private int dim1;
   private int dim2;
   private int dim3;
   private static final double EPSILON = 0.00000000000000001;

   private final Map<Integer, Double> values = new HashMap<>();

   public SparseMatrix(int dim1, int dim2, int dim3) {
      this.dim1 = dim1;
      this.dim2 = dim2;
      this.dim3 = dim3;
   }

   public SparseMatrix(int dim1, int dim2) {
      this.dim1 = dim1;
      this.dim2 = dim2;
      this.dim3 = 1;
   }

   public SparseMatrix(int dim1) {
      this.dim1 = dim1;
      this.dim2 = 1;
      this.dim3 = 1;
   }


   public double get(int index1, int index2, int index3) {
      return values.getOrDefault(index(index1, index2, index3), 0.0);
   }

   public void set(int index1, int index2, int index3, double value){
      int index = index(index1, index2, index3);
      if(Math.abs(value) < EPSILON){
         values.remove(index);
      } else {
         values.put(index, value);
      }
   }

   public void set(int index1, int index2, double value){
      set(index1, index2, 0, value);
   }

   public void set(int index1, double value){
      set(index1, 0, value);
   }

   private int index(int index1, int index2, int index3){
      return index1 * dim2 * dim3 + index2 * dim3 + index3;
   }

   public double sum(int index1, int index2) {
      double sum = 0;
      for(int i=0; i < dim3; ++i){
         sum += get(index1, index2, i);
      }
      return sum;
   }

   public double sum(int index1) {
      double sum = 0;
      for(int index2 = 0; index2 < dim2; ++index2) {
         for (int index3 = 0; index3 < dim3; ++index3) {
            sum += get(index1, index2, index3);
         }
      }
      return sum;
   }


   public double sum() {
      double sum = 0;
      for(int index1 = 0; index1 < dim1; ++index1) {
         for (int index2 = 0; index2 < dim2; ++index2) {
            for (int index3 = 0; index3 < dim3; ++index3) {
               sum += get(index1, index2, index3);
            }
         }
      }

      return sum;
   }

   public void normalize(int index1, int index2){

      double sum = sum(index1, index2);
      if(sum > 0) {
         for (int i = 0; i < dim3; ++i) {
            set(index1, index2, i, get(index1, index2, i) / sum);
         }
      }
   }

   public void normalize(int index1){

      double sum = sum(index1);
      if(sum > 0) {
         for(int index2 = 0; index2 < dim2; ++index2) {
            for (int index3 = 0; index3 < dim3; ++index3) {
               set(index1, index2, index3, get(index1, index2, index3) / sum);
            }
         }
      }
   }

   public void normalize(){

      double sum = sum();
      if(sum > 0) {
         for(int index1= 0; index1 < dim1; ++index1) {
            for (int index2 = 0; index2 < dim2; ++index2) {
               for (int index3 = 0; index3 < dim3; ++index3) {
                  set(index1, index2, index3, get(index1, index2, index3) / sum);
               }
            }
         }
      }
   }


   public SparseMatrix makeCopy() {
      SparseMatrix clone = new SparseMatrix(dim1, dim2, dim3);
      clone.copy(this);
      return clone;
   }

   public void copy(SparseMatrix that){
      this.dim1 = that.dim1;
      this.dim2 = that.dim2;
      this.dim3 = that.dim3;

      this.values.clear();
      for(Map.Entry<Integer,Double> entry : that.values.entrySet()){
         this.values.put(entry.getKey(),  entry.getValue());
      }
   }


   public double get(int index1, int index2) {
      return get(index1, index2, 0);
   }

   public double get(int index1){
      return get(index1, 0);
   }
}

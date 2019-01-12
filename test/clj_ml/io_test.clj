(ns clj-ml.io-test
  (:require [clojure.string :as str]
            [clojure.java.io :as jio])
  (:use [clj-ml io data] :reload-all)
  (:use clojure.test midje.sweet))

(deftest test-load-instances-iris-arff-url
  (let [ds (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
               (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
        ds-sparse (do (println "Loading instances from http://clj-ml.artifice.cc/testsparse.arff ...")
                      (load-instances :arff "http://clj-ml.artifice.cc/testsparse.arff"))
        ds-nonsparse (do (println "Loading instances from http://clj-ml.artifice.cc/testnonsparse.arff ...")
                         (load-instances :arff "http://clj-ml.artifice.cc/testnonsparse.arff"))]
    (is (= 150 (dataset-count ds)))
    (is (= 2 (dataset-count ds-sparse)))
    (is (= 2 (dataset-count ds-nonsparse)))
    (is (= weka.core.DenseInstance (class (first (dataset-seq ds)))))
    (is (= weka.core.SparseInstance (class (first (dataset-seq ds-sparse)))))
    (is (= (dataset-as-vecs ds-sparse) (dataset-as-vecs ds-nonsparse)))))

(deftest test-load-instances-iris-csv-url
  (let [ds (do (println "Loading instances from http://clj-ml.artifice.cc/iris.csv ...")
               (load-instances :csv "http://clj-ml.artifice.cc/iris.csv"))]
    (is (= 150 (dataset-count ds)))))

(deftest test-save-instances
  (let [ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        ds-sparse (make-sparse-dataset :test
                                       [:age :iq {:favorite-color [:none :red :blue :green]}]
                                       [{0 12 2 :red}
                                        {1 110 2 :blue}
                                        {0 Double/NaN 1 120}]
                                       {:weight 2})]
    (is (= (let [_ (save-instances :csv "test.csv" ds)
                res(str/replace (slurp "test.csv") #"\r\n" "\n")
                _ (jio/delete-file "test.csv")]
             res)
           "a,b,c\n1,2,m\n4,5,m\n"))
    (is (= (let [_  (save-instances :arff "test.arff" ds)
               res (str/replace (slurp "test.arff") #"\r\n" "\n")
               _ (jio/delete-file "test.arff")]
             res)
           "@relation test\n\n@attribute a numeric\n@attribute b numeric\n@attribute c {m,n}\n\n@data\n1,2,m\n4,5,m\n"))
    (is (= (let [_  (save-instances :arff "testsparse.arff" ds-sparse)
               res (str/replace (slurp "testsparse.arff") #"\r\n" "\n")
                _ (jio/delete-file "testsparse.arff")]
             res)
           "@relation test\n\n@attribute age numeric\n@attribute iq numeric\n@attribute favorite-color {none,red,blue,green}\n\n@data\n{0 12,2 red},{2}\n{1 110,2 blue},{2}\n{0 ?,1 120},{2}\n"))))


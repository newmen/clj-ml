(ns clj-ml.test-association-rules
  (:require [clj-ml.association-rules :refer :all]
            [clojure.test :refer :all]))

(deftest fpgrowth
  (is (= weka.associations.FPGrowth)
      (class  (make-association-rule-miner :fpgrowth {})))
  (is (= weka.associations.FPGrowth)
      (class  (make-association-rule-miner :fpgrowth {:num-rules 20
                                                      :metric 1
                                                      :min-metric-score 0.3
                                                      :min-support-upper-bound 0.4
                                                      :min-support-lower-bound 0.7
                                                      :delta-min-support 0.8}))))
(deftest apriori
  (is (= weka.associations.Apriori)
      (class  (make-association-rule-miner :apriori {})))
  (is (= weka.associations.Apriori)
      (class  (make-association-rule-miner :apriori {:output-itemsets true
                        :remove-missing-value-columns true
                        :report-iterative-progress true
                                                     :num-rules 20
                                                      :metric 1
                                                      :delta-min-support 0.8}))))

(slurp "http://fimi.ua.ac.be/data/T10I4D100K.dat")

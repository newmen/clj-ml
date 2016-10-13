;;
;; Association rules
;; @author @shark8me
;;

(ns clj-ml.association-rules
  "Association rule mining "
  (:use [clj-ml utils data kernel-functions options-utils])
  (:import (java.util Date Random)
           (weka.associations Apriori FPGrowth Associator AssociationRulesProducer)
           (weka.core Instance Instances)))

;; Setting up options
(defmulti #^{:skip-wiki true}
            make-association-rules-options
  "Creates the right parameters for an association rules miner Returns the parameters as a Clojure vector."
  (fn [algorithm map] [algorithm]))

(defmethod make-association-rules-options [:apriori]
  ([algorithm m]
   (->> (check-options m
                       {:output-itemsets "-I"
                        :remove-missing-value-columns "-R"
                        :report-iterative-progress "-V"})
        (check-option-values m
                             {:num-rules "-N"
                              :metric "-T"
                              :delta-min-support "-D"}))))

(defmethod make-association-rules-options [:fpgrowth]
  ([algorithm m]
   (check-option-values m
                        {:num-rules "-N"
                         :metric "-T"
                         :min-metric-score "-C"
                         :min-support-upper-bound "-U"
                         :min-support-lower-bound "-M"
                         :delta-min-support "-D"})))

;; Building the ARM


(defn make-arm-with
  #^{:skip-wiki true}
    [algorithm ^Class arm-class options]
  (capture-out-err
   (let [options-read (if (empty? options) {} (first options))
         ^Associator asc (.newInstance arm-class)
         opts (into-array String (make-association-rules-options algorithm options-read))]
     (.setOptions asc opts)
     asc)))

(defmulti make-association-rule-miner
  " "
  (fn [algorithm & options] [algorithm]))

(defmethod make-association-rule-miner [:apriori]
  ([algorithm & options]
   (make-arm-with algorithm Apriori options)))

(defmethod make-association-rule-miner [:fpgrowth]
  ([algorithm & options]
   (make-arm-with algorithm FPGrowth options)))

;; fit association rules 

(defn build-associations
  "Build associations in the given dataset"
  ([^Associator asc dataset]
   (do (.buildAssociations asc dataset)
       asc)))

(defn get-association-rules
  "Performs a deep copy of the classifier"
  [^AssociationRulesProducer asc]
  (.getAssociationRules asc))

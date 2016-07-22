;;
;; Estimators
;;

(ns clj-ml.estimators
  (:use [clj-ml utils data kernel-functions options-utils])
  (:import (java.util Date Random)
           (weka.core Instance Instances)
           (weka.classifiers.bayes.net.estimate DiscreteEstimatorBayes)
           (weka.estimators NormalEstimator
                            KernelEstimator
                            UnivariateKernelEstimator
                            UnivariateDensityEstimator)))

;; Estimator options

(defmulti #^{:skip-wiki true}
            make-estimator
  "Creates the right parameters for a estimator."
  (fn [kind & options] [kind options]))

(defmethod make-estimator [:normal-estimator]
  [_ & {:keys [precision]}]
  {:pre [(opt-check precision :precision)]}
  (new NormalEstimator precision))

(defmethod make-estimator [:kernel-estimator]
  [_ & {:keys [precision]}]
  {:pre [(opt-check precision :precision)]}
  (new KernelEstimator precision))

(defmethod make-estimator [:discrete-estimator-bayes]
  [_ & {:keys [nsymbols fprior]}]
  {:pre [(opt-check nsymbols :nsymbols)
         (opt-check fprior :fprior)]}
  (new DiscreteEstimatorBayes nsymbols fprior))

((seq (vector 1 2)))
(defmulti abcd (fn [a b] b))
(defmethod abcd ::collection
  [args]
  (println "seq"))

(defmethod abcd String
  [args]
  (println "not seq"))

(abcd 12 "asd")
(abc [])
(abc '())
(abc {})
(abc #{})
(defmethod estimator-update class)

(defmethod estimator-update ::collection
  ([estimator data] (add-values estimator data (repeat (count data) 1.0)))
  ([estimator data weight]
   (if (is-dataset? data)
     (. estimator addValues)))
  ([estimator data weight attrIndex classIndex classValue]))
([estimator data weight attrIndex classIndex classValue amin amax])

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

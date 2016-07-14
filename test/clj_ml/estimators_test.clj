(ns clj-ml.estimators-test
  (:use [clj-ml estimators] :reload-all)
  (:use [clojure.test])
  (:import (weka.core Instance Instances)
           (weka.classifiers.bayes.net.estimate DiscreteEstimatorBayes)
           (weka.estimators NormalEstimator
                            KernelEstimator
                            UnivariateKernelEstimator
                            UnivariateDensityEstimator)))

(deftest kernel-estimator
  (is (thrown? java.lang.AssertionError
               (clj-ml.estimators/make-estimator :kernel-estimator)))
  (is (instance? KernelEstimator
                 (clj-ml.estimators/make-estimator :kernel-estimator :precision 0.1))))

(deftest normal-estimator
  (is (thrown? java.lang.AssertionError
               (clj-ml.estimators/make-estimator :normal-estimator)))
  (is (instance? NormalEstimator
                 (clj-ml.estimators/make-estimator :normal-estimator :precision 0.1))))

(deftest discrete-estimator-bayes
  (is (thrown? java.lang.AssertionError
               (clj-ml.estimators/make-estimator :discrete-estimator-bayes)))
  (is (thrown? java.lang.AssertionError
               (clj-ml.estimators/make-estimator :discrete-estimator-bayes :nsymbols 2)))
  (is (thrown? java.lang.AssertionError
               (clj-ml.estimators/make-estimator :discrete-estimator-bayes :fprior 0.2)))
  (is (instance? DiscreteEstimatorBayes
                 (clj-ml.estimators/make-estimator :discrete-estimator-bayes :fprior 0.2
                                                   :nsymbols 2))))
(comment
  (run-tests))

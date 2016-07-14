;;
;; Utilities for converting clojure hash maps into Weka string options
;; @author Ben Mabey
;;

(ns #^{:author "Ben Mabey <ben@benmabey.com>"
       :skip-wiki true}
      clj-ml.options-utils
  (:use [clj-ml utils])
  (:require [clojure [string :as str]]))

;; Manipulation of array of options

(defn check-option [opts val flag map]
  "Sets an option for a filter"
  (if (get map val)
    (conj opts flag)
    opts))

(defn check-option-value [opts val flag map]
  "Sets an option with value for a filter"
  (if-let [val-in-map (get map val)]
    (vec (concat opts [flag (str val-in-map)]))
    opts))

;; attr-name and dataset-index-attr copy and pasted from data due to Clojure's inability
;; to handle circular dependencies. :(
(defn- attr-name [^weka.core.Attribute attr]
  (.name attr))

(defn- dataset-index-attr
  "Returns the index of an attribute in the attributes definition of a dataset."
  [^weka.core.Instances dataset attr]
  (if (number? attr)
    attr
    (find-first #(= (name attr) (attr-name (.attribute dataset (int %)))) (range (.numAttributes dataset)))))

(defn extract-attributes
  "Transforms the :attributes value from m into the appropriate weka flag"
  ([m] (extract-attributes "-R" m))
  ([flag m] (extract-attributes flag :attributes m))
  ([flag key-name m]
   (if-let [attributes (key-name m)]
     [flag (str/join ","
                     (for [attr attributes]
                       (if (string? attr) attr
                           (inc (dataset-index-attr (:dataset-format m) attr)))))]
     [])))

;; TODO: Raise a helpful exception when the keys don't match up with the provided flags.
(defn check-options
  "Checks the presence of a set of options for a filter"
  ([args-map opts-map]
   (check-options args-map opts-map []))
  ([args-map opts-map options-so-far]
   (reduce (fn [so-far [option flag]] (check-option so-far option flag args-map)) options-so-far opts-map)))

(defn check-option-values
  "Checks the presence of a set of options with value for a filter"
  ([args-map opts-map]
   (check-option-values args-map opts-map []))
  ([args-map opts-map options-so-far]
   (reduce (fn [so-far [option flag]] (check-option-value so-far option flag args-map)) options-so-far opts-map)))

(defn option-handler
  "Creates ther right parameters for objects that implement the 
  OptionHandler interface"
  ([map]
   (let [cols (get map :attributes)
         pre-cols (reduce #(str %1 "," (+ %2 1)) "" cols)
         cols-val-a ["-R" (.substring pre-cols 1 (.length pre-cols))]
         cols-val-b (check-options {:invert "-V"
                                    :no-normalization "-D"}
                                   map
                                   cols-val-a)]
     (into-array cols-val-b))))

;;TODO: replace with clojure.spec post 1.9 
(defn opt-check
  "Used as method precondition to check for option correctness.
  "
  ([opt opt-name] (opt-check opt opt-name (comp not nil?)))
  ([opt opt-name checkfn]
   (or (checkfn opt)
       (throw (java.lang.AssertionError. (format "mandatory argument value for %s not provided " opt-name))))))

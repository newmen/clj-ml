(defproject cc.artifice/clj-ml "0.8.10"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :java-source-paths ["src/java"]
  :license {:name "MIT License"
            :url "http://opensource.org/licenses/MIT"}
  :url "https://github.com/joshuaeckroth/clj-ml"
  :dependencies [[org.clojure/clojure "1.10.2"]
                 [nz.ac.waikato.cms.weka/weka-dev "3.9.5"]
                 [nz.ac.waikato.cms.weka/chiSquaredAttributeEval "1.0.4" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/attributeSelectionSearchMethods "1.0.7" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/classifierBasedAttributeSelection "1.0.5" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/linearForwardSelection "1.0.2" :exclusions [nz.ac.waikato.cms.weka/weka-dev nz.ac.waikato.cms.weka/classifierBasedAttributeSelection]]
                 [nz.ac.waikato.cms.weka/rotationForest "1.0.3" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/paceRegression "1.0.2" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/SPegasos "1.0.2" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/racedIncrementalLogitBoost "1.0.2" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/partialLeastSquares "1.0.5" :exclusions [nz.ac.waikato.cms.weka/weka-dev]]
                 [nz.ac.waikato.cms.weka/LibSVM "1.0.10" :exclusions [nz.ac.waikato.cms.weka/weka-dev tw.edu.ntu.csie/libsvm]]
                 [junit/junit "4.13"]
                 [tw.edu.ntu.csie/libsvm "3.25"]
                 [org.clojure/data.xml "0.0.8"]
                 [org.apache.lucene/lucene-analyzers-common "4.10.1"]
                 [org.apache.lucene/lucene-snowball "3.0.3"]]
  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.7"]]
                   :dependencies [[midje "1.9.8"]]}}
  :codox {:output-path "codox"
          :src-linenum-anchor-prefix "L"})

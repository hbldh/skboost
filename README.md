# skboost

[![Build Status](https://travis-ci.org/hbldh/skboost.svg)](https://travis-ci.org/hbldh/skboost)

Boosting Algorithms compatible with [scikit-learn](http://scikit-learn.org/).

## Boosting Algorithms

The `skboost` package contains implementations of some boosting algorithms that
are outside the scope of [scikit-learn](http://scikit-learn.org/).

The main point of interest is the MILBoost algorithm, which performs boosting
with a Multiple Instance Learning formulation.

### MILBoost

See \[1\], \[2\] and \[4\].

### GentleBoost

See \[3\].

### LogitBoost

See \[3\].

## Datasets

This repository includes a vendored copy of the MUSK datasets (\[5\]), both version 1 and version 2.
These are used for multiple instance learning benchmarks:

> This dataset describes a set of 92 molecules of which 47 are judged by human experts 
> to be musks and the remaining 45 molecules are judged to be non-musks. The goal is 
> to learn to predict whether new molecules will be musks or non-musks. However, the 166 
> features that describe these molecules depend upon the exact shape, or conformation, 
> of the molecule. Because bonds can rotate, a single molecule can adopt many different 
> shapes. To generate this data set, the low-energy conformations of the molecules were 
> generated and then filtered to remove highly similar conformations. This left 476 
> conformations. Then, a feature vector was extracted that describes each conformation. 
>
> This many-to-one relationship between feature vectors and molecules is 
> called the "multiple instance problem". When learning a classifier for this data, 
> the classifier should classify a molecule as "musk" if ANY of its conformations is 
> classified as a musk. A molecule should be classified as "non-musk" if NONE of its 
> conformations is classified as a musk.

## References

\[1\] [B. Babenko, P. Dollar, Z. Tu, and S. Belongie. Simultaneous learning
and alignment: Multi-instance and multi-pose learning. In Faces in
Real-Life Images, October 2008.](http://vision.ucsd.edu/~pdollar/research/papers/BabenkoEtAlECCV08simul.pdf)

\[2\] [Babenko, B.; Ming-Hsuan Yang; Belongie, S., "Robust Object Tracking 
with Online Multiple Instance Learning," in Pattern Analysis and Machine 
Intelligence, IEEE Transactions on , vol.33, no.8, pp.1619-1632, Aug. 2011
doi: 10.1109/TPAMI.2010.226](http://vision.ucsd.edu/~bbabenko/data/miltrack-pami-final.pdf)

\[3\] [Friedman, Jerome, Hastie, Trevor, Tibshirani, Robert & others (2000). 
Additive logistic regression: a statistical view of 
boosting (with discussion and a rejoinder by the authors). 
The annals of statistics, 28, 337-407.](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)

\[4\] [Paul Viola, John C. Platt, and Cha Zhang. Multiple instance boosting
for object detection. In In NIPS 18, pages 1419–1426. MIT Press, 2006.](http://vision.ucsd.edu/~bbabenko/data/miltrack-pami-final.pdf)

\[5\] Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml). 
Irvine, CA: University of California, School of Information and Computer Science.

### Other references

\[6\] C.M. Bishop. Pattern recognition and machine learning. Information
science and statistics. Springer, 2006.

\[7\] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. 
Cambridge University Press, March 2004.

\[8\] Thomas G. Dietterich and Richard H. Lathrop. Solving the multiple-
instance problem with axis-parallel rectangles. Artificial Intelligence,
89:31–71, 1997.

\[9\] Yoav Freund and Robert E. Schapire. A short introduction to boosting,
1999.

\[10\] Jerome H. Friedman. Stochastic gradient boosting. Computational
Statistics and Data Analysis, 38:367–378, 1999.

\[11\] Jerome H. Friedman. Greedy function approximation: A gradient
boosting machine. Annals of Statistics, 29:1189–1232, 2000.

\[12\] James D. Keeler, David E. Rumelhart, and Wee Kheng Leow. Integrated 
segmentation and recognition of hand-printed numerals. In
NIPS’90, pages 557–563, 1990.

\[13\] Llew Mason, Jonathan Baxter, Peter Bartlett, and Marcus Frean.
Boosting algorithms as gradient descent in function space, 1999.

\[14\] William H. Press, Saul A. Teukolsky, William T. Vetterling, and
Brian P. Flannery. Numerical Recipes 3rd Edition: The Art of Sci-
entific Computing. Cambridge University Press, New York, NY, USA,
3 edition, 2007.

\[15\] Vladimir N. Vapnik. The nature of statistical learning theory. Springer-
Verlag New York, Inc., New York, NY, USA, 1995.

\[16\] Paul Viola and Michael Jones. Robust real-time object detection. 
International Journal of Computer Vision, 57(2):137–154, 2002.



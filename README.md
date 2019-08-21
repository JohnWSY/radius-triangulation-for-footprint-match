# radius-triangulation-for-footprint-match
a method to judge whether two footprints are same source or not

1.json2csv is based on a algorithm which label the features, we can know the coordinate, type and direction of the feature from csv

2.radius triangulation is made on the foundation of evidence above, then we can get type, direction, polyline, radius length and area of triangle 

3.one footprint with one feature vector, we calc the Euclid distance between two feature vector(other ways to calc distance are positive to try to see whether better or not)

4.count the distance we get, they generate two distribution, one is from same source, the other is from different source footprint. Besides they are seperated obviously

5.use LR to make Tippett plot, we can get rate of misleading of the evidence. Furthermore, I will coninue to calc the Likelihood of picture of each footprint feature to make another scoring system

6.then I will fuse two scoring sysytem with two-dimensional Gaussion distribution. At last, the programme can come into use.

7.with a score of two footprints, we can give a conclusion whether they are same source or not.

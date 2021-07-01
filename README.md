# ExKmeans

cubedata.py generates our lower instance

embed.py contains the terminal embedding 

ExTree3.py generates the threshold tree

ExKmeans.py contains the function ExKmeans(X, k, C, numr)

- X: data set

- k: number of clusters

- C: k centers (If not provide, it will run a kmeans++ to compute a center set)

- numr: number of times to repeat the explainable tree (default = 10)

cube_exkmeans: compare ExKMC algorithm with our algorithm on the cube instance

- To run the ExKMC, you need to install the 'ExKMC'. I could not install ExKMC under Python 3.9, but I can install it under Python 3.8. (I think any version before 3.8 is Ok.) 



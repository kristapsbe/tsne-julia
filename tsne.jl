# the implementation is based on http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

# Just setting up a test array that looks like
# [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]]
# by using julia list comprehensions and the shorthand range function 
test_data = [[v for v in 0+i:5+i] for i in 0:4]

# calculates the euclidean distance between two lists of the same length
function euclidean(p,q)
  # remember that the lists in julia are 1 based
  return sqrt(sum([(p[i]-q[i])^2 for i in 1:length(p)]))
end

# "Stochastic Neighbor Embedding (SNE) starts by converting the high-dimensional Euclidean distances between data points..."
#
# first thing's first => getting the euclidean distances first
#
# !! TODO: don't need to re-calc everything twice -> split it on the diagonal !!
distances = [[euclidean(test_data[i], test_data[j]) for i in 1:length(test_data)] for j in 1:length(test_data)]

print(distances)

# ?
function sigma(i)
  return 0
end

# p i|j
#
# !! TODO: figure out what x is supposed to be at this point? !!
function similarity(i,j,x)
  return exp((-abs(x[i]-x[j])^2)/(2*sigma(i)^2))/sum([exp((-abs(x[i]-x[k])^2)/(2*sigma(i)^2)) for k in 1:length(x) if k!=i])
end

# q i|j
#
# !! TODO: figure out what y is supposed to be at this point? !!
function conditional(i,j,y)
  return exp(-abs(y[i]-y[j])^2)/sum([exp(-abs(y[i]-y[k])^2) for k in 1:length(y) if k!=i])
end

# !! TODO: same thing as for the 2 before it -> don't quite know what the input parameters are supposed to be !!
function cost(x,y)
  return sum([sum([similarity(i,j,x)*log(similarity(i,j,x)/conditional(i,j,y)) for j in 1:length(x)]) for i in 1:length(x)])
end

function perplexity(i,x)
  return 2^shannon(i,x)
end

# I should really pre-prep the matrices to avoid re-calculating stuff constantly
function shannon(i,x)
  return -sum([similarity(i,j,x)*log2(similarity(i,j,x)) for j in 1:length(x)])
end

function gradient(i,x,y)
  return 2*sum([(similarity(j,i,x)-conditional(j,i,x)+similarity(i,j,x)-conditional(i,j,y))*(y[i]-y[j]) for j in 1:length(x)])
end

function descent()
  return 0
end
# the implementation is based on http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

# Just setting up a test array that looks like
# [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]]
# by using julia list comprehensions and the shorthand range function 
test_data = [[v for v in 0+i:5+i] for i in 0:4]

# calculates the euclidean distance between two lists of the same length
function euclidean(p, q)
  # remember that the lists in julia are 1 based
  return sqrt(sum([(p[i]-q[i])^2 for i in 1:length(p)]))
end

# "Stochastic Neighbor Embedding (SNE) starts by converting the high-dimensional Euclidean distances between data points..."
#
# first thing's first => getting the euclidean distances first
distances = [[euclidean(test_data[i], test_data[j]) for i in 1:length(test_data)] for j in 1:length(test_data)]

print(distances)

# "...into conditional probabilities that represent similarities"
#
# ?
function similarity(p, q)
  # σi? 
  return 0
end

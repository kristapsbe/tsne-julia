# TODO: i cannot for the life of me remember which package was i supposed to add for MLDatasets
import Pkg; Pkg.add("Distributions"); Pkg.add("Plots"); Pkg.add("MLDatasets")

using Plots
using Random, Distributions
using MLDatasets

# the implementation is based on http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
# and https://raw.githubusercontent.com/danaugrs/go-tsne/master/tsne/tsne.go was used to understand some of the math and get some guidance on the general layout

# setting up constants that we'll need
epsilon = 10^-7
fake_zero = 10^-12 # can't really have 0 for some of the calculations - making a mumber really close to it
tol = 10^-5 # enthropy tolerance
max_binary = 50 # max number of steps that the binary search gets to do
init_stdev = 10^-4 # the stdev that we're initialising Y with

# and init parameters that we'll be working with
n = 50 # the number of entries that we're working with
m = 5 # the number of dimensions
m_out = 2 # the number of dimensions we want to get as an output
T = 1000 # the number of iterations

perp = 5 # in a sense, a guess about the number of close neighbors each point has
l_rate = 100 # the learning rate

# creating our dummy dataset that'll look like
# [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10]]
X = [[v for v in i:m+i] for i in 1:n]
#println("Loading MNIST")
#X = MNIST.convert2features(MNIST.traintensor())
#m, n = size(X)
#n = 2000 # limit the size for testing
colmap = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
labels = MNIST.trainlabels() # need the labels to set up the colors
cols = [labels[i]+1 for i in 1:n] #MNIST.trainlabels()[keys] # getting the corresponding lables so that we can color the scatterplot
#X = [X[1+(m*(i-1)):(m*(i-1))+m] for i in 1:n] # this is probably inneficient as all hell, but makes thinking a lot easier for me
println("MNIST loaded")

# calculates the squared euclidean distance between two lists of the same length
function squared_euclidean(p, q, m)
  # remember that the lists in julia are 1 based
  # sicne the distance is a square root of the sum of squares we can just omit the sqrt 
  # (since we're interested in the squared distance)
  return sum([(p[i]-q[i])^2 for i in 1:m])
end


# calculates a squared distance matrix for row vectors in X
function squared_distances(X, n, m) 
  dists = [[Float64(0) for i in 1:n] for j in 1:n] 
  for i in 1:n
    for j in (i+1):n # skip the diagonal as well
      tmp_dist = squared_euclidean(X[i], X[j], m)
      dists[i][j] = tmp_dist
      dists[j][i] = tmp_dist
    end
  end
  return dists
end


function similarities(D, m, n, tol, perp, epsilon)
	# the target entropy of the gaussian kernels is the log of the target perplexity
	h_target = log(perp)

  # the similarity matrix that we're going to populate
  P = [[Float64(0) for i in 1:n] for j in 1:n]

  # work through all of the rows
  for i in 1:n
		# perform a binary search for the Gaussian kernel precision (beta)
		# such that the entropy (and thus the perplexity) of the distribution
		# given the data is the same for all data points
		beta_min = -Inf
		beta_max = Inf
		beta = 1 # initial value of precision

    for j in 1:max_binary
      # compute raw probabilities with beta precision (along with sum of all raw probabilities)
      p_sum = Float64(0)
      for k in 1:m
        p = 0
        if i != k
          p = exp(-D[i][k]*beta)
        end

        P[i][k] = p
        p_sum += p
      end
      
      # normalize probabilities and compute entropy H
      h = Float64(0) # distribution entropy
      for k in 1:m
        p = 0
        if p_sum != 0
          p = P[i][k]/p_sum
        end

        P[i][k] = p
        if p > epsilon
          h -= p*log(p)
        end
      end

			# adjust beta to move H closer to Htarget
      h_diff = h-h_target
      if h_diff > 0
        # entropy is too high (distribution too spread-out)
        # so we need to increase the precision
        beta_min = beta # move up the bounds
        if beta_max == Inf
          beta *= 2
        else
          beta = (beta+beta_max)/2
        end
      else
				# entropy is too low - need to decrease precision
        beta_max = beta # move down the bounds
        if beta_min == -Inf
          beta /= 2
        else
          beta = (beta+beta_min)/2
        end
      end

			# if current entropy is within specified tolerance - we are done with this data point
			if abs(h_diff) < tol
				break
      end
    end
  end
  
  return P
end


function cost_gradient(P, Y, n, m_out, P_log_P, dCdY, fake_zero)
  Y_dist = squared_distances(Y, n, m_out)
  Y_dist = [[((i != j) ? 1/(1+Y_dist[i][j]) : 0) for j in 1:n] for i in 1:n] # clear out the diagonal and replace values with the student t-distribution
  # normalize the distance matrix
  sum_q = sum(sum(Y_dist))
  Q = [[max(Y_dist[i][j]/sum_q, fake_zero) for j in 1:n] for i in 1:n]
  # the non-constant portion of the divergence
  Q_log = [[log(Y_dist[i][j]) for j in 1:n] for i in 1:n]
  # compute the divergence
  P_log_Q = sum(sum([[P[i][j]*Q_log[i][j] for j in 1:n] for i in 1:n]))
  divergence = P_log_P-P_log_Q
  # compute the matrix of scalar multiple for scaling the 3d difference matrix prior to squashing
  mult = P.-Q
  mult = mult./4
  mult = [[mult[i][j]*Y_dist[i][j] for j in 1:n] for i in 1:n]
  # compute gradient
  for r in 1:n
    for c in 1:n
      m = mult[r][c]
      for k in 1:m_out
        y_diff = Y[r][k]-Y[c][k]
        orig = dCdY[r][k]
        dCdY[r][k] = orig+m*y_diff
      end
    end
  end

  return divergence, dCdY
end


println("Prepping")
# compute pairwise affinities p i|j with perplexity Perp (using Equation 1)
D = squared_distances(X, n, m)
println("Distances calculated")
P = similarities(D, m, n, tol, perp, epsilon)
println("Similarities calculated")
# set p ij = (p j|i + p i|j)/2n
P = [[max((P[j][i]+P[i][j])/(2*n), fake_zero) for i in 1:n] for j in 1:n]
println("Similarities adjusted")
# sample initial solution Y(0)={y1, y2, ..., yn} for N(0,10^-4*I)
init_vals = rand(Normal(0, init_stdev), m_out*n)
Y = [[init_vals[m_out*i-j+1] for j in 1:m_out] for i in 1:n]
println("Y initialised")
# set up the gradien matrix
dCdY = [[Float64(0) for j in 1:m_out] for i in 1:n]
# compute and store the constant portion of the KL divergence
P_log_P = sum(sum([[log(P[i][j])*P[i][j] for i in 1:n] for j in 1:n]))
# for t=1 to T do
#
#   where T is the number of epochs that we're looking to work through
println("Training")
for t in 1:T
  global dCdY, P, Y, n, m_out, P_log_P, fake_zero, cols
  # compute KL divergence and update the gradient matrix
  divergence, dCdY = cost_gradient(P, Y, n, m_out, P_log_P, dCdY, fake_zero)
  # step in the direction of negative gradient (times the learning rate)
  Y = dCdY.*l_rate
  # reproject Y to have zero mean
  y_mean = [sum([Y[j][i] for j in 1:n]) for i in 1:m_out]
  Y = [[Y[i][j]-y_mean[j] for j in 1:m_out] for i in 1:n]

  println(t)
  if t%10 == 0 # keeping myself from getting swamped in charts
    xs = [[Y[i][1]] for i in 1:n]
    ys = [[Y[i][2]] for i in 1:n]
    
    p = scatter(xs[cols .== 1], ys[cols .== 1], markercolor=colmap[1])
    for k = 2:length(colmap)
        scatter!(xs[cols .== k], ys[cols .== k], markercolor=colmap[k])
    end
    display(p)
  end
end

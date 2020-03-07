import Pkg; Pkg.add("Distributions"); Pkg.add("Plots"); Pkg.add("MLDatasets"); Pkg.add("CSV")

using CSV
using Plots
using Random, Distributions
using MLDatasets

# the implementation is based on http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
# and uses https://raw.githubusercontent.com/danaugrs/go-tsne/master/tsne/tsne.go as a basis
# with input params borrowed from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# setting up constants that we'll need
epsilon = 10^-7 # minimum gradient norm (if smaller optimization is stopped)
fake_zero = 10^-12 # can't really have 0 for some of the calculations - making a mumber really close to it
tol = 10^-5 # enthropy tolerance
max_binary = 50 # max number of steps that the binary search gets to do
init_stdev = 10^-4 # the stdev that we're initialising Y with

# and init parameters that we'll be working with
m_out = 2 # the number of dimensions we want to get as an output
T = 1000 # the number of iterations

perp = 50 # in a sense, a guess about the number of close neighbors each point has
l_rate = 300 # the learning rate

# loading our dataset
csv_data = CSV.read("tsne-julia/abalone.csv", header=false)
X = convert(Matrix, csv_data[:, 2:9])
n, m = size(X)
cols = convert(Matrix, csv_data[:, 1:1])
unique_cols = unique(cols)
colmap = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]

# calculates the squared euclidean distance between two lists of the same length
function squared_euclidean(p, q, m)
  # remember that the lists in julia are 1 based
  # sicne the distance is a square root of the sum of squares we can just omit the sqrt 
  # (since we're interested in the squared distance)
  return sum([(p[i]-q[i])^2 for i in 1:m])
end


# calculates a squared distance matrix for row vectors in X
function squared_distances(X, n, m) 
  dists = zeros(n, n)
  for i in 1:n
    for j in (i+1):n # skip the diagonal as well
      tmp_dist = squared_euclidean(X[i,:], X[j,:], m) # and reuse the value for the pair
      dists[i, j] = tmp_dist
      dists[j, i] = tmp_dist
    end
  end
  return dists
end


function similarities(D, m, n, tol, perp, epsilon)
  # the target entropy of the gaussian kernels is the log of the target perplexity
  h_target = log(perp)

  # the similarity matrix that we're going to populate
  P = zeros(n, n)

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
          p = exp(-D[i, k]*beta)
        end

        P[i, k] = p # note that we're not updating the whole thing (can't really do shorthand as a result)
        p_sum += p
      end
      
      # normalize probabilities and compute entropy H
      h = Float64(0) # distribution entropy
      for k in 1:m
        p = 0
        if p_sum != 0
          p = P[i, k]/p_sum
        end

        P[i, k] = p
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


function cost_gradient(P, Y, dCdY, n, m_out, P_log_P, fake_zero)
  # q i|j (1+dist^2)^-1 / sum(i≠j, (1+dist^2)^-1)
  Y_dist = squared_distances(Y, n, m_out)
  # set up the top part of that calculation
  Q_top = zeros(n, n)
  for i in 1:n
    for j in 1:n
      if i!=j
        Q_top[i, j] = 1/(1+Y_dist[i, j])
      end
    end
  end
  # get the bottom bit of that function
  sum_q = sum(sum(Q_top))
  # build up the q i|j set
  Q = zeros(n, n)
  for i in 1:n
    for j in 1:n
      Q[i, j] = max(Q_top[i, j]/sum_q, fake_zero)
    end
  end
  # the non-constant portion of the divergence
  Q_log = zeros(n, n)
  for i in 1:n
    for j in 1:n
      Q_log[i, j] = log(Q[i, j])
    end
  end
  # compute the Kullback-Leiber divergence
  P_log_Q = sum(sum(P*Q_log))
  # work out the gradient of the Kullback-Leibler divergence between P and the Student-t based joint probability distribution Q
  mult = P-Q
  mult = mult.*4
  mult = mult*Y_dist
  for r in 1:n
    for c in 1:n
      m = mult[r, c]
      for k in 1:m_out
        y_diff = Y[r, k] - Y[c, k]
        dCdY[r, k] += m*y_diff
      end
    end
  end
  return dCdY
  #return [4*sum([(P[i][j]-Q[i][j])*(Y[i]-Y[j])/(1+Y_dist[i][j]) for j in 1:n]) for i in 1:n]
end


println("Prepping")
# compute pairwise affinities p i|j with perplexity Perp (using Equation 1)
D = squared_distances(X, n, m)
println("Distances calculated")
P = similarities(D, m, n, tol, perp, epsilon)
println("Similarities calculated")
# set p ij = (p j|i + p i|j)/2n
P = zeros(n, n)
for i in 1:n
  for j in 1:m_out
    P[i, j] = max((P[j, i]+P[i, j])/(2*n), fake_zero)
  end
end
println("Similarities adjusted")
# sample initial solution Y(0)={y1, y2, ..., yn} for N(0,10^-4*I)
init_vals = rand(Normal(0, init_stdev), m_out*n)
Y = zeros(n, m_out)
for i in 1:n
  for j in 1:m_out
    Y[i, j] = init_vals[m_out*i-j+1]
  end
end
dCdY = zeros(n, m_out)
println("Y initialised")
# compute and store the constant portion of the KL divergence
P_log_P = zeros(n, n)
for i in 1:n
  for j in 1:n
    P_log_P[i, j] = log(P[i, j])*P[i, j] 
  end
end
P_log_P = sum(sum(P_log_P))
# for t=1 to T do
#
#   where T is the number of epochs that we're looking to work through
println("Training")
for t in 1:T
  global P, Y, n, m_out, P_log_P, fake_zero, cols, momentum, dCdY
  # compute the gradient matrix
  dCdY = cost_gradient(P, Y, dCdY, n, m_out, P_log_P, fake_zero)
  # apply scaled gradient
  println(Y[1:5, :])
  Y -= dCdY.*l_rate
  # reproject Y to have zero mean
  Y_mean = zeros(m_out)
  for i in 1:n
    for j in 1:m_out
      Y_mean[j] += Y[i, j]
    end
  end

  for i in 1:n
    for j in 1:m_out
      Y[i, j] -= (Y_mean[j]/n)
    end
  end
  println(Y[1:5, :])
  println(t)
  if t%10 == 0 # keeping myself from getting swamped in charts
    indices = [i for i in 1:n if cols[i]==unique_cols[1]]
    p = scatter(Y[:, 1][indices], Y[:, 2][indices], markercolor=colmap[1])
    for k = 2:length(unique_cols)
      indices = [i for i in 1:n if cols[i]==unique_cols[k]]
      scatter!(Y[:, 1][indices], Y[:, 2][indices], markercolor=colmap[k])
    end
    display(p)
  end
end

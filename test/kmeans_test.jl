using Base.Test
using Clustering

include("common.jl")


num_samples = 100
num_features = 50
num_centers = 10 #the 'k' in k-means

X, y = generate_data_blobs(;n_samples=num_samples, n_features=num_features, n_centers=num_centers, center_box = (-50.0, 50.0))

orig_vectors, orig_assignments = randomize_data(X, y; randomize=false)
random_vectors, random_assigments = randomize_data(X, y; randomize=true)

#non-weighted
r = kmeans(random_vectors, num_centers; maxiter=100)
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (num_samples, num_centers)
@test length(r.assignments) == num_samples
@test all(r.assignments .>= 1) && all(r.assignments .<= num_centers)
@test_approx_eq all(r.assignments == orig_assignments)

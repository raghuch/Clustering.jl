using Base.Test
using Clustering

include("common.jl")


num_samples = 20
num_features = 3
num_centers = 5 #the 'k' in k-means

X, y = generate_data_blobs(;n_samples=num_samples, n_features=num_features, n_centers=num_centers, center_box = (-50.0, 50.0), cluster_std = 9.0)

orig_vectors, orig_assignments = randomize_data(X, y; randomize=false)
random_vectors, random_assigments = randomize_data(X, y; randomize=true)

#non-weighted
r = kmeans(random_vectors, num_centers; maxiter=100)
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (num_features, num_centers)
@test length(r.assignments) == num_samples
@test all(r.assignments .>= 1) && all(r.assignments .<= num_centers)
#@test length(r.costs) = num_samples
#@test_approx_eq r.centers 
#@test all(r.assignments == orig_assignments)

for i in 1:num_samples
    #if r.assignments[i] != orig_assignments[i]
        #println("For sample", i, " cluster assignment:", r.assignments[i], " original assignment:", orig_assignments[i])
        println(i, " cluster#:", r.assignments[i])

    #end
end




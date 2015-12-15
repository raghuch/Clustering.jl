#Common Utils for testing Clustering

#Generate Sample Data (already clustered with some centers)

function generate_clustered_data(;seed::Float64=0.0, n_clusters::Int64=3, n_features::Int64=2,
                                 n_samples_per_cluster::Int64=20, std::Float64=0.4)

    prng = randn(n_samples_per_cluster, n_features)

    offset = 10   #Means don't have to be centered at the origin

    #We start with means with an offset and form clusters around them.
    means = [1 1 1 0;
             -1 -1 0 1;
             1 -1 1 1;
             -1 1 1 0] + offset

    A = zeros(0, n_features)

    for i in 1:n_clusters
        A = vcat(A, means[i, 1:n_features].+(std*prng))
    end

    return A
end

#@doc"""
#takes in an argument ``random_state`` and returns a random number generator type.
#``random_state`` can be an integer, or an instance of the type ``AbstractRNG``, or 
#a symbol ``:None``.
#""" ->
function check_random_state(random_state)
#To add: ability to specify size of the vector/matrix of random numbers.

    if typeof(random_state) <: AbstractRNG
        return random_state
    elseif typeof(random_state) <: Int
        return MersenneTwister(random_state)
    elseif random_state == :None
        return RandomDevice()
    else
        return error("random_state should be an integer or the symbol ':None' or an rng...")
    end
end

#@doc"""
#Generate a normal random distribution centered at ``loc`` and standard deviation ``scale``
#""" ->
#function randn(rng::AbstractRNG; loc=0.0, scale=1.0)
#    return loc + (scale * randn(rng))
#end

#function randn(rng::AbstractRNG; loc = 0.0, scale=1.0 )
#end


function generate_data_blobs(;n_samples::Int64=100, n_features::Int64=5, n_centers::Int64=5, cluster_std::Float64=1.0, center_box = (-10.0, 10.0), shuffle::Bool=true, random_state=:None )

    #X = zeros(n_samples, n_features)
    #y = zeros(n_samples)
    X = []
    y = []

    rng = check_random_state(random_state)
    cluster_centers = rand(rng, -center_box[1]:center_box[2], (n_centers, n_features))

    samples_per_center = round(Int, n_samples/n_centers)
    #sample_size_array =  ones(samples_per_center) * n_centers
    sample_size_array = [n_centers for i in samples_per_center]

    for i in 1:(n_samples % n_centers)
        sample_size_array[i] += 1
    end
    
    #We assume the std deviation of all clusters is the same and passed as a parameter. We form
    # a vector of of size "number of clusters", with std devation of each cluster
    std = ones(size(cluster_centers, 1)) * cluster_std

    for (i, n) in enumerate(sample_size_array)
        curr_std = std[i]
        push!(X, cluster_centers[i, :] .+ (curr_std* randn( n, n_features) ))
        push!(y, [i for el in n])
    end

    return X, y
end

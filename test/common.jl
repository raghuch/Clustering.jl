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

function generate_data_blobs(;n_samples::Int64=100, n_features::Int64=7, n_centers::Int64=5, cluster_std::Float64=1.0, center_box = (-10.0, 10.0), random_state=:None )

    rng = check_random_state(random_state)
    cluster_centers = rand(rng, -center_box[1]:center_box[2], (n_centers, n_features))

    samples_per_center = round(Int, n_samples/n_centers)
    sample_size_array = [(1*samples_per_center) for i in 1:n_centers]

    for i in 1:(n_samples % n_centers)
        sample_size_array[i] += 1
    end

    X = Array{Array{Float64, 1}}(n_samples)
    y = Array{Int64}(n_samples)
    
    #We assume the std deviation of all clusters is the same and passed as a parameter. We form
    # a vector of of size "number of clusters", with std devation of each cluster
    std = ones(size(cluster_centers, 1)) * cluster_std

    curr_sample_num = 0
    for (i, n) in enumerate(sample_size_array)
        curr_std = std[i]
        for j in 1:n
            X[curr_sample_num + j] = vec(cluster_centers[i, :]) + (curr_std * randn(n_features))
            y[curr_sample_num+j] = i
        end
        curr_sample_num += n
    end

    #for i in 1:length(X)
    #    dataset[i, :] = X[i]
    #end

    return X, y
end

function randomize_data(X::AbstractArray, y::AbstractVector; randomize::Bool=true)

    a = collect(1:length(y))
    random_order = shuffle!(a)
    dataset = zeros(length(X), length(X[1]) )

    if randomize==true
        y = y[random_order]
        X = X[random_order]
    end
    
    for i in 1:length(X)
        dataset[i, :] = X[i]
    end

    return dataset, y
end


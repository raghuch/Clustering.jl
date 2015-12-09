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


function generate_data_blobs(;n_samples::Int64=100, n_features::Int64=5, centers::Int64=5, clustered_std::Float64=1.0, center_box = (-10.0, 10.0), shuffle::Bool=true, random_state=0 )

    rng = MersenneTwister(random_state)
    cluster_centers = rand(rng, (centers, n_features))
    
    cluster_std = ones(size(cluster_centers, 1)) * cluster_std


end

"""
@description
An implementation of rough k-means clustering for multi-dimensional
numerical features. This extends conventional
k-means to rough set theory by inclusion of upper and lower
approximations in both entity-cluster distance measures and cluster
centroid computations.

See: Lingras and Peter, Applying Rough Set Concepts to Clustering, in
G. Peters et al. (eds.), Rough Sets: Selected Methods and Applications
in Management and Engineering, Advanced Information and Knowledge
Processing, DOI 10.1007/978-1-4471-2760-4_2, Springer-Verlag London
Limited, 2012.

@options
    self.max_clusters = max_clusters # Number of clusters to return
    self.wght_lower = wght_lower     # Rel. weight of lower approxs
    self.wght_upper = wght_upper     # Rel. weight of upper approxs
    self.dist_threshold = None       # Threshold for cluster similarity
    self.p_param = p_param           # parameter for weighted distance
                                       centroid option below
    self.weighted_distance = wght    # Option (True) to use weighted
                                       distance centroid calculations

@notes
Distance threshold option:
    self.dist_threshold = 1.25 by default (entity assigned to all
    centroids within 25% of the optimal cluster distance)
    if self.dist_threshold <=1.0 conventional kmeans clusters returned

    The larger self.dist_threshold the more rough (entity overlap) will
    exist across all k clusters returned

Lower and Upper Approximation Weight options:
    SUM(wght_lower,wght_upper) must equal 1.0, else it will be set to
    defaults on execution

    wght_lower=0.75 by default
    wght_upper=0.25 by default

    The larger wght_lower is relative to wght_upper the more important
    cluster lower approximations will be and v.v

@author Michael Tompkins
@copyright 2016
"""

# Externals
import warnings
import time
import operator
import numpy as np
from copy import deepcopy


class RoughKMeans:

    def __init__(self,input_data,
                 max_clusters,
                 wght_lower=0.75,
                 wght_upper=0.25,
                 threshold=1.25,
                 p_param=1.0,
                 wght=False):

        # Rough clustering options
        self.normalize = False            # Option to Z-score normalize features
        self.max_clusters = max_clusters    # Number of clusters to return
        self.dist_threshold = threshold     # <=1.0 threshold for centroids indiscernibility
        self.tolerance = 1.0e-04            # Tolerance for stopping iterative clustering
        self.previous_error = 1.0e+32       # Back storage of centroid error
        self.wght_lower = wght_lower        # Rel. weight of lower approx for each cluster centroid
        self.wght_upper = wght_upper        # Rel. weight of upper approx to each cluster centroid
        self.p_param = p_param              # parameter for weighted distance centroid option
        self.weighted_distance = wght       # Option (True) to use alt. weighted distance centroid

        # Enforce wght_lower + wght_upper == 1.0
        if self.wght_lower + self.wght_upper > 1.0:
            self.wght_lower = 0.75
            self.wght_lower = 0.25
            warnings.warn("Upper + Lower Weights must == 1.0, Setting Values to Default")

        # Rough clustering internal vars
        self.data = input_data
        self.data_array = None
        self.feature_names = input_data.keys()
        self.data_length = len(self.data[list(self.feature_names)[0]])

        # Rough clustering external vars
        self.keylist = None                 # Ordered list of keys
        self.tableau_lists = None           # List order of data keys for centroid arrays
        self.centroids = {}                 # Centroids for all returned clusters
        self.cluster_list = {}              # Internal listing of membership for all clusters
        self.distance = {}                  # Entity-cluster distances for all candidate clusters
        self.clusters = None                # upper and lower approx membership for all clusters
        self.d_weights = {}                 # Weight func. for entities if weighted_distance = True

        # Overhead
        self.timing = True                  # Timing print statements flag
        self.debug = False                  # Debug flag for entire class print statements
        self.debug_assign = False           # Debug flag assign_cluster_upper_lower_approximation()
        self.debug_dist = False             # Debug flag get_entity_centroid_distances()
        self.debug_update = False           # Debug flag update_centroids()
        self.small = 1.0e-04
        self.large = 1.0e+10

    def get_rough_clusters(self):

        """
        Run iterative clustering solver for rough k-means and return
        max_cluster rough clusters

        :return: self.centroids, self.assignments, self.upper_approximation,
        self.lower_approximation
        """

        # Transform data to nd-array for speed acceleration
        self.transform_data()

        # Get initial random entity clusters
        self.initialize_centroids()

        if self.dist_threshold <= 1.0:
            warnings.warn("Rough distance threshold set <= 1.0 and will produce conventional \
            k-means solution")

        # Iterate until centroids convergence
        ct = 0
        stop_flag = False
        while stop_flag is False:

            t1 = time.time()
            # Back-store centroids
            prev_centroids = deepcopy(self.centroids)

            # Get entity-cluster distances
            self.get_entity_centroid_distances()

            # Compute upper and lower approximations
            self.assign_cluster_upper_lower_approximation()

            # Update centroids with upper and lower approximations
            if self.weighted_distance is True:        # Run entity-centroid weighted distance update
                self.update_centroids_weighted_distance()
            else:   # Run standard rough k-means centroid update
                self.update_centroids()

            # Determine if convergence reached
            stop_flag = self.get_centroid_convergence(prev_centroids)

            t2 = time.time()
            iter_time = t2-t1
            if self.debug_update:
                print("Clustering Iteration", ct, " in: ", iter_time," secs")
            ct += 1

        return

    def transform_data(self):

        """
        Convert input data dictionary to float nd-array for
        accelerated clustering speed

        :var self.data
        :return: self.data_array
        """

        t1 = time.time()
        self.keylist = self.data.keys()
        self.tableau_lists = [self.data[key][:] for key in self.data]
        self.data_array = np.asfarray(self.tableau_lists).T

        # Normalize if requested
        if self.normalize is True:
            self.data_array -= np.mean(self.data_array, axis=0)
            tmp_std = np.std(self.data_array, axis=0)
            for i in range(len(self.data_array[0, :])):
                if tmp_std[i] >= 0.001:
                    self.data_array[:, i] /= tmp_std[i]

        if self.timing is True:
            t3 = time.time()
            print("transform_data Time",t3-t1)
            print("shape",self.data_array.shape)

    def initialize_centroids(self):

        """
        Randomly select [self.max_clusters] initial entities as
        centroids and assign to self.centroids

        :var self.max_clusters
        :var self.data
        :var self.data_array
        :var self.feature_names
        :return: self.centroids : current cluster centroids
        """

        t1 = time.time()

        # Select max cluster random entities from input and assign as
        # initial cluster centroids
        candidates = np.random.permutation(self.data_length)[0:self.max_clusters]

        if self.debug is True:
            print("Candidates",candidates,self.feature_names,self.data)

        # self.centroids = {str(k): {v: self.data[v][candidates[k]] for v in self.feature_names} for
        #                  k in range(self.max_clusters)}

        self.centroids = {str(k): self.data_array[candidates[k], :] for k in
                          range(self.max_clusters)}

        if self.timing is True:
            print('Max Clusters',self.max_clusters)
            t3 = time.time()
            print("initialize_centroids Time",t3-t1)

        return

    def get_centroid_convergence(self,previous_centroids):

        """
        Convergence test. Determine if centroids have changed, if so, return False, else True

        :arg previous_centroids : back stored values for last iterate centroids
        :var self.centroids
        :var self.feature_names
        :var self.tolerance
        :return boolean : centroid_error <= self.tolerance (True) else (false)
        """

        t1 = time.time()

        # centroid_error = np.sum([[abs(self.centroids[k][val] - previous_centroids[k][val])
        #                          for k in self.centroids] for val in self.feature_names])

        centroid_error = np.sum([np.linalg.norm(self.centroids[k] - previous_centroids[k])
                                 for k in self.centroids])

        if self.timing is True:
            t3 = time.time()
            print("get_centroid_convergence Time",t3-t1, " with error:",centroid_error)

        if self.debug is True:
            print("Centroid change", centroid_error)

        if centroid_error <= self.tolerance or np.abs(self.previous_error - centroid_error) \
                <= self.tolerance:
            return True
        else:
            self.previous_error = centroid_error.copy()
            return False

    def update_centroids_weighted_distance(self):

        """
        Update rough centroids for all candidate clusters given their
        upper/lower approximations set membership plus a weighted
        distance function based on distance of each entity to the given
        cluster centroid

        Cluster centroids updated/modified for three cases:
            if sets {lower approx} == {upper approx}, return
                conventional k-means cluster centroids
            elif set {lower approx] is empty and set {upper approx}
                is not empty, return upper-lower centroids
            else return weighted mean of lower approx centroids and
                upper-lower centroids

        :var self.data_array
        :var self.wght_lower
        :var self.wght_upper
        :var self.feature_names
        :var self.clusters
        :var self.d_weights
        :return: self.centroids : updated cluster centroids
        """

        t1 = time.time()

        for k in self.clusters:

            if len(self.clusters[k]["lower"]) == len(self.clusters[k]["upper"]) and \
                            len(self.clusters[k]["lower"]) != 0:
                # Get lower approximation vectors and distance weights
                weights = np.asarray([self.d_weights[k][str(l)] for l in self.clusters[k]["lower"]])
                weights /= np.sum(weights)
                self.centroids[str(k)] = \
                    np.sum([weights[m] * self.data_array[l,:]
                            for m,l in enumerate(self.clusters[k]["lower"])], axis=0)

            elif len(self.clusters[k]["lower"]) == 0 and len(self.clusters[k]["upper"]) != 0:
                # Get upper approximation vectors
                weights = np.asarray(
                    [self.d_weights[k][str(l)] for l in self.clusters[k]["upper"]])
                weights /= np.sum(weights)
                self.centroids[str(k)] = \
                    np.sum([weights[m] * self.data_array[l, :]
                            for m,l in enumerate(self.clusters[k]["upper"])], axis=0)

            else:
                # Get both upper-exclusive and lower approximation sets
                exclusive_set = \
                    list(set(self.clusters[k]["upper"]).difference(set(self.clusters[k]["lower"])))
                weights1 = np.asarray(
                    [self.d_weights[k][str(l)] * self.data_array[l, :]
                     for l in self.clusters[k]["lower"]])
                weights1 /= np.sum(weights1)
                weights2 = np.asarray(
                    [self.d_weights[k][str(l)] * self.data_array[l, :] for l in exclusive_set])
                weights2 /= np.sum(weights2)
                self.centroids[str(k)] = \
                    self.wght_lower * np.sum([weights1[m] * self.data_array[l, :]
                                              for m,l in enumerate(self.clusters[k]["lower"])], axis=0) \
                    + self.wght_upper * np.sum([weights2[m] * self.data_array[l, :]
                                                for m,l in enumerate(exclusive_set)], axis=0)

            if self.debug_update is True:
                print("""###Cluster""", k, self.clusters[k]["lower"], self.clusters[k]["upper"])

        if self.timing is True:
            t3 = time.time()
            print("update_centroids Time", t3 - t1)

        return

    def update_centroids(self):

        """
        Update rough centroids for all candidate clusters given their
        upper/lower approximations set membership

        Cluster centroids updated/modified for three cases:
            if sets {lower approx} == {upper approx}, return
                conventional k-means cluster centroids
            elif set {lower approx] is empty and set {upper approx}
                is not empty, return upper-lower centroids
            else return weighted mean of lower approx centroids and
                upper-lower centroids

        :var self.data_array
        :var self.wght_lower
        :var self.wght_upper
        :var self.feature_names
        :var self.clusters
        :return: self.centroids : updated cluster centroids
        """

        t1 = time.time()

        for k in self.clusters:

            if len(self.clusters[k]["lower"]) == len(self.clusters[k]["upper"]):
                # Get lower approximation vectors
                lower = self.data_array[self.clusters[k]["lower"], :]
                self.centroids[str(k)] = np.mean(lower,axis=0)

            elif len(self.clusters[k]["lower"]) == 0 and len(self.clusters[k]["upper"]) != 0:
                # Get upper approximation vectors
                upper = self.data_array[self.clusters[k]["upper"], :]
                self.centroids[str(k)] = np.mean(upper,axis=0)

            else:
                # Get both upper-exclusive and lower approximation sets
                # upper = self.data_array[self.clusters[k]["upper"], :]
                lower = self.data_array[self.clusters[k]["lower"], :]
                exclusive_set = \
                    list(set(self.clusters[k]["upper"]).difference(set(self.clusters[k]["lower"])))
                boundary = self.data_array[exclusive_set, :]
                self.centroids[str(k)] = \
                    self.wght_lower*np.mean(lower,axis=0) + self.wght_upper*np.mean(boundary,axis=0)

            if self.debug_update is True:
                print("""###Cluster""", k, self.clusters[k]["lower"], self.clusters[k]["upper"])

        if self.timing is True:
            t3 = time.time()
            print("update_centroids Time", t3 - t1)

        return

    def assign_cluster_upper_lower_approximation(self):

        """
        Compute entity-to-cluster optimal assignments +
        upper/lower approximations for all current clusters

        :var self.distance
        :var self.distance_threshold
        :var self.cluster_list
        :var self.max_clusters
        :var self.data_length
        :return: self.clusters[clusters]["upper"] : upper approx.
        :return: self.clusters[clusters]["lower"] : lower approx.
        """

        t1 = time.time()

        # Reset clusters and distance weights for each method call
        self.clusters = {str(q): {"upper": [], "lower": []} for q in range(self.max_clusters)}
        self.d_weights = {str(q): {} for q in range(self.max_clusters)}

        # Assign each entity to cluster upper/lower approximations as appropriate
        for k in range(0, self.data_length):
            v_clust = self.cluster_list[str(k)]     # Current entity nearest cluster

            # Compile all clusters for each entity that are within
            # self.threshold distance of best entity cluster
            T = {j: self.distance[str(k)][j] / np.max([self.distance[str(k)][v_clust], self.small])
                 for j in self.distance[str(k)] if
                 (self.distance[str(k)][j] / np.max([self.distance[str(k)][v_clust], self.small])
                  <= self.dist_threshold)
                 and (v_clust != j)}

            # Assign entity to lower and upper approximations of all clusters as needed
            if len(T.keys()) > 0:
                self.clusters[v_clust]["upper"].append(k)      # Assign entity to its nearest cluster upper approx.
                self.d_weights[v_clust][str(k)] = \
                    ((2 / np.pi) * np.arctan(-self.p_param * (self.distance[str(k)][v_clust]))) + 1
                for cluster_name in T:
                    self.clusters[cluster_name]["upper"].append(k)  # Assign entity to upper approx of near cluster
                    self.d_weights[cluster_name][str(k)] = \
                        ((2 / np.pi) * np.arctan(-self.p_param * (self.distance[str(k)][cluster_name]))) + 1
            else:
                self.clusters[v_clust]["upper"].append(k)      # Assign entity to its nearest cluster upper approx.
                self.clusters[v_clust]["lower"].append(k)      # Assign entity to its nearest cluster lower approx.
                self.d_weights[v_clust][str(k)] = \
                    ((2 / np.pi) * np.arctan(-self.p_param * (self.distance[str(k)][v_clust]))) + 1
            if self.debug_assign is True:
                print("Current Cluster", v_clust)
                print("distance", self.distance[str(k)])
                print("T",T)

        if self.timing is True:
            t3 = time.time()
            print("assign_cluster_upper_lower_approximation Time", t3 - t1)

        return

    def get_entity_centroid_distances(self):

        """
        Compute entity-cluster distances and find nearest cluster
        for each entity and assign for all entities

        :var self.data_array : nd-array of all features for all entities
        :var self.centroids : nd-array of all cluster centroids
        :var self.max_clusters
        :return: self.distance : centroid-entity distance vectors
        :return self.cluster_list : best fit cluster-entity assignment
        """

        t1 = time.time()

        # Enumerate centroid distance vector for all entities and find nearest cluster and assign
        # distance1 = {}
        # for k in range(0,self.data_length):
        #     distance1[str(k)] = {str(j): np.linalg.norm([abs(self.data[val][k]-self.centroids[str(j)][val])
        #                                                      for val in self.feature_names])
        #                              for j in range(self.max_clusters)}
        #
        #     best_key = min(distance1[str(k)].iteritems(), key=operator.itemgetter(1))[0]
        #     self.cluster_list[str(k)] = best_key
        # t2 = time.time()

        tmp = []
        for l in range(0,self.max_clusters):
            tmp.append(np.linalg.norm(self.data_array - np.asarray(self.centroids[str(l)]),axis=1))

        for k in range(0,self.data_length):
            self.distance[str(k)] = {str(j): tmp[j][k] for j in range(self.max_clusters)}
            best_key = min(self.distance[str(k)].items(), key=operator.itemgetter(1))[0]
            self.cluster_list[str(k)] = best_key

        if self.debug_dist is True:
            print("Cluster List",self.cluster_list)
            print("Distances",self.distance)

        # Determine self.dist_threshold based on percentile all entity-cluster distances
        # curr_dists = list(itertools.chain([self.distance[h][g] for h in self.distance for g in self.distance[h]]))
        # self.dist_threshold = np.percentile(curr_dists,50)

        if self.timing is True:
            t3 = time.time()
            print("get_entity_centroid_distances Time",t3-t1)

        return
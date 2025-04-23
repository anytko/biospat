import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import warnings
from shapely.ops import unary_union
from scipy.spatial.distance import cdist


def merge_touching_groups(gdf, buffer_distance=0):
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    gdf = gdf.copy()

    if gdf.crs.to_epsg() != 3395:
        gdf = gdf.to_crs(epsg=3395)

    # Apply small positive buffer if requested (only for matching)
    if buffer_distance > 0:
        gdf["geometry_buffered"] = gdf.geometry.buffer(buffer_distance)
    else:
        gdf["geometry_buffered"] = gdf.geometry

    # Build spatial index on buffered geometry
    sindex = gdf.sindex

    groups = []
    assigned = set()

    for idx, geom in gdf["geometry_buffered"].items():
        if idx in assigned:
            continue
        # Find all polygons that touch or intersect
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        touching = possible_matches[
            possible_matches["geometry_buffered"].touches(geom)
            | possible_matches["geometry_buffered"].intersects(geom)
        ]

        # Include self
        touching_idxs = set(touching.index.tolist())
        touching_idxs.add(idx)

        # Expand to fully connected group
        group = set()
        to_check = touching_idxs.copy()
        while to_check:
            checking_idx = to_check.pop()
            if checking_idx in group:
                continue
            group.add(checking_idx)
            checking_geom = gdf["geometry_buffered"].loc[checking_idx]
            new_matches_idx = list(sindex.intersection(checking_geom.bounds))
            new_matches = gdf.iloc[new_matches_idx]
            new_touching = new_matches[
                new_matches["geometry_buffered"].touches(checking_geom)
                | new_matches["geometry_buffered"].intersects(checking_geom)
            ]
            new_touching_idxs = set(new_touching.index.tolist())
            to_check.update(new_touching_idxs - group)

        assigned.update(group)
        groups.append(group)

    # Merge geometries and attributes
    merged_records = []
    for group in groups:
        group_gdf = gdf.loc[list(group)]

        # Merge original geometries (NOT buffered ones)
        merged_geom = unary_union(group_gdf.geometry)

        # Aggregate attributes
        record = {}
        for col in gdf.columns:
            if col in ["geometry", "geometry_buffered"]:
                record["geometry"] = merged_geom
            else:
                if np.issubdtype(group_gdf[col].dtype, np.number):
                    record[col] = group_gdf[
                        col
                    ].sum()  # Sum numeric fields like AREA, PERIMETER
                else:
                    record[col] = group_gdf[col].iloc[
                        0
                    ]  # Keep the first value for text/categorical columns

        merged_records.append(record)

    merged_gdf = gpd.GeoDataFrame(merged_records, crs=gdf.crs)

    # Reset warnings filter to default
    warnings.filterwarnings("default", category=RuntimeWarning)

    return merged_gdf


def classify_range_edges(gdf, largest_polygons):
    """
    Classifies polygons into leading (poleward), core, and trailing (equatorward)
    edges within each cluster based on distance from the centroid of the largest polygon within each cluster.
    Includes longitudinal relict detection.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame with 'geometry' and 'cluster' columns.

    Returns:
        GeoDataFrame: The original GeoDataFrame with a new 'category' column.
    """

    # Ensure CRS is in EPSG:3395 (meters)
    if gdf.crs is None or gdf.crs.to_epsg() != 3395:
        gdf = gdf.to_crs(epsg=3395)

    # Compute centroids and extract coordinates
    gdf["centroid"] = gdf.geometry.centroid
    gdf["latitude"] = gdf["centroid"].y
    gdf["longitude"] = gdf["centroid"].x
    gdf["area"] = gdf.geometry.area  # Compute area

    # Find the centroid of the largest polygon within each cluster
    def find_largest_polygon_centroid(sub_gdf):
        largest_polygon = sub_gdf.loc[sub_gdf["area"].idxmax()]
        return largest_polygon["centroid"]

    cluster_centroids = (
        gdf.groupby("cluster")
        .apply(find_largest_polygon_centroid)
        .reset_index(name="cluster_centroid")
    )

    gdf = gdf.merge(cluster_centroids, on="cluster", how="left")

    # Classify polygons within each cluster based on latitude and longitude distance
    def classify_within_cluster(sub_gdf):
        cluster_centroid = sub_gdf["cluster_centroid"].iloc[0]
        cluster_lat = cluster_centroid.y
        cluster_lon = cluster_centroid.x

        largest_polygon_area = largest_polygons[0]["AREA"]

        # Define long_value based on area size
        if largest_polygon_area > 1000:
            long_value = 0.5  # for very large polygons, allow 10% longitude diff
        else:
            long_value = 0.05  # very small polygons, strict 1% longitude diff

        # Then calculate thresholds
        lat_threshold_01 = 0.1 * cluster_lat
        lat_threshold_05 = 0.05 * cluster_lat
        lat_threshold_02 = 0.02 * cluster_lat
        lon_threshold_01 = long_value * abs(cluster_lon)  # 5% of longitude

        def classify(row):
            lat_diff = row["latitude"] - cluster_lat
            lon_diff = row["longitude"] - cluster_lon

            # Relict by latitude
            if lat_diff <= -lat_threshold_01:
                return "relict (0.01 latitude)"
            # Relict by longitude
            if abs(lon_diff) >= lon_threshold_01:
                return "relict (0.001 longitude)"
            # Leading edge (poleward, high latitudes)
            if lat_diff >= lat_threshold_01:
                return "leading (0.99)"
            elif lat_diff >= lat_threshold_05:
                return "leading (0.95)"
            elif lat_diff >= lat_threshold_02:
                return "leading (0.9)"
            # Trailing edge (equatorward, low latitudes)
            elif lat_diff <= -lat_threshold_05:
                return "trailing (0.05)"
            elif lat_diff <= -lat_threshold_02:
                return "trailing (0.1)"
            else:
                return "core"

        sub_gdf["category"] = sub_gdf.apply(classify, axis=1)
        return sub_gdf

    gdf = gdf.groupby("cluster", group_keys=False).apply(classify_within_cluster)

    # Drop temporary columns
    gdf = gdf.drop(
        columns=["centroid", "latitude", "longitude", "area", "cluster_centroid"]
    )

    return gdf


def update_polygon_categories(largest_polygons, classified_polygons, island_states_gdf):

    largest_polygons_gdf = gpd.GeoDataFrame(largest_polygons)
    classified_polygons_gdf = gpd.GeoDataFrame(classified_polygons)
    island_states_gdf = island_states_gdf.to_crs("EPSG:3395")

    # Step 1: Set CRS for both GeoDataFrames (ensure consistency)
    largest_polygons_gdf.set_crs("EPSG:3395", inplace=True)
    classified_polygons_gdf.set_crs(
        "EPSG:3395", inplace=True
    )  # Ensure this matches the CRS of largest_polygons_gdf

    largest_polygons_gdf = gpd.sjoin(
        largest_polygons_gdf,
        classified_polygons[["geometry", "category"]],
        how="left",
        predicate="intersects",
    )

    # Step 3: Perform spatial join with classified_polygons_gdf and island_states_gdf (Assuming this is correct)
    overlapping_polygons = gpd.sjoin(
        classified_polygons_gdf, island_states_gdf, how="inner", predicate="intersects"
    )

    # Step 4: Clean up overlapping polygons
    overlapping_polygons = overlapping_polygons.rename(
        columns={"index": "overlapping_index"}
    )
    overlapping_polygons_new = overlapping_polygons.drop_duplicates(subset="geometry")

    # Step 5: Compute centroids for distance calculation
    overlapping_polygons_new["centroid"] = overlapping_polygons_new.geometry.centroid
    largest_polygons_gdf["centroid"] = largest_polygons_gdf.geometry.centroid

    # Step 6: Extract coordinates of centroids
    overlapping_centroids = (
        overlapping_polygons_new["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )
    largest_centroids = (
        largest_polygons_gdf["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )

    # Step 7: Calculate pairwise distance matrix
    distances = cdist(overlapping_centroids, largest_centroids)

    # Step 8: Find closest largest_polygon for each overlapping polygon
    closest_indices = distances.argmin(axis=1)

    # Step 9: Reassign 'category' from closest largest_polygons to overlapping_polygons
    overlapping_polygons_new["category"] = largest_polygons_gdf.iloc[closest_indices][
        "category"
    ].values

    # Step 10: Update categories in the original classified_polygons based on matching geometries
    # Here, we're only updating the category for polygons in the original gdf that overlap
    updated_classified_polygons = classified_polygons.copy()

    # Update only the overlapping polygons in the original GeoDataFrame
    updated_classified_polygons.loc[overlapping_polygons_new.index, "category"] = (
        overlapping_polygons_new["category"]
    )

    return updated_classified_polygons


def assign_polygon_clusters(polygon_gdf, island_states_gdf):
    range_test = polygon_gdf.copy()

    # Step 1: Reproject if necessary
    if range_test.crs.is_geographic:
        range_test = range_test.to_crs(epsg=3395)

    range_test = range_test.sort_values(by="AREA", ascending=False)

    largest_polygons = []
    largest_centroids = []
    clusters = []

    # Add the first polygon as part of num_largest with cluster 0
    first_polygon = range_test.iloc[0]

    # Check if the first polygon intersects or touches any island-state polygons
    if (
        not island_states_gdf.intersects(first_polygon.geometry).any()
        and not island_states_gdf.touches(first_polygon.geometry).any()
    ):
        largest_polygons.append(first_polygon)
        largest_centroids.append(first_polygon.geometry.centroid)
        clusters.append(0)

    # Step 2: Loop through the remaining polygons and check area and proximity
    for i in range(1, len(range_test)):
        polygon = range_test.iloc[i]

        # Calculate the area difference between the largest polygon and the current polygon
        area_difference = abs(largest_polygons[0]["AREA"] - polygon["AREA"])

        # Set the polygon threshold dynamically based on the area difference
        if area_difference > 600:
            polygon_threshold = (
                0.2  # Use a smaller threshold (1% of the largest polygon's area)
            )
        elif area_difference > 200:
            polygon_threshold = 0.005
        else:
            polygon_threshold = (
                0.2  # Use a larger threshold (20% of the largest polygon's area)
            )

        # Check if the polygon's area is greater than or equal to the threshold
        if polygon["AREA"] >= polygon_threshold * largest_polygons[0]["AREA"]:

            # Check if the polygon intersects or touches any island-state polygons
            if (
                island_states_gdf.intersects(polygon.geometry).any()
                or island_states_gdf.touches(polygon.geometry).any()
            ):
                continue  # Skip the polygon if it intersects or touches an island-state polygon

            # Calculate the distance between the polygon's centroid and all existing centroids in largest_centroids
            distances = []
            for centroid in largest_centroids:
                lat_diff = abs(polygon.geometry.centroid.y - centroid.y)
                lon_diff = abs(polygon.geometry.centroid.x - centroid.x)

                # If both latitude and longitude difference is below the threshold, this polygon is close
                if lat_diff <= 5 and lon_diff <= 5:
                    distances.append((lat_diff, lon_diff))

            # Check if the polygon is not within proximity threshold
            if not distances:
                # Add to num_largest polygons if it's not within proximity and meets the area condition
                largest_polygons.append(polygon)
                largest_centroids.append(polygon.geometry.centroid)
                clusters.append(
                    len(largest_polygons) - 1
                )  # Assign a new cluster for the new largest polygon
        else:
            pass

    # Step 3: Assign clusters to the remaining polygons based on proximity to largest polygons
    for i in range(len(range_test)):
        polygon = range_test.iloc[i]

        # If the polygon is part of num_largest, it gets its own cluster (already assigned)
        if any(
            polygon.geometry.equals(largest_polygon.geometry)
            for largest_polygon in largest_polygons
        ):
            continue  # Skip, as the num_largest polygons already have their clusters

        # Find the closest centroid in largest_centroids
        closest_centroid_idx = None
        min_distance = float("inf")

        for j, centroid in enumerate(largest_centroids):
            lat_diff = abs(polygon.geometry.centroid.y - centroid.y)
            lon_diff = abs(polygon.geometry.centroid.x - centroid.x)

            distance = np.sqrt(lat_diff**2 + lon_diff**2)  # Euclidean distance
            if distance < min_distance:
                min_distance = distance
                closest_centroid_idx = j

        # Assign the closest cluster
        clusters.append(closest_centroid_idx)

    # Add the clusters as a new column to the GeoDataFrame
    range_test["cluster"] = clusters

    return range_test, largest_polygons

map_df = map_df.sort_values(by="Lower Bound", ascending=False).reset_index(drop=True)
map_df.insert(0, "Assigned Credit Rating", range(1, num_buckets + 1))
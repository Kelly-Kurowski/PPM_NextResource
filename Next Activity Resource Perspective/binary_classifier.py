# Calculate the resource activity matrix
def create_diversity_matrix(log):

    activity_counts = log.pivot_table(index='org:resource',
                                  columns='concept:name',
                                  aggfunc='size',
                                  fill_value=0)

    # Resetting the index for a cleaner look
    activity_counts.reset_index(inplace=True)

    return activity_counts

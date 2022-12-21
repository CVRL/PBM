from helper_methods import get_centre,get_features,get_centre_fast,run_matching


def match_pair(probe_features,gallery_features,centre_probe,centre_gallery):

    delta_x = centre_probe[0] - centre_gallery[0]
    delta_y = centre_probe[1] - centre_gallery[1]

    if len(gallery_features) < len(probe_features):
        # print("Switching...")
        features_temp = probe_features
        probe_features = gallery_features
        gallery_features = features_temp

        centre_temp = centre_probe
        centre_probe = centre_gallery
        centre_gallery = centre_temp

        delta_x = centre_probe[0] - centre_gallery[0]
        delta_y = centre_probe[1] - centre_gallery[1]
        # centre_mask
        # cen = centre_mask2

    [matching_score,pairings,coord_pairings] = run_matching(probe_features,gallery_features,[centre_probe[0],centre_probe[1]],delta_x,delta_y)
    # all_pair_scores = []
    # for qwerty in pairings:
    #     all_pair_scores.append(pairings[qwerty])
    
    return [matching_score,coord_pairings]

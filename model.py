import random
from collections import defaultdict

# create validation set
def create_validation_set(reviews, meta, val_size=10000):
    random.seed(14)

    all_users = list(set(rev["user_id"] for rev in reviews))

    sampled_users = random.sample(all_users, val_size)

    reviews_valid = [rev for rev in reviews if rev["user_id"] in sampled_users]

    businesses_valid = set(rev["gmap_id"] for rev in reviews_valid)

    meta_valid = {business_id: meta[business_id] for business_id in businesses_valid if business_id in meta}

    return reviews_valid, meta_valid

# generate positive/negative samples for model training
def generate_samples(reviews, meta):
    random.seed(14)

    user_businesses = defaultdict(set)
    all_businesses = set()

    for rev in reviews:
        user_id = rev['user_id']
        business_id = rev['gmap_id']
        user_businesses[user_id].add(business_id)
        all_businesses.add(business_id)

    pos = []
    neg = []

    for user_id, reviewed_businesses in user_businesses.items():
        # pos samples (businesses the user has reviewed)
        for business_id in reviewed_businesses:
            pos.append((user_id, business_id))

        # neg samples (businesses the user has not reviewed)
        unreviewed_businesses = list(all_businesses - reviewed_businesses)

        n_neg = len(reviewed_businesses)
        sampled_negs = random.sample(unreviewed_businesses, n_neg)

        for business_id in sampled_negs:
            neg.append((user_id, business_id))
    
    return pos, neg

# split data into train/test sets
def train_test_split(pos, neg, test_size=0.2):
    random.seed(14)

    all_samples = [(user, business, 1) for user, business in pos] + [(user, business, 0) for user, business in neg]

    random.shuffle(all_samples)

    split_index = int(len(all_samples) * (1 - test_size))
    train_samples = all_samples[:split_index]
    test_samples = all_samples[split_index:]

    return train_samples, test_samples

# build baseline model
def baseline_model(train_samples):
    business_popularity = defaultdict(int)

    for user_id, business_id, label in train_samples:
        if label == 1:
            business_popularity[business_id] += 1

    sorted_businesses = [(count, business_id) for business_id, count in business_popularity.items()]
    sorted_businesses.sort(reverse=True)

    return sorted_businesses

# predict yes for top 50% popular businesses
def baseline_predict(sorted_businesses, total_reviews, thresh=0.5):
    topFifty = set()
    cum_revs = 0
    threshold_reviews = total_reviews * thresh

    for count, business_id in sorted_businesses:
        topFifty.add(business_id)
        cum_revs += count

        if cum_revs >= threshold_reviews:
            break

    return topFifty

def evaluate_baseline_model(test_samples, topFifty):
    TP = FP = TN = FN = 0

    for user_id, business_id, label in test_samples:
        pred = 1 if business_id in topFifty else 0

        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 1:
            FP += 1
        elif label == 0 and pred == 0:
            TN += 1
        
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def jaccard_similarity(set1, set2):
    # compute jaccard sim
    if not set1 or not set2:
        return 0
    
    numer = len(set1.intersection(set2))
    denom = len(set1.union(set2))

    if denom > 0:
        return numer/denom
    return 0

def find_similar_users(user_id, user_businesses, k=10):
    # get k most similar users based on reviewed businesses
    target_businesses = user_businesses.get(user_id, set())
    if not target_businesses:
        return []
    
    similarities = []
    for other_user, other_businesses in user_businesses.items():
        if other_user == user_id:
            continue
        
        sim = jaccard_similarity(target_businesses, other_businesses)
        if sim > 0:
            similarities.append((sim, other_user))
    
    similarities.sort(reverse=True)
    return similarities[:k]


def collaborative_filtering_predict(test_samples, user_businesses, 
                                   similarity_threshold=0.01, 
                                   min_similar_reviews=1):
    # predicts yes if similar users have reviewed the business
    predictions = set()
    
    for user_id, business_id, _ in test_samples:
        # get similar users
        similar_users = find_similar_users(user_id, user_businesses, k=20)
        
        # count num similar users who reviewed the business
        similar_review_count = 0
        for sim_score, similar_user in similar_users:
            # low threshold -> keep max num similar users
            if sim_score < similarity_threshold:
                break
            
            if business_id in user_businesses.get(similar_user, set()):
                similar_review_count += 1
        
        # predict yes if enough similar users reviewed the business
        if similar_review_count >= min_similar_reviews:
            predictions.add((user_id, business_id))
    
    return predictions

def evaluate_filtering_model(test_samples, predictions):
    # evaluate model
    TP = FP = TN = FN = 0
    
    for user_id, business_id, label in test_samples:
        pred = 1 if (user_id, business_id) in predictions else 0
        
        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 1:
            FP += 1
        elif label == 0 and pred == 0:
            TN += 1
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

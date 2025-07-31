import time

import numpy as np
from LLM.orchestrator import is_narrative
from sklearn.metrics.pairwise import cosine_similarity


def get_fake_label(top_matches, text, verbose=True):
    for node, sim_score in top_matches:
        idx = 0
        if node.level == 1:
            parent = node.parent
        else:
            parent = node
        while parent and idx < 3:
            start_time = time.time()  # Record the start time
            ent_label, result = is_narrative(text, parent.text)
            ent_label2, result2 = is_narrative(text, parent.text)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            print(f"Elapsed 2 ent gemma calls time: {elapsed_time:.4f} seconds")
            if (ent_label + ent_label2 > 1 ) and (result['score'] + result2['score'] > 1.5):
                print(f"Ent {result} {result2}")
                return 1, parent
            parent = parent.parent
            idx = idx + 1

        ent_label, result = is_narrative(text, node.text)
        ent_label2, result2 = is_narrative(text, node.text)
        if (ent_label + ent_label2 > 1 ) and (result['score'] + result2['score'] > 1.5):
            print(f"Ent {result} {result2}")
            return 1, node
    return 0, None


def get_label(row, matches):
    text = row['title']
    input_embedding = np.array(row['embedding']).reshape(1, -1)
    # Compute cosine similarities in bulk
    similarities = []
    start_time = time.time()  # Record the start time
    for node, embedding, idx in matches:
        embedding = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(input_embedding, embedding)
        similarities.append((node, similarity, idx))
    # Pair nodes with similarity scores and sort
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for search: {elapsed_time:.4f} seconds")

    start_time = time.time()  # Record the start time
    sorted_matches = sorted(similarities, key=lambda x: x[1], reverse=True)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for sort: {elapsed_time:.4f} seconds")

    to_send = [(match[0], match[1]) for match in sorted_matches][:3]

    start_time = time.time()  # Record the start time
    label, narrative = get_fake_label(to_send, text)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for get label: {elapsed_time:.4f} seconds")

    return label, narrative

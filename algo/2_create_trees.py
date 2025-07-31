import ast
import itertools
import json
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from LLM.orchestrator import fetch_embedding, is_narrative, get_common_narrative
from TreeNode import TreeNode
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def create_tree_list(df):
    """
        Create a list of trees from the dataframe

        Args:
            df (pandas df): dataframe

        Returns:
            list[TreeNode]: the list of disinformation trees
        """
    nodes = []
    for idx, row in df.iterrows():
        embedding = row['embedding']
        nodes.append(TreeNode(row['title'], embedding))
        if idx % 100 == 0:
            print(f"Getting embeddings {idx}")
    return nodes


def paralel_process_in_narrative(node, common_narrative):
    """
        Check if a text is part of a narrative
    """
    ent_label1, result = is_narrative(node.text, common_narrative)
    ent_label2, result = is_narrative(node.text, common_narrative)
    ent_label3, result = is_narrative(node.text, common_narrative)
    if ent_label3 + ent_label2 + ent_label1 >= 2:
        return node, 1
    return node, 0


def process_cluster(node_list):
    """
    Compute common narrative and embedding for a cluster.
    Args:
            node_list (list[TreeNode]): a list of nodes which are in the same cluster

        Returns:
            list[TreeNode]: the list of disinformation trees,
            with the nodes having the same narrative merged as children of a new node
    """
    if len(node_list) < 2:
        return node_list
    text_list = [node.text for node in node_list]
    common_narrative = get_common_narrative(text_list)
    if common_narrative == "no narrative" or common_narrative == "" or common_narrative is None:
        return node_list
    with ThreadPoolExecutor(max_workers=1) as executor:
        ent_results = list(
            executor.map(paralel_process_in_narrative, node_list, itertools.repeat(common_narrative)))

    not_narrative = [node for node, ent_label in ent_results if ent_label != 1]
    in_narrative = [node for node, ent_label in ent_results if ent_label == 1]

    if len(in_narrative) < 2:
        return node_list

    narrative_embedding = fetch_embedding(common_narrative)
    in_narrative_texts = [node.text for node in in_narrative]
    print(f"Cluster:\n{';  '.join(in_narrative_texts)}")
    print(f"Narrative: {common_narrative}")
    result = not_narrative
    result.append(TreeNode(common_narrative, narrative_embedding, children=in_narrative))
    return result


def recursive_compression_tree(node, parent):
    '''
    Merge recursively very similar nodes with their parent to reduce the depth of the tree
    :param node:
    :param parent:
    :return:
    '''
    if node.children:
        for child in node.children:
            recursive_compression_tree(child, node)
    if node.children and parent:
        if cosine_similarity(np.array(parent.embedding).reshape(1, -1), np.array(node.embedding).reshape(1, -1))[0][
            0] > 0.95:
            print(f"Merge: {parent.text} {node.text}")
            parent.children.remove(node)
            parent.children.extend(node.children)


def narrative_compression(nodes):
    for node in nodes:
        if node.level > 1:
            recursive_compression_tree(node, None)


def save_state(nodes, file_name):
    '''
    Save disinformation  structure to file
    '''
    all_nodes = nodes.copy()
    narrative_compression(all_nodes)
    sorted_trees = sorted(all_nodes, key=lambda x: x.level, reverse=True)
    with open("results/full_" + file_name + ".json", "w") as file:
        json.dump([tree.to_dict() for tree in sorted_trees], file, indent=4)
    with open("results/readable_" + file_name + ".json", "w") as file:
        json.dump([tree.to_clean_dict() for tree in sorted_trees], file, indent=4)

def run_algo(df):
    '''
    Main algorithm for creating the list of disinformation narratives
    '''
    #load elements in memory
    all_nodes = create_tree_list(df)
    stop_criteria = False
    cluster_max_dist = 0.1
    verbose = True
    index = 0

    #start processing
    while not stop_criteria:
        print(f"\n new iter {index} {cluster_max_dist}")
        embeddings = np.array([node.embedding for node in all_nodes])
        #create clusters
        linkage_matrix = linkage(embeddings, metric='cosine')
        clusters = fcluster(linkage_matrix, t=cluster_max_dist, criterion="distance")

        # Group elements from each cluster in dict
        cluster_dict = defaultdict(list)
        for node, cluster_id in zip(all_nodes, clusters):
            cluster_dict[cluster_id].append(node)

        # Iterate through clusters
        with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust worker count based on memory
            next_iter_nodes = list(executor.map(process_cluster, cluster_dict.values()))
        # flatten list
        next_iter_nodes = [item for sublist in next_iter_nodes for item in sublist]

        if len(all_nodes) <= len(next_iter_nodes):
            if cluster_max_dist < 0.9:
                #save the state
                save_state(next_iter_nodes, "result_" + str(cluster_max_dist))
                cluster_max_dist = round(cluster_max_dist + 0.1, 2)
            else:
                stop_criteria = True
        all_nodes = next_iter_nodes.copy()
        index = index + 1

    save_state(all_nodes, "result_" + str(cluster_max_dist))

if __name__ == '__main__':
    df = pd.read_csv('data/work_data/train_only_fake.csv')
    df = df[df['title'] != ""]
    df = df.dropna()
    df['title'] = df["title"].str.replace('"', '', regex=False)
    print(f"Train data {df.shape}")
    print("Get embeddings")
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    run_algo(df)

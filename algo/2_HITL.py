import ast
import json
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from LLM.orchestrator import is_narrative
from TreeNode import TreeNode
from sklearn.metrics.pairwise import cosine_similarity

node_objects = []
node_embeddings = []


# set_node_parent(all_nodes, None)
def set_node_parent(nodes, parent):
    for node in nodes:
        node.parent = parent
        set_node_parent(node.children, node)


matches = []


def traverse_tree(node, matches, node_id):
    # Add the current node and its embedding to the matches list
    matches.append((node, node.embedding, node_id))
    for child in node.children:
        traverse_tree(child, matches, node_id)


count_found_narrative = 0
found_narrative_correct = 0
count_found_te = 0
has_parent = 0
te_correct = 0


def get_fake_label(top_matches, text, row):
    global count_found_narrative, count_found_te, has_parent, found_narrative_correct, te_correct
    texts = []
    for node, sim_score in top_matches:
        texts.append(node.text)

        idx = 0
        if node.level == 1:
            parent = node.parent
        else:
            parent = node

        while parent and idx < 10:
            texts.append(parent.text)
            ent_label, result = is_narrative(text, parent.text)
            ent_label2, result = is_narrative(text, parent.text)

            if ent_label + ent_label2 > 1:
                if row['label'] == 0:
                   print(
                       f"\n0-{idx}Text: {text} ; \nParent Text: {parent.text} ;\nSim Score: {sim_score} ;  \nEntP 1 parent label:{row['label']} Entailm {result}")
                count_found_narrative = count_found_narrative + 1
                if row['label'] == 1:
                    found_narrative_correct = found_narrative_correct + 1

                return 1
            parent = parent.parent
            idx = idx + 1
        # if node.level != 1:
        ent_label, result = is_narrative(text, node.text)
        ent_label2, result = is_narrative(text, node.text)


        if ent_label + ent_label2 > 1:
            count_found_te = count_found_te + 1
            if node.parent and len(node.children) == 0:
                has_parent = has_parent + 1
            if row['label'] == 1:
                te_correct = te_correct + 1
            if row['label'] == 0:
                print(
                    f"\n\n1Text: {text} ; \nFn: {node.text} ; \nSim Score: {sim_score} ; \nParent Text: {node.parent.text if node.parent else 'none'} \nEntN 1 parent label:{row['label']} Entailm {result}")
            return 1

    if row['label'] == 1:
        texts = '\n'.join(texts)
        print(f"\nNot Found Narrative: {text} ; \n;"
        f" Top matches: {texts}")

    return 0


index = 0
all_nodes = []
big_embeddings = []


def process_row(row):
    global node_objects, node_embeddings, matches, index
    text = row['title']
    input_embedding = np.array(row['embedding']).reshape(1, -1)
    # Compute cosine similarities in bulk
    similarities = []
    label = 0
    for node, embedding, idx in matches:
        embedding = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(input_embedding, embedding)
        similarities.append((node, similarity, idx))
    # Pair nodes with similarity scores and sort
    sorted_matches = sorted(similarities, key=lambda x: x[1], reverse=True)
    to_send = [(match[0], match[1]) for match in sorted_matches][:3]
    label = get_fake_label(to_send, text, row)
    if index % 10 == 0:
        print(f"Index: {index}")
    index = index + 1
    return label


def main():
    global node_objects, node_embeddings, matches, count_found_narrative, count_found_te, all_nodes, has_parent, found_narrative_correct, te_correct
    df = pd.read_csv('data/work_data/correction_df.csv')
    df['label'] = df['label'].apply(lambda x: 1 if not x else 0)
    print(f"Shape of test dataframe {df.shape} with {df['label'].value_counts()}")

    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    print(f"finish literal eval of embeddings {df.columns} df shape {df.shape}")
    acc = []
    res = []

    for i in range(4, 5):
        index = i / 10
        count_found_narrative = 0
        count_found_te = 0
        has_parent = 0
        found_narrative_correct = 0
        te_correct = 0
        print(f"\nStart new Index: {index}")
        with open("results/full_result_" + str(index) + ".json", "r") as file:
            data = json.load(file)
        all_nodes = [TreeNode.from_dict(tree) for tree in data]

        node_id = 0
        for node in all_nodes:
            traverse_tree(node, matches, node_id)
            node_id = node_id + 1
        with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust worker count as needed
            label_results = list(executor.map(process_row, df.to_dict(orient="records")))
        accuracy = accuracy_score(df['label'].tolist(), label_results)
        cm = confusion_matrix(df['label'].tolist(), label_results)
        print("Confusion Matrix:")
        print(cm)

        print()
        print(f"Accuracy {index}: {accuracy}")
        print(f"Count foun narrative {count_found_narrative} ot of which correct {found_narrative_correct}, TX {count_found_te}, out of which has parent {has_parent} out of which correct {te_correct}")
        acc.append(accuracy)
        res.append(cm)


if __name__ == '__main__':
    main()

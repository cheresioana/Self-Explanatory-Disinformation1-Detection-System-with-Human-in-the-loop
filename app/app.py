import time

from flask import Flask, render_template, request, url_for, redirect

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LLM.orchestrator import fetch_embedding
from algo.TreeNode import TreeNode
from algo.eval import load_structure_file, traverse_tree
from algo.manual_clean import remove_node
from utils import clean_text, sort_tree_recursive
from algo.get_label import get_label
app = Flask(__name__)

all_nodes, matches = load_structure_file("results/full_result_0.4_change3.json")
intro_texts = []
all_nodes = sort_tree_recursive(all_nodes)
spotlight_node = None #all_nodes[0] #None #all_nodes[0].children[0].children[0]
spot_children = [] #all_nodes[0].children


@app.route("/")
def home():
    return render_template("index.html", intro_texts=intro_texts, spotlight_node=spotlight_node, children=spot_children)

@app.route('/process_text', methods=['POST'])
def process_text():
    '''
    Finds if a text is part pf a narrative
    Expects:
        JSON body with a 'user_text' field.

    Returns:
        renders the page with the new element
    '''
    global matches
    user_text = request.form.get('user_text')
    print("Received text:", user_text)
    row = {
        'title': user_text,
        'embedding': fetch_embedding(clean_text(user_text))
    }
    start_time = time.time()  # Record the start time
    label, node = get_label(row, matches)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed total time: {elapsed_time:.4f} seconds")
    intro_texts.insert(0,{'label':label, 'node':node, 'user_text': user_text})
    print(f"Results for text: {user_text}:\nLabel: {label}, Narrative: {node.text if node else ''}")
    return redirect(url_for('home'))

@app.route('/get_internal_structure', methods=['POST'])
def get_internal_structure():
    '''
        Show internal structure of a node
        Expects:
            JSON body with a 'narrative_text' field.

        Returns:
            renders the page with the internal structure of the node by updating spotlight_node and spot_children
        '''
    global spotlight_node, spot_children
    narrative_text = request.form.get('narrative_text')
    print("Received text:", narrative_text)
    for element in intro_texts:
        print(element['node'])

        if element['node'] and element['node'].text == narrative_text:
            spotlight_node = element['node']
            spot_children = spotlight_node.children
            print(len(spot_children))
    return redirect(url_for('home'))

@app.route('/edit_node', methods=['POST'])
def edit_node():
    global spotlight_node
    old_text = request.form.get('old_text')
    text = request.form.get('new_text')
    print(old_text)
    print(text)
    spotlight_node.text = text
    spotlight_node.embedding = fetch_embedding(clean_text(text))
    for matche in matches:
        if matche[0].text == old_text:
            print('found it')
            matches.remove(matche)
            matches.add((spotlight_node, spotlight_node.embedding, -1))
    return redirect(url_for('home'))

@app.route('/mark_item_as_fake', methods=['POST'])
def mark_item_as_fake():
    global all_nodes, matches
    print("Mark item as fake")
    text = request.form.get('narrative_text')
    print(text)
    embedding = fetch_embedding(clean_text(text))
    node = TreeNode(text, embedding=embedding, children=[], parent=None)
    print(len(matches))
    all_nodes.append(node)
    matches.append((node, embedding, -1))
    print(len(matches))
    print(matches[-1][0].to_dict())
    for element in intro_texts:
        if element['user_text'] == text:
            intro_texts.remove(element)
    return redirect(url_for('home'))

@app.route('/go_to_parent', methods=['POST'])
def go_to_parent():
    global spotlight_node, spot_children
    spotlight_node = spotlight_node.parent
    if spotlight_node == None:
        spot_children = all_nodes[:15]
    else:
        spot_children = spotlight_node.children
    return redirect(url_for('home'))

@app.route('/go_to_children', methods=['POST'])
def go_to_children():
    global spotlight_node, spot_children
    narrative_text = request.form.get('narrative_text')
    print("Received text:", narrative_text)
    if spotlight_node != None:
        for element in spotlight_node.children:
            if element.text == narrative_text:
                print(f"Found element {element.text}")
                spotlight_node = element
                spot_children = spotlight_node.children
    else:
        for element in spot_children:
            if element.text == narrative_text:
                spotlight_node = element
                spot_children = spotlight_node.children
    return redirect(url_for('home'))


def remove_node_nodes(node):
    global all_nodes, matches
    new_nodes = []
    for element in all_nodes:
        new_nodes.extend(remove_node(element, node.text, []))
    all_nodes = new_nodes
    new_matches = []
    node_id = 0
    for node in all_nodes:
        traverse_tree(node, new_matches, node_id)
        node_id = node_id + 1
    matches = new_matches

@app.route('/remove_node', methods=['POST'])
def remove_node_endpoint():
    global spotlight_node, spot_children
    narrative_text = request.form.get('narrative_text')
    print("Received remove text:", narrative_text)
    if spotlight_node is not None and spotlight_node.text == narrative_text:
        print("does it for spotlight node")
        remove_node_nodes(spotlight_node)
        spotlight_node = None
        spot_children = []
    else:
        new_children = []
        for element in spot_children:
            if element.text == narrative_text:
                remove_node_nodes(element)
            else:
                new_children.append(element)
        spot_children = new_children
    for element in intro_texts:
        if element['node'] is not None and (element['node'].text == narrative_text or element['user_text'] == narrative_text):
            intro_texts.remove(element)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
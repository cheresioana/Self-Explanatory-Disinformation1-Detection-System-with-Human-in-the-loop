import json

from LLM.orchestrator import fetch_embedding
from algo.TreeNode import TreeNode


covid_04_remove_narrative = []
covid_04_keepchild = []
covid_04_add_narratives =  ["Ioana kept secret the number of deaths"]


covid_05_remove_narrative = []
covid_05_keepchild = []
covid_05_add_narratives = [ "Ukrainians are connected to nazis",
                           "Ukrainian refugees abuse countries",
                            "Masks and lockdown are useless",
                            "Russia is fighting in Ukraine for important values"]


covid_06_remove_narrative = []
covid_06_keepchild = []
covid_06_add_narratives =  [ "Ukrainians are connected to nazis",
                           "Ukrainian refugees abuse countries",
                            "Masks and lockdown are useless",
                            "Russia is fighting in Ukraine for important values"]


covid_07_remove_narrative = []
covid_07_keepchild = []
covid_07_add_narratives = [ "Ukrainians are connected to nazis",
                           "Ukrainian refugees abuse countries",
                            "Masks and lockdown are useless",
                            "Russia is fighting in Ukraine for important values"]


covid_08_remove_narrative = []
covid_08_keepchild = []
covid_08_add_narratives = [ "Ukrainians are connected to nazis",
                           "Ukrainian refugees abuse countries",
                            "Masks and lockdown are useless",
                            "Russia is fighting in Ukraine for important values"]

covid_09_remove_narrative = []
covid_09_keepchild = []
covid_09_add_narratives = [ "Ukrainians are connected to nazis",
                           "Ukrainian refugees abuse countries",
                            "Masks and lockdown are useless",
                            "Russia is fighting in Ukraine for important values"]

# covid_09_add_narratives = ["Romania will be dismantled in the event of a Russian invasion in Ukraine",
# "Masks and Lockdown are useless", "Ukrainians are connected to nazis", " Russia is fighting in Ukraine for important values",
#                            "Ukrainian refugees abuse countries"]
def remove_node(node, del_narrative, keep_child, rename=[]):
    new_nodes = []
    #print(node.text)
    if (node.text in del_narrative):
        children = node.children
        print(f"OUT :{node.text}")
        for child in children:
            if (child.text in keep_child):
                print(f"Keep: {child.text}")
                new_nodes.append(child)
    elif (node.text in ['Ukraine is a threat to Russia and must be eliminated.']):
        print('HERE')
        node.text = 'Ukraine is committing genocide and must be eliminated'
        return [node]
    # elif (node.text in ['Ukraine is a destabilizing force in Eastern Europe.']):
    #     node.text = 'Ukraine is destabilizing Europe'
    #     print('HERE  2')
    #     return [node]

    else:
        new_children = []
        for child in node.children:
            new_children.extend(remove_node(child, del_narrative, keep_child))
        for child in new_children:
            child.parent = node
        if node.text == "COVID-19 is an artificial virus from a laboratory in China":
            print("I am here!")
            child_node = TreeNode("COVID-19 was done by the chinese in their secret labs",
                                  fetch_embedding("COVID-19 was done by the chinese in their secret labs"))
            narrative_node = TreeNode("COVID-19 was created in a laboratory in China",
                                      fetch_embedding("COVID-19 was created in a laboratory in China"),
                                      children=[node, child_node], parent=None)
            node.parent = narrative_node
            child_node.parent = narrative_node
            return [narrative_node]
        node.children = new_children
        return [node]
    return new_nodes


def add_trees(nodes, narratives):
    for narrative in narratives:
        nodes.append(TreeNode(narrative, fetch_embedding(narrative), children=[]))
    return nodes


def process_07():
    with open("results/full_result_0.7.json", "r") as file:
        data = json.load(file)
    all_nodes = [TreeNode.from_dict(tree) for tree in data]
    new_nodes = []
    print(len(all_nodes))
    for nodes in all_nodes:
        new_nodes.extend(remove_node(nodes, covid_07_remove_narrative, covid_07_keepchild))
    new_nodes = add_trees(new_nodes, covid_07_add_narratives)
    sorted_trees = sorted(new_nodes, key=lambda x: x.level, reverse=True)
    with open("results/full_result_0.7_change2.json", "w") as file:
        json.dump([tree.to_dict() for tree in sorted_trees], file, indent=4)
    with open("results/readable_result_0.7_change2.json", "w") as file:
        json.dump([tree.to_clean_dict() for tree in sorted_trees], file, indent=4)


def process_08():
    with open("results/full_result_0.8.json", "r") as file:
        data = json.load(file)
    all_nodes = [TreeNode.from_dict(tree) for tree in data]
    new_nodes = []
    print(len(all_nodes))
    for nodes in all_nodes:
        new_nodes.extend(remove_node(nodes, covid_08_remove_narrative, covid_08_keepchild))
    new_nodes = add_trees(new_nodes, covid_08_add_narratives)
    sorted_trees = sorted(new_nodes, key=lambda x: x.level, reverse=True)
    with open("results/full_result_0.8_change2.json", "w") as file:
        json.dump([tree.to_dict() for tree in sorted_trees], file, indent=4)
    with open("results/readable_result_0.8_change2.json", "w") as file:
        json.dump([tree.to_clean_dict() for tree in sorted_trees], file, indent=4)

def process_06():
    with open("results/full_result_0.6.json", "r") as file:
        data = json.load(file)
    all_nodes = [TreeNode.from_dict(tree) for tree in data]
    new_nodes = []
    print(len(all_nodes))
    for nodes in all_nodes:
        new_nodes.extend(remove_node(nodes, covid_06_remove_narrative, covid_06_keepchild))
    new_nodes = add_trees(new_nodes, covid_06_add_narratives)
    sorted_trees = sorted(new_nodes, key=lambda x: x.level, reverse=True)
    with open("results/full_result_0.6_change2.json", "w") as file:
        json.dump([tree.to_dict() for tree in sorted_trees], file, indent=4)
    with open("results/readable_result_0.6_change2.json", "w") as file:
        json.dump([tree.to_clean_dict() for tree in sorted_trees], file, indent=4)


def process_05():
    with open("results/full_result_0.5.json", "r") as file:
        data = json.load(file)
    all_nodes = [TreeNode.from_dict(tree) for tree in data]
    new_nodes = []
    print(len(all_nodes))
    for nodes in all_nodes:
        new_nodes.extend(remove_node(nodes, covid_05_remove_narrative, covid_05_keepchild))
    new_nodes = add_trees(new_nodes, covid_05_add_narratives)
    sorted_trees = sorted(new_nodes, key=lambda x: x.level, reverse=True)
    with open("results/full_result_0.5_change2.json", "w") as file:
        json.dump([tree.to_dict() for tree in sorted_trees], file, indent=4)
    with open("results/readable_result_0.5_change2.json", "w") as file:
        json.dump([tree.to_clean_dict() for tree in sorted_trees], file, indent=4)

def process_04():
    with open("results/full_result_0.4.json", "r") as file:
        data = json.load(file)
    all_nodes = [TreeNode.from_dict(tree) for tree in data]
    new_nodes = []
    print(len(all_nodes))
    for nodes in all_nodes:
        new_nodes.extend(remove_node(nodes, covid_04_remove_narrative, covid_04_keepchild))
    new_nodes = add_trees(new_nodes, covid_04_add_narratives)
    sorted_trees = sorted(new_nodes, key=lambda x: x.level, reverse=True)
    with open("results/full_result_0.4_change3.json", "w") as file:
        json.dump([tree.to_dict() for tree in sorted_trees], file, indent=4)
    with open("results/readable_result_0.4_change3.json", "w") as file:
        json.dump([tree.to_clean_dict() for tree in sorted_trees], file, indent=4)

if __name__ == '__main__':
    #process_07()
    #process_08()
    #process_06()
    #process_05()
    process_04()

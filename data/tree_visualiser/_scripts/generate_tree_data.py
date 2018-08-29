import hashlib
import os
import copy
import shutil
import json
import pandas as pd
from markdown import markdown
from collections import OrderedDict
from treelib import Tree, exceptions


# Create a SuperTree class to allow modification of show() method
class SuperTree(Tree):
    def __init__(self):
        super().__init__()

    def to_dict(self, nid=None, key=None, sort=True, reverse=False, with_data=False, show_ids=False):
        """transform self into a dict"""

        nid = self.root if (nid is None) else nid
        tree_dict = OrderedDict({"children": []})
        tree_dict["size"] = 20
        node_key = ''
        for potential_key in ['Question', 'FollowUpQuestion', 'FinalAnswer']:
            if potential_key in self[nid].data:
                node_key = self[nid].data[potential_key]
                if 'AssignmentId' in self[nid].data and show_ids:
                    node_key += ' (' + self[nid].data['AssignmentId'] + ')'

        tree_dict["key"] = node_key
        # Add colorCode
        if 'LastRandomAnswer' in self[nid].data:
            tree_dict["colorCode"] = self[nid].data['LastRandomAnswer']

        if self[nid].expanded:
            queue = [self[i] for i in self[nid].fpointer]
            # Sort the queue
            if sort:
                queue = sorted(queue, key=lambda x: x.data["LastRandomAnswer"], reverse=True)

            for elem in queue:
                tree_dict["children"].append(
                    self.to_dict(elem.identifier, with_data=with_data, sort=sort, reverse=reverse))

            # If no children left
            if len(tree_dict["children"]) == 0:
                # Drop children from dict
                if "children" in tree_dict:
                    del tree_dict["children"]

            return tree_dict


# Generating the Trees
RANDOM_SEED = 27


def prepro(text):
    return text.strip().lower()


def convert_markdown_to_html(text):
    return markdown(text).replace('\n', '')


def get_tree_id(snippet, question, scenario):
    str_to_encode = prepro(snippet) + prepro(question) + prepro(scenario)
    hash_sha256 = hashlib.sha256(str_to_encode.encode('utf8')).hexdigest()
    hash_sha1 = hashlib.sha1(hash_sha256.encode('utf8')).hexdigest()
    return hash_sha1


def build_forest(data):
    forest = {}
    for d in data:
        utterance_id = d['utterance_id']
        snippet = d['snippet']
        question = d['question']
        # scenario = d['scenario']
        history = d['history']
        answer = d['answer']
        # evidence = d['evidence']
        # print_evidence = str([(e['follow_up_question'], e['follow_up_answer']) for e in evidence])
        y = answer.capitalize()
        this_id = d['tree_id']
        if this_id in forest:
            tree = forest[this_id]
        else:
            tree = SuperTree()
            tree.create_node(identifier=this_id,
                             data={
                                 'TreeId': this_id,
                                 'Snippet': convert_markdown_to_html(snippet.replace('_new_line_', '\n')),
                                 'Question': question,
                                 # 'Scenario': scenario
                             })

        last_parent_node = this_id
        prev_fua = ''
        for h in history:
            fuq, fua = h['follow_up_question'], h['follow_up_answer']
            fua = fua.capitalize()
            next_selector = last_parent_node.replace('-y', '&y').replace('-n', '&n')
            fuq_id = next_selector + '|' + fuq
            fua_id = fuq_id + '-' + prepro(fua)
            if not tree.contains(fuq_id):
                # If node already in tree, don't recreate
                tree.create_node(tag=fuq,
                                 identifier=fuq_id,
                                 parent=last_parent_node,
                                 data={
                                     'FollowUpQuestion': fuq,
                                     'LastRandomAnswer': prev_fua,
                                 })
            if not tree.contains(fua_id):
                # If node already in tree, don't recreate
                tree.create_node(tag=fua,
                                 identifier=fua_id,
                                 parent=fuq_id,
                                 data={
                                     'RandomAnswer': fua,
                                 })

            prev_fua = fua
            last_parent_node = fua_id

        if y in ['Yes', 'No', 'Irrelevant']:
            # Add the final value
            try:
                print_final_answer = y
                # if print_evidence:
                #     print_final_answer += ' ' + print_evidence
                tree.create_node(tag=y,
                                 identifier=last_parent_node + '_' + y,
                                 parent=last_parent_node,
                                 data={
                                     'FinalAnswer': print_final_answer,
                                     'LastRandomAnswer': prev_fua,
                                 })
            except exceptions.DuplicatedNodeIdError:
                print(f"WARNING: duplicate FINAL node attempt. Utterance Id: {utterance_id}")

        forest[this_id] = tree
    return forest


def trim_tree(tree):
    all_nodes = tree.all_nodes()
    nodes_to_link_past = [node.identifier for node in all_nodes if 'RandomAnswer' in node.data]
    for nid in nodes_to_link_past:
        tree.link_past_node(nid)
    return tree


def trim_forest(forest):
    new_forest = copy.deepcopy(forest)
    forest = {key: trim_tree(tree) for key, tree in new_forest.items()}
    return forest


def count_utterances(forest):
    num_utterances = 0
    for key, tree in forest.items():
        utterance_nodes = [node for node in tree.nodes.values() if 'FinalAnswer' in node.data or
                           'FollowUpQuestion' in node.data]
        num_nodes = len(utterance_nodes)
        num_utterances += num_nodes
    return num_utterances


def load_and_build_forest(files, verbose=True):
    loaded_data = []
    for file in files:
        with open(file, 'r') as f:
            loaded_data.extend(json.load(f))
        # Filter the loaded data to keep the positive utterances only and those without scenarios

    loaded_data = [x for x in loaded_data if not prepro(x['scenario']) and prepro(x['answer']) != "irrelevant"]
    forest = build_forest(loaded_data)

    if verbose:
        print(f"{len(loaded_data)} utterances loaded and filtered from '{file}'.")
        print(f"{count_utterances(forest)} utterances converted to trees.")

    return forest


def generate_visualisation(files, tree_data_dir='../tree_data_dev_test', verbose=True):
    forest = load_and_build_forest(files=files, verbose=verbose)
    # Tree visualiser
    list_keys = []
    new_forest = trim_forest(forest)
    if os.path.exists(tree_data_dir):
        shutil.rmtree(tree_data_dir)
    os.makedirs(tree_data_dir, exist_ok=True)
    for key, tree in new_forest.items():
        # Save the generated data file
        list_keys.append((len(list_keys), key, tree.get_node(tree.root).data["Snippet"]))
        with open(os.path.join(tree_data_dir, f'{key}.json'), 'w') as outfile:
            outfile.write(tree.to_json())

    # Save list_keys as a DataFrame.to_csv() for d3 to read
    df = pd.DataFrame(sorted(list_keys, key=lambda x: x[1]), columns=['value', 'label', 'snippet'])
    df.to_csv(os.path.join(tree_data_dir, 'data.csv'), index=False)

    if verbose:
        print(f'{len(df)} trees successfully generated.')


def main(files, verbose=True):
    generate_visualisation(files=files, verbose=verbose)


if __name__ == '__main__':
    main(files=['sharc_train.json'])

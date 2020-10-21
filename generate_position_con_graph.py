# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')



def dependency_adj_matrix(text, aspect, position):
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    text_list = text.split()
    
    for token in document:
        if str(token) in aspect:
            weight = 1
            if token.i < seq_len:
                for j in range(seq_len):
                    if text_list[j] in aspect:
                        sub_weight = 1
                    else:
                        sub_weight = 1 / (abs(j - int(position)) + 1)
                    matrix[token.i][j] = 1 * sub_weight
                    matrix[j][token.i] = 1 * sub_weight
        else:
            weight = 1 / (abs(token.i - int(position)) + 1)
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            for child in token.children:
                if str(child) in aspect:
                    weight += 1
                else:
                    weight += 1 / (abs(child.i - int(position)) + 1)
                if child.i < seq_len:
                    matrix[token.i][child.i] += 1 * weight
                    matrix[child.i][token.i] += 1 * weight

    return matrix


def get_con_adj_matrix(aspect, position, aspect_graphs, other_aspects):
    adj_matrix = aspect_graphs[aspect]
    position = int(position)
    if len(aspect_graphs) == 1:
        return adj_matrix
    for other_a in other_aspects:
        other_p = int(other_aspects[other_a])
        other_m = aspect_graphs[other_a]
        alpha = 1 / (abs(position - other_p) + 1)
        weight = 1 / len(aspect_graphs)
        adj_matrix += alpha * weight * other_m
    return adj_matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph_con_a', 'wb')
    graph_idx = 0
    for i in range(len(lines)):
        aspects, polarities, positions, text = lines[i].split('\t')
        aspect_list = aspects.split('||')
        polarity_list = polarities.split('||')
        position_list = positions.split('||')
        text = text.lower().strip()
        aspect_graphs = {}
        aspect_positions = {}
        for aspect, position in zip(aspect_list, position_list):
            aspect_positions[aspect] = position
        for aspect, position in zip(aspect_list, position_list):
            other_aspects = aspect_positions.copy()
            aspect = aspect.lower().strip()
            del other_aspects[aspect]
            adj_matrix = dependency_adj_matrix(text, aspect, position)
            aspect_graphs[aspect] = adj_matrix
        for aspect, position in zip(aspect_list, position_list):
            aspect = aspect.lower().strip()
            other_aspects = aspect_positions.copy()
            del other_aspects[aspect]
            adj_matrix = get_con_adj_matrix(aspect, position, aspect_graphs, other_aspects)
            idx2graph[graph_idx] = adj_matrix
            graph_idx += 1
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close() 

if __name__ == '__main__':
    process('./con_datasets/rest14_train.raw')
    process('./con_datasets/rest14_test.raw')
    process('./con_datasets/lap14_train.raw')
    process('./con_datasets/lap14_test.raw')
    process('./con_datasets/rest15_train.raw')
    process('./con_datasets/rest15_test.raw')
    process('./con_datasets/rest16_train.raw')
    process('./con_datasets/rest16_test.raw')


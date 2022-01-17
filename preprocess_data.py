#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: preprocess_data.py
# @Version: 
# @Author: BinLiang
# @Mail: 18b951033@stu.hit.edu.cn
# @For: 
# @Created Time: Sun 14 Jun 2020 04:12:55 PM CST
# ------------------

def process():
    path = './orig_datasets/lap14_test.raw'
    w_path = './con_datasets/lap14_test.raw'
    fp = open(path, 'r')
    w_fp = open(w_path, 'w')
    lines = fp.readlines()
    pre_context = ''
    aspect_list = []
    polarity_list = []
    context_dic = {}
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        position = str(len(text_left.split()))
        context = text_left + " " + aspect + " " + text_right
        context = context.strip()
        #print('orig text:', lines[i].strip())
        #print('aspect:', aspect)
        #print('text_left:', text_left)
        #print('text_right:', text_right)
        #print('context:', context)
        #print('seq len:', len(context.split()))
        #print('='*30)
        if context not in context_dic:
            context_dic[context] = {'aspect': [], 'polarity': [], 'position': []}
            context_dic[context]['aspect'] = [aspect]
            context_dic[context]['polarity'] = [polarity]
            context_dic[context]['position'] = [position]
        else:
            context_dic[context]['aspect'].append(aspect)
            context_dic[context]['polarity'].append(polarity)
            context_dic[context]['position'].append(position)
    for context in context_dic:
        aspect_list = context_dic[context]['aspect']
        polarity_list = context_dic[context]['polarity']
        position_list = context_dic[context]['position']
        aspects = '||'.join(aspect_list)
        polarities = '||'.join(polarity_list)
        positions = '||'.join(position_list)
        line = aspects + '\t' + polarities + '\t' + positions + '\t' + context + '\n'
        w_fp.write(line)

    fp.close()
    w_fp.close()

if __name__ == '__main__':
    process()

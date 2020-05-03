# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:50:48 2019

@author: 973065
"""
import sys
import os
import unicodedata
import string
import re

exclude = set(string.punctuation)

#%%
def file_parser(file1, file2):
    
    fw = open(file2, "a", encoding="utf8")
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        print("File path {} does not exist. Exiting...".format(file1))
        sys.exit()
  
    word_list = []
    with open(file1, "r", encoding="utf8") as fp:

        for line in fp:
            line = line.strip()
            stop_free = " ".join([j for j in line.lower().split()])
            punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
            normalized = " ".join(word for word in punc_free.split())
            processed = re.sub(r"\d+", "", normalized)
            words = processed.split(' ')
            #words = line.strip().split(' ')
            #print(words)

            for word in words:
                if not words:
                    break
                if len(words) == 1 and word == 'text':
                    word_list.append('zaciatok_mailu:')
                    break
                if ((len(word) > 20) or ('slspocloudouexchange' in word) or ('account' in word)
                    ('dakuj' in word)):
                    word_list.append('koniec_mailu:')
                    break
                
                word_list.append(word)
                    
    start_substr = False
    parsed_text = []
    new_word = ""
    mail_count = 0
    
    for word in word_list:
        if 'zaciatok_mailu:' in word:
            start_substr = True
            continue
        
        if 'koniec_mailu:' in word and start_substr:
            start_substr = False
            new_word = unicodedata.normalize('NFKD', new_word).encode('ASCII', 'ignore').decode("utf-8")
            parsed_text.append(new_word.lower())
            fw.write(new_word.lower() + '\n')
            new_word = ""
            mail_count += 1
            continue
        
        if start_substr:
            new_word += (word.lower() + ' ')
            
    print(parsed_text)
    print()
    print('pocet mailov ', mail_count)
    fw.close()
            
               
def remove_diacritics(source_file, undiacrized_file):
    if not os.path.isfile(source_file):
        print("File path {} does not exist. Exiting...".format(source_file))
        sys.exit()
        
    fw1 = open(undiacrized_file, "w", encoding="utf8")
    fw2 = open("/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/undiacritized_wiki2_source.txt", "w", encoding="utf8")
        
    with open(source_file, "r", encoding="utf8") as fp:
        for line in fp:
            line = line.strip()
            stop_free = " ".join([j for j in line.lower().split()])
            punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
            normalized = " ".join(word for word in punc_free.split())
            processed = re.sub(r"\d+", "", normalized)
            words = processed.split()
            
            for word in words:
                new_word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode("utf-8")
                fw1.write(new_word + ' ')
                fw2.write(word + ' ')
                
            fw1.write('\n')
            fw2.write('\n')
                
    fw1.close()
    fw2.close()
    
def compare_files(file_A, file_B):
    read_A = open(file_A, 'r', encoding="utf-8").read()
    read_B = open(file_B, 'r', encoding="utf-8").read()
    
    #words_A = read_A.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace('\n', '').replace('–', '').split(' ')
    #words_B = read_B.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace('\n', '').replace('–', '').split(' ')
    #chars_A = read_A.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace('\n', '').replace('–', '')
    #chars_B = read_B.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace('\n', '').replace('–', '')
        
    char_diff = 0
    total_char = 0
    word_diff = 0
    total_words = 0
    
    for word1, word2 in zip(read_A.split(' '), read_B.split(' ')):
        total_words += 1
        if (not word1) or (not word2):
            continue
            
        if word1 != word2:
            print(word1, word2)
            word_diff += 1
        
        for char1, char2 in zip(word1, word2):
            total_char += 1
            if char1 != char2:
                char_diff += 1
                
    good_chars = total_char - char_diff
    char_accruracy = good_chars / total_char
    print('character accuracy: ', char_accruracy * 100, "%")
            
    correct_words = total_words - word_diff
    word_accruracy = correct_words / total_words
    print('word accuracy: ', word_accruracy * 100, "%")
    

#%%
if __name__ == '__main__':
   #file_parser('/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/all_emails.csv', '/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/all_undiacritized_emails.txt')
   #remove_diacritics('/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/wiki_text2.txt', '/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/undiacritized_wiki2.txt')
   compare_files('/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/undiacritized_wiki2_source.txt', '/u00/au973065/git_repo/Semanticka_analyza_textu/diacritics_restoration/test_texts/diacritized_wiki2.txt')
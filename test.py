from collections import deque
import nltk
import numpy
import pickle
from random import shuffle
import re


WIT = False
ENG = True

WORDS = 24

SOLUTION = [0,7,16,8,18,11,12,17,2,15,19,5,20,6,14,9,10,21,22,4,23,3,1,13]


eng_vec = {}


def main():
    global eng_vec

    eng_words_orig = [
        'abdomen', 'bat', 'blanket', 'broadleaf plantain', 'cold', 'dog',
        'dog harness', 'face', 'feather', 'female dog', 'fine powder snow',
        'frog', 'icicle', 'light blue', 'male dog', 'man', 'penny', 'snow',
        'snowflake', 'snow on branches or rooftops', 'tooth', 'top',
        'upper part of stomach', 'wolf'
        ]
    wit_words_orig = [
        "bət", "bətdeχ", "cəs", "deχ", "deχyəs", "dəlkv'aχ", "dəlkv'aχbət",
        "dəlkv'aχdətay", "dəlkv'aχneɬdəc", "dəni", "dəninin", "ɬəc",
        "ɬəctl'ol", "ɬəcyəs", "ɬədəni", "ɬənani", "neɬdəc", "nin", "wəq'əz",
        "wəq'əz yəs", "wəq'əz ɣu", "yəs", "yəscəs", "ɣu"
        ]

    eng_words = [Word(w, ENG) for w in eng_words_orig]
    wit_words = [Word(w, WIT) for w in wit_words_orig]

    # print(wit_sim(wit_words[6], wit_words[7]))
    while any(w.seperate() for w in wit_words):
        continue

    # print([w.words for w in wit_words])

    # prepare_vec(extra=['bird', 'wing', 'rope'])
    load_vec()

    eng_similarity = [[0.0 for i in range(len(eng_words))] \
                      for j in range(len(eng_words))]
    for i in range(len(eng_words)):
        for j in range(len(eng_words)):
            eng_similarity[i][j] = eng_sim(eng_words[i], eng_words[j])

    wit_similarity = [[0.0 for i in range(len(wit_words))] \
                      for j in range(len(wit_words))]
    for i in range(len(wit_words)):
        for j in range(len(wit_words)):
            wit_similarity[i][j] = wit_sim(wit_words[i], wit_words[j])

    # Check the highest similarities
    # all_eng_s = [s for s_li in eng_similarity for s in s_li]
    # all_eng_s.sort(reverse=True)
    # for s in all_eng_s[24:54]:
    #     print('%.2f' % s)
    # print('---')
    # all_wit_s = [s for s_li in wit_similarity for s in s_li]
    # all_wit_s.sort(reverse=True)
    # for s in all_wit_s[24:74]:
    #     print('%.2f' % s)

    # Print sim table.
    # print('wit \\ eng')
    # for i in range(WORDS):
    #     for j in range(WORDS):
    #         if i >= j:
    #             print('%.2f' % wit_similarity[i][j], end=' ')
    #         else:
    #             print('%.2f' % eng_similarity[SOLUTION[i]][SOLUTION[j]], end=' ')
    #     print()

    # Try to use the clues to taboo but failed :(
    #
    # print(wit_similarity)
    #
    # wit_dict = {'dətay': 'bird', "tl'ol": 'rope'}
    # correlation = {}
    #
    # print(eng_sim(Word('wing', lang=ENG), Word('bat', lang=ENG)))
    #
    # print(correctness(eng_similarity, wit_similarity,
    #                   [0, 22, 8, 21, 19, 11, 13, 1, 3, 15, 16, 5,
    #                    6,23, 14, 9, 2, 7, 4, 10, 12, 17, 18, 20]))
    # try_taboo(eng_similarity, wit_similarity)

    no_clue_taboo(wit_similarity, eng_similarity)
    # print(no_clue_score(wit_similarity, eng_similarity))
    # print(no_clue_score(eng_similarity, wit_similarity,
    #                     [0, 22, 8, 21, 19, 11, 13, 1, 3, 15, 16, 5,
    #                      6,23, 14, 9, 2, 7, 4, 10, 12, 17, 18, 20]))


class W:

    def __init__(self, word='', lang=None):
        self._word = str(word)
        self._lang = ENG if lang == ENG else WIT

    def __str__(self):
        return self._word

    def __getitem__(self, i):
        return self._word[i]

    def __len__(self):
        return len(self._word)

    def __iter__(self):
        for ch in self._word:
            yield ch

    def __get__(self):
        return self._word

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def lang(self):
        return self._lang

    @property
    def is_eng(self):
        return True if self._lang == ENG else False

    @property
    def is_wit(self):
        return True if self._lang == WIT else False

    def translate(self, d) -> bool:
        if self._word in d:
            self._word = d[self._word]
            self._change_lang()
            return True
        else:
            raise KeyError
        return False


class Word:
    """
    __init__():
        word: str
        lang: str = None
        eng: bool = False
        wit: bool = False

    Class Attributes:
        eng_kw: list of english keywords
        wit_kw: list of wit keywords

    Properties:
        lang: bool or None
        is_wit: bool
        is_eng: bool
        word: str
        sep_word: tuple
        words: int

    Methods:
        translate(d, lang):
            d: the dictionary
            lang: one of the consts
        seperate(): seperates the word
    """

    kw_list = {WIT: [], ENG: []}

    @classmethod
    def update_kw(cls, w_str, lang):
        if len(w_str) < 2 or w_str in cls.kw_list[lang]:
            return False
        cls.kw_list[lang].append(w_str)
        cls.kw_list[lang].sort(key=lambda x:-len(x))

    def __init__(self, word, lang=None):
        split_word = [w for w in word.split() \
                      if w not in nltk.corpus.stopwords.words('English')]
        self._word_li = [W(w, lang=lang) for w in split_word]
        for w in split_word:
            self.__class__.update_kw(w, lang=lang)
        if lang == WIT:
            self.seperate()

    # def __get__(self):
    #     return ''.join(self._word_li)

    def __str__(self):
        ret_str = ''
        for w in self._word_li:
            ret_str += str(w)
        return ret_str

    def __iter__(self):
        for w in self._word_li:
            yield w

    def __getitem__(self, i):
        return self._word_li[i]

    def __len__(self):
        return len(self._word_li)

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def lang(self):
        if all(w.lang == WIT for w in self._word_li):
            return WIT
        if all(w.lang == ENG for w in self._word_li):
            return ENG
        return None

    @property
    def is_eng(self):
        return True if self.lang == ENG else False

    @property
    def is_wit(self):
        return True if self.lang == WIT else False

    @property
    def sep_word(self):
        return tuple(self._word_li)

    @property
    def words(self):
        return len(self._word_li)

    def translate(self, d, lang):
        for w in self._word_li:
            if w.lang != lang:
                try:
                    w.translate(d)
                except KeyError:
                    ...

    def seperate(self):
        for i in range(len(self._word_li)):
            w = self._word_li[i]
            kw_list = self.__class__.kw_list[w.lang]
            for kw in kw_list:
                if kw in str(w) and kw != str(w) and len(w) - len(kw) >= 2:
                    result = re.split("("+kw+")", str(w))
                    while '' in result:
                        result.remove('')
                    for new_kw in result:
                        self.__class__.update_kw(new_kw, lang=w.lang)
                    result_w = [W(new_kw, lang=w.lang) for new_kw in result]
                    self._word_li[i:i+1] = result_w
                    return True


def prepare_vec(extra=None):
    global eng_vec
    to_find = list(Word.kw_list[ENG])

    if extra:
        to_find += extra

    with open("C:\\Users\\Kiyume\\hackathon\\crawl-300d-2M.vec",
              encoding='utf-8') as fp:
        row1 = fp.readline()
        # if the first row is not header
        if not re.match('^[0-9]+ [0-9]+$', row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in enumerate(fp):
            cols = line.rstrip().split(' ')
            word = cols[0]

            # skip word not in words if words are provided
            if any(word == kw for kw in to_find):
                eng_vec[word] = [float(v) for v in cols[1:]]
                print(word, end=' ')
                while word in to_find:
                    to_find.remove(word)

            if len(to_find) == 0:
                print()
                break

    with open('vector.p', 'wb') as pf:
        pickle.dump(eng_vec, pf)

    return True


def load_vec():
    global eng_vec
    with open('vector.p', 'rb') as pf:
        eng_vec = pickle.load(pf)
    return eng_vec


def cos_sim(a, b):
    global eng_vec
    vec_a = eng_vec[str(a)]
    vec_b = eng_vec[str(b)]
    return numpy.dot(vec_a, vec_b) / (numpy.linalg.norm(vec_a) \
           * numpy.linalg.norm(vec_b))


def eng_sim(word1, word2):
    if word1 == word2:
        return 1
    sim = []
    for w1 in word1:
        for w2 in word2:
            sim.append(cos_sim(w1, w2))

    sim.sort()
    if len(sim) == 1:
        return sim[-1]
    else:
        if(sim[-1] > 0.85):
            return (0.7*sim[-1]+0.3*sim[-2])
        else:
            return (sim[-1]+sim[-2]) / 2


def wit_sim(word1, word2):
    if word1 == word2:
        return 1
    # print(len(word1), word1)
    sim = 0
    for w1 in word1:
        for w2 in word2:
            if w1 == w2:
                if len(word1) == 1 or len(word2) == 1:
                    sim += 0.4
                else:
                    sim += 0.2
            else:
                if nltk.edit_distance(w1, w2) == 1:
                    sim += 0.1
    return sim


def correctness(li_1, li_2, sol):
    cor = 0
    for i in range(len(li_1)):
        for j in range(len(li_1)):
            cor += abs(li_1[i][j] - li_2[sol[i]][sol[j]])
    return cor


def find_best(sim1, sim2, eng_order, taboo):
    to_beat = correctness(sim1, sim2, eng_order)
    cur_best = tuple(eng_order)
    found = False

    for i in range(len(eng_order)):
        for j in range(i+1, len(eng_order)):
            new_order = list(eng_order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_tuple = tuple(new_order)
            if hash(new_tuple) in taboo:
                continue
            new_cor = correctness(sim1, sim2, new_order)
            if new_cor < to_beat:
                to_beat = new_cor
                cur_best = new_order
                found = True

    if found:
        return cur_best
    else:
        return False


def try_taboo(sim1, sim2):
    eng_order = list(range(len(sim1)))
    taboo = deque(maxlen=30)
    taboo.append(hash(tuple(eng_order)))
    cur_cor = correctness(sim1, sim2, eng_order)

    new_best = find_best(sim1, sim2, eng_order, taboo)
    counter = 0
    while new_best:
        taboo.append(hash(tuple(new_best)))
        new_best = find_best(sim1, sim2, eng_order, taboo)
        counter += 1
        if counter % 100 == 0:
            print(new_best)


def no_clue_score(sim_2d_wit, sim_2d_eng, sol):
    """
    wit: >= 0.5 is high
    eng: >= 0.6 is high
    """
    score = 0
    for i in range(WORDS):
        for j in range(i, WORDS):
            if i == j:
                continue
            sim_a = sim_2d_wit[i][j]
            sim_b = sim_2d_eng[sol[i]][sol[j]]
            if sim_b >= 0.8:
                score += 100
                print(sol[i], sol[j])
            # elif sim_b >= 0.6:
            #     score += 10
            elif sim_a >= 0.4 and sim_b >= 0.2:
                score += 3
            # elif sim_a >= 0.4:
            #     score += 3
            # elif sim_a >= 0.2 and sim_b >= 0.2:
            #     score += 1
            # elif sim_a >= 0.2 and sim_b >= 0.1:
            #     score += 0.5
            # elif sim_a >= 0.3 and sim_b >= 0.2:
            #     score += 1
    return score


def no_clue_taboo(sim1, sim2):
    next_stack = []
    starting = list(range(len(sim1)))
    shuffle(starting)
    # starting = list(SOLUTION)
    taboo = deque(maxlen=100)
    taboo.append(hash(tuple(starting)))
    next_stack.append(starting)

    best_sol = tuple(starting)
    best_score = no_clue_score(sim1, sim2, starting)

    counter = 0

    while len(next_stack):
        cur_step = next_stack.pop()
        best_next = list(cur_step)
        best_next_score = -1
        for i in range(WORDS):
            for j in range(i+1, WORDS):
                for k in range(j+1, WORDS):
                    all_pos = [(i, j, k), (i, k, j), (j, i, k), (j, k, i),
                               (k, i, j), (k, j, i)]
                    for x, y, z in all_pos:
                        new_try = list(cur_step)
                        new_try[i], new_try[j], new_try[k] = \
                            new_try[x], new_try[y], new_try[z]
                        new_tuple = tuple(new_try)
                        if hash(new_tuple) in taboo:
                            continue
                        new_cor = no_clue_score(sim1, sim2, new_try)
                        if new_cor > best_next_score:
                            best_next_score = new_cor
                            best_next = new_try
        next_stack.append(best_next)
        taboo.append(hash(tuple(best_next)))

        if best_next_score > best_score:
            best_sol = tuple(best_next)
            best_score = best_next_score

        if counter % 10 == 0:
            print(counter)
            print(best_score, best_next_score)
            print('best_sol :', end=' ')
            for s in best_sol:
                print('%2i' % s, end=' ')
            print()
            sol_diff = 0
            for i in range(WORDS):
                sol_diff += 1 if SOLUTION[i] != best_sol[i] else 0
            print(sol_diff)
            print('best_next:', end=' ')
            for s in best_next:
                print('%2i' % s, end=' ')
            print()
            sol_diff = 0
            for i in range(WORDS):
                sol_diff += 1 if SOLUTION[i] != best_next[i] else 0
            print(sol_diff)
            print('SOLUTION :', end=' ')
            for s in SOLUTION:
                print('%2i' % s, end=' ')
            print()

        counter += 1


main()

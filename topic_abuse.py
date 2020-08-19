# this is our approach to detect aspect-oriented sentiment with topic abuse
# indicators include duplicates across contents and reviewers, and burstiness
# step-1: get one item from JD.com or TMALL.com and its nominated topics
# step-2: filter short reviews
# step-3: recognize explicit topics and its sentiments
# step-4: recognize implicit topics and its sentiments
# step-5: evaluate correlations
# step-6: evluate duplicates and burstiness
import datetime
import re
import numpy as np
import jieba.posseg as jp
import pandas as pd

MIN_LEN = 5
MAX_ASPECT_LEN = 5


# positive = ['非常好', '太棒了']
# neutral = ['一般般', '凑合', '还行']
# negative = ['差', '烂', '没用']

def cut_words(texts):
    # 切词时保留sentiment相关词汇
    # jieba.load_userdict(r'sentiment_strength.txt')
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd', 'a', 'z')  # 词性
    stopwords = ('\n', '，', ',', '没', '就', '知道', '是', '才', '也', '拿到',
                 '听听', '坦言', '越来越', '评价', '放弃', '人', '好好', '收到', '还')  # 停词

    # with open('stop_words.txt') as fs:
    #     for line in fs:
    #         stopwords.append(line.strip())

    # 分词
    words = [w for w in jp.cut(texts) if w.flag in flags and w.word not in stopwords]
    # words 含flag 和 word

    chinese_words = []
    for w in words:
        if u'\u4e00' <= w.word <= u'\u9fff':
            chinese_words.append(w)

    # return words
    return chinese_words


# to dig any aspects associated with sentiment
# and then filter any related aspects
# for now, extra aspects are not included
def find_aspects_sentiments(review, s_labels):
    words = cut_words(review)

    terms = []
    term_flags = []
    for w in words:
        terms.append(w.word)
        term_flags.append(w.flag)

    aspects = dict()
    flags = ('n', 'nr', 'ns', 'nt')

    aspect_indices = []
    hit_labels = []
    for index, term in enumerate(terms):
        if term_flags[index] in flags:
            for label, alternatives in s_labels.items():
                if term in alternatives:
                    hit_labels.append(label)
                    aspect_indices.append(index)

    # print('aspect_indices', aspect_indices)
    sentiments = dict()
    for index1, aspect1 in enumerate(aspect_indices):
        for index2, aspect2 in enumerate(aspect_indices):
            if index2 - index1 == 1 and aspect2 - aspect1 - len(terms[aspect1]) > 2:
                sentiments[hit_labels[index1]] = terms[aspect1 + 1:aspect2]

    return sentiments


def partial_aspect_case(content):
    pass


def load_corpus(reviews, labels, similar_labels):
    pattern2 = '；确认收货后[\d]+天追加'
    pattern1 = '此用户没有填写评价。'

    corpus = []
    for review in reviews:
        ass = dict()
        if len(review) < MIN_LEN:
            ass[labels[0]] = 0
            corpus.append(ass)
            continue
        else:
            # to check if suggested labels are applied
            _, content = re.split(r'%%%', review)
            content = re.sub(pattern1, '', content)
            content = re.sub(pattern2, '', content)
            if '：' in content and '；' in content:
                terms = re.split(r'；', content)
                ass = dict()
                for aspect_sentiment in terms:
                    if '：' in aspect_sentiment and len(aspect_sentiment) > 2:
                        # print('aspect_sentiment', aspect_sentiment)
                        a_s = re.split(r'：', aspect_sentiment)
                        aspect = a_s[0]
                        if aspect in labels:
                            sentiment = cut_words(a_s[1])
                            senti_words = []
                            for pair in sentiment:
                                senti_words.append(pair.word)
                            ass[aspect] = senti_words
            # to find any words which could be seen as an alternative expression of any individual aspect
            else:
                # to cut the review into words and find aspect alternatives
                ass = find_aspects_sentiments(content, similar_labels)
            if len(ass) == 0:
                ass[labels[0]] = 0
            corpus.append(ass)
    return corpus


# to quantify correlations
def inspect_sentiment(corpus, product_meta, labels, similar_labels, log_path):
    logfile = open(log_path, 'w')
    c = []
    meta_words = []
    for pm in product_meta:
        meta_words.append(pm)

    for review in corpus:
        # if review[labels[0]] == 0:
        #     continue
        ci = 0
        ct = 0
        logfile.write('---------------Here comes a new review-------------------')
        logfile.write(str(review))
        for aspect, sentiment in review.items():
            logfile.write('Review aspect: {0} - {1}'.format(aspect, sentiment))
            if ci + ct == 1:
                break
            if sentiment == 0:
                logfile.write('0; ')
                continue
            # collect all characters from aspects and sentiment
            aspect_words = []
            for a in aspect:
                aspect_words.append(a)

            sentiment_words = []
            for s in sentiment:
                for a in s:
                    sentiment_words.append(a)

            similar_words = []
            for s in similar_labels[aspect]:
                similar_words.append(s)

            # if set(sentiment).intersection(set(product_meta)):
            if set(sentiment_words).intersection(set(meta_words)):
                ci = 1
                logfile.write('ci=1; ')
            # if set(sentiment).intersection(set(aspect)) or set(sentiment).intersection(set(similar_labels[aspect])):
            if set(sentiment_words).intersection(set(aspect_words)) or set(sentiment_words).intersection(
                    set(similar_words)):
                ct = 1
                logfile.write('ct=1; ')
        c.append(ci + ct)
        logfile.write('\n')

    return c


def check_aspect(aspect):
    punctuations = ['，', '。', '！', ',', '.', '!']
    forbidden_words = ['第', '步', '评论']
    if len(aspect) > 6:
        return False
    for word in forbidden_words:
        if word in aspect:
            return False
    for punctuation in punctuations:
        if punctuation in aspect:
            return False
    return True


def find_direct_aspects(reviews):
    aspects = list()

    for line in reviews:
        _, content = re.split(r'%%%', line.strip())
        if '：' in content and '；' in content:
            ass = re.split(r'；', content)
            for a_s in ass:
                if '：' in a_s:
                    aspect, _ = re.split(r'：', a_s)
                    if check_aspect(aspect):
                        aspects.append(aspect)
    return aspects

def time_error(times):
    time_differences = []
    means = []
    error_var = 0
    for index_1, time_1 in enumerate(times):
        time_difference = []
        for index_2, time_2 in enumerate(times):
            if index_1 == index_2:
                continue
            # compute the time difference
            # first translate 2020年04月07日 18:53 to 2020-04-07 18:53:00
            year = time_1[:4]
            month = time_1[5:7]
            day = time_1[8:10]
            dt_1 = datetime.datetime(int(year), int(month), int(day))

            year = time_2[:4]
            month = time_2[5:7]
            day = time_2[8:10]
            dt_2 = datetime.datetime(int(year), int(month), int(day))

            time_diff = abs((dt_1 - dt_2).days)
            time_difference.append(time_diff)

        mean_time_diff = np.mean(time_difference)
        means.append(mean_time_diff)
        for diff in time_difference:
            error_var += (diff - mean_time_diff) * (diff - mean_time_diff)

        time_differences.append(error_var)

    # return time_differences
    return means

if __name__ == "__main__":
    # build corpus
    # filename = 'review_sample_clean.txt'
    # base_path = r'E://reviewData//Kingston//'
    base_path = r'E://reviewData//'
    # product_path = r'11_562719599738_Kingston-金士顿 SA400S37-240G 台式机笔记本电脑SSD固态硬盘//'
    # product_path = r'12_573854225689_金士顿480g固态硬盘sata3接口2.5寸固体机械台式机电脑手提笔记本加装预装带//'
    product_path = r'10_545020495706_金士顿240G固态 sata3固态硬盘非256G 笔记本 2.5寸台式机电脑120G 480G SSD//'
    # product_path = r'13_4311178_金士顿 Kingston 240GB SSD固态硬盘 SATA3.0接口 A400系列//'
    label_path = base_path + product_path + r'labels.txt'
    label_similar_path = base_path + product_path + r'labels_alternatives.txt'
    aspects_sentiments_path = base_path + product_path + r'aspects_sentiments.txt'
    log_path = base_path + product_path + r'log.txt'
    meta_file_path = base_path + product_path + r'review_meta.txt'
    pure_review_path = base_path + product_path + r'purereviewcontent.txt'

    correlative_path = base_path + product_path + r'correlative.txt'
    performance_path = base_path + product_path + r'performance.txt'
    aspects = set()

    product_meta = cut_words(product_path)

    labels = []
    with open(label_path) as label_file:
        for line in label_file:
            labels.append(line.strip())

    similar_labels = dict()
    with open(label_similar_path) as similar_file:
        for line in similar_file:
            label, alternatives = re.split(r'：', line.strip())
            words = re.split(r'，', alternatives)
            similar_labels[label] = words

    # print('similar_labels:', similar_labels)

    assdf = pd.DataFrame()

    stopreview = ['此用户未填写评价内容']
    review_metas = []
    with open(meta_file_path) as meta_file:
        for line in meta_file:
            review_metas.append(line.strip())

    pure_reviews = []
    with open(pure_review_path) as pure_review_file:
        for line in pure_review_file:
            pure_reviews.append(line.strip())

    # # to automatically find aspects
    # labels = find_direct_aspects(pure_reviews)
    # print('len of labels ', len(labels))
    # aspects = aspects.union(set(labels))

    # corpus is a list of dicts, in which multi-aspect-oriented sentiments are collected
    corpus = load_corpus(pure_reviews, labels, similar_labels)
    # print('corpus:', corpus)
    # to quantify correlations
    c = inspect_sentiment(corpus, product_meta, labels, similar_labels, log_path)

    print('length of c: ', len(c))
    total_reviews = len(pure_reviews)
    print('count of 0: {0}, {1}'.format(c.count(0), c.count(0) / len(c)))
    print('count of 1: {0}, {1}'.format(c.count(1), c.count(1) / len(c)))
    print('c: ', c)

    corr_review_count = c.count(1)
    non_review_count = c.count(0)

    # assdf = assdf.append(pd.DataFrame(data=aspects_sentiments))

    correlated = []
    noncorrelated = []
    # to evaluate
    for index, score in enumerate(c):
        if score == 1:
            correlated.append(index)
        else:
            noncorrelated.append(index)

    # with open(correlative_path, 'w') as cf:
    #     cf.write('------------correlated reviews--------------\n\n')
    #     for co in correlated:
    #         cf.write(review_metas[co])
    #         cf.write(': ')
    #         cf.write(pure_reviews[co])
    #         cf.write('\n')
    #     cf.write('\n------------noncorrelated reviews--------------\n\n')
    #     for nc in noncorrelated:
    #         cf.write(review_metas[nc])
    #         cf.write(': ')
    #         cf.write(pure_reviews[nc])
    #         cf.write('\n')

    # evaluate duplicate and burstiness within these two subgroups
    reviewers_corr = []
    reviewers_non = []
    review_time_corr = []
    review_time_non = []
    reviews_corr = []
    reviews_non = []
    for index in correlated:
        meta = review_metas[index]
        reviewer = meta[-7:]
        _, rest = re.split(r'%%%', meta)
        review_time = rest[:12]
        reviewers_corr.append(reviewer)
        review_time_corr.append(review_time)
        no, review = re.split(r'%%%', pure_reviews[index])
        reviews_corr.append(review)

    for index in noncorrelated:
        meta = review_metas[index]
        reviewer = meta[-7:]
        _, rest = re.split(r'%%%', meta)
        review_time = rest[:12]
        reviewers_non.append(reviewer)
        review_time_non.append(review_time)
        no, review = re.split(r'%%%', pure_reviews[index])
        reviews_non.append(review)

    # comparison with pure duplicate method
    # to check duplication
    corr_time_differences = time_error(review_time_corr)
    non_time_differences = time_error(review_time_non)
    mean_corr_time_diff = np.mean(corr_time_differences)
    mean_non_time_diff = np.mean(non_time_differences)

    uni_reviewer_count_corr = len(set(reviewers_corr))
    uni_reviewer_corr_ratio = 1 - uni_reviewer_count_corr / corr_review_count
    uni_reviewer_count_non = len(set(reviewers_non))
    uni_reviewer_non_ratio = 1 - uni_reviewer_count_non / non_review_count

    uni_review_count_corr = len(set(reviews_corr))
    uni_review_corr_ratio = 1 - uni_review_count_corr / corr_review_count
    uni_review_count_non = len(set(reviews_non))
    uni_review_non_ratio = 1 - uni_review_count_non / non_review_count

    with open(performance_path, 'w') as pf:
        pf.write('----------performance of correlative reviews-------------\n\n')
        pf.write('total count: {0}\n'.format(corr_review_count))
        pf.write('duplicated reviewer count and ratio {0}, {1}\n'.
                 format(uni_reviewer_count_corr, uni_reviewer_corr_ratio))
        pf.write('duplicated review count and ratio {0}, {1}\n'.format(uni_review_count_corr, uni_review_corr_ratio))
        pf.write('time error {0}\n'.format(mean_corr_time_diff))

        pf.write('\n----------performance of noncorrelative reviews-------------\n\n')
        pf.write('total count: {0}\n'.format(non_review_count))
        pf.write('duplicated reviewer count and ratio {0}, {1}\n'.
                 format(uni_reviewer_count_non, uni_reviewer_non_ratio))
        pf.write('duplicated review count and ratio {0}, {1}\n'.format(uni_review_count_non, uni_review_non_ratio))
        pf.write('time error {0}\n'.format(mean_non_time_diff))


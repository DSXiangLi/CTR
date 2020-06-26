import random
import pickle

random.seed(1234)

with open('data/amazon/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  def gen_neg():
    neg = pos_list[0]
    # generate item that user doesn't
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))] # generate same length negative item sample: balance 1:1

  # 1 reviewer has N positive item: generate N negative item
  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    if i != (len(pos_list)-1):
      train_set.append((reviewerID, hist, pos_list[i], 1)) # use N-1 for train and 1 for test
      train_set.append((reviewerID, hist, neg_list[i], 0)) # generate N-1 negative sample for train
    else:
      test_set.append((reviewerID, hist, pos_list[i], 1))
      test_set.append((reviewerID, hist, neg_list[i], 0))

random.shuffle(train_set)
random.shuffle(test_set)

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL )




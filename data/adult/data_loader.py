from six.moves import urllib
import re

def data_loader():

    DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.{}'
    DATA_DIR = './data/adult/{}.csv'
    files = {'data':'train',
            'test':'valid'}

    for ifile in files.keys():
        url = DATA_URL.format(ifile)

        f ,_ = urllib.request.urlretrieve(url)

        data_dir = DATA_DIR.format(files[ifile])
        with open(data_dir, 'w') as f_writer:
            with open(f, 'r', encoding='UTF-8') as f_reader:
                for line in f_reader:
                    line = line.strip()
                    line = re.sub(r'\s+',"", line)

                    if not line:
                        continue

                    if line[-1] == '.':
                        line = line[:-1]

                    line +='\n'
                    f_writer.write(line)


if __name__ == '__main__' :
    data_loader()



df = pd.readc
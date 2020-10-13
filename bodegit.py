#  Module bodegit.py
#
#  Copyright (c) 2020 Mehdi Golzadeh <golzadeh.mehdi@gmail.com>.
#
#  Licensed under GNU Lesser General Public License version 3.0 (LGPL3);
#  you may not use this file except in compliance with the License.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""
Descriptions
"""


# --- Prerequisites ---
from multiprocessing import Pool
import pandas
import pickle
import threading
import git
from Levenshtein import distance as lev
import itertools
from sklearn.cluster import DBSCAN
import json
import sys
import dateutil
import pkg_resources
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request
import argparse
from tqdm import tqdm


# --- Exception ---
class BodegitError(ValueError):
    pass


# --- Extract commit messages ---
def process_comments(repositories, authors, date, min_commits, max_commits):
    comments = pandas.DataFrame()

    for repository in repositories:
        try:
            repo = git.Repo(repository)
        except Exception as e:
            print('Unable to access repository {} ({}:{})'.format(repository, e.__class__.__name__, e))
            sys.exit()

        commits = repo.iter_commits('--all')
        for commit in commits:
            comments = comments.append({
                        'author': commit.author.name,
                        'email': commit.author.email,
                        'body': commit.message[:-1] if commit.message.endswith('\n') else commit.message,
                        'repository': repository,
                        'created_at': datetime.date.fromtimestamp(commit.authored_date),
                        'type': 'gitmsg',
                    }, ignore_index=True,sort=True)
        
    comments = comments.assign(empty = lambda x: np.where(x['body'].str.len()<5,1,0))

    return comments

# --- Text process and feature production ---
def tokenizer(text):
    return text.split(' ')


def compute_distance(items, distance):
    """
    Computes a distance matrix for given items, using given distance function.
    """
    m = np.zeros((len(items), len(items)))
    enumitems = list(enumerate(items))
    for xe, ye in itertools.combinations(enumitems, 2):
        i, x = xe
        j, y = ye
        d = distance(x, y)
        m[i, j] = m[j, i] = d
    return m


def jaccard(x, y):
    """
    To tokenize text and compute jaccard disatnce
    """
    x_w = set(tokenizer(x))
    y_w = set(tokenizer(y))
    return (
        len(x_w.symmetric_difference(y_w)) / (len(x_w.union(y_w)) if len(x_w.union(y_w)) > 0 else 1)
    )


def levenshtein(x, y, n=None):
    if n is not None:
        x = x[:n]
        y = y[:n]
    return lev(x, y) / (max(len(x), len(y)) if max(len(x), len(y)) > 0 else 1)


def average_jac_lev(x, y):
    """
    Computes average of jacard and levenshtein for 2 given strings
    """
    return (jaccard(x, y) + levenshtein(x, y)) / 2


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    if len(array) == 0:
        return 0
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return round(((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))),3)


def count_empty_comments(comments):
    empty_comments = 0
    for comment in comments:
        if comment == "":
            empty_comments += 1
    return empty_comments


# --- Load model and prediction ---
def get_model():
    path = 'model.json'
    filename = pkg_resources.resource_filename(__name__, path)
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def predict(model, df):
    df = (
        df
        .assign(
            prediction=lambda x: np.where(model.predict(
                x[['comments', 'empty comments', 'patterns', 'dispersion']]) == 1, 'Bot', 'Human')
        )
    )
    return df


# --- Thread and progress ---
def task(data):
    author,email , group, max_commits, params = data
    group = group[:max_commits]
    clustering = DBSCAN(eps=params['eps'], min_samples=1, metric='precomputed')
    items = compute_distance(getattr(group, params['source']), params['func'])
    clusters = clustering.fit_predict(items)
    empty_comments = np.count_nonzero(group['empty'])

    return (
        author,
        email,
        len(group),
        empty_comments,
        len(np.unique(clusters)),
        gini(items[np.tril(items).astype(bool)]),
    )


def run_function_in_thread(pbar, function, max_value, args=[], kwargs={}):
    ret = [None]

    def myrunner(function, ret, *args, **kwargs):
        ret[0] = function(*args, **kwargs)

    thread = threading.Thread(target=myrunner, args=(function, ret) + tuple(args), kwargs=kwargs)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=.1)
        if(pbar.n < max_value - .3):
            pbar.update(.1)
    pbar.n = max_value
    return ret[0]


def progress(repositories, authors, authors_email, exclude, exclude_emails, date, verbose, min_commits, max_commits, output_type):
    download_progress = tqdm(
        total=25, desc='Extracting commits', smoothing=.1,
        bar_format='{desc}: {percentage:3.0f}%|{bar}', leave=False)
    comments = run_function_in_thread(
        download_progress, process_comments, 25,
        args=[repositories, authors, date, min_commits, max_commits])
    download_progress.close()

    if comments is None:
        raise BodegitError('Download failed please check your apikey or required libraries.')

    if len(comments) < 1:
        raise BodegitError('Available comments are not enough to predict the type of authors')

    df = (
        comments
        [comments['author'].isin(
            comments
            .groupby('author', as_index=False)
            .count()[lambda x: x['body'] >= min_commits]['author'].values
        )]
        .sort_values('created_at', ascending=False)
        .groupby('author').head(100)
    )
    if exclude != []:
        df = df[~df["author"].isin(exclude)]

    if authors != []:
        df = df[lambda x: x['author'].isin(authors)]

    if(len(df) < 1):
        raise BodegitError('There are not enough comments in the selected time period to\
predict the type of authors. At least 10 comments is required for each author.')

    inputs = []
    for author, group in df.groupby(['author','email']):
        inputs.append(
            (
                author[0],
                author[1],
                group.copy(),
                max_commits,
                {'func': average_jac_lev, 'source': 'body', 'eps': 0.5}
            )
        )

    data = []
    with Pool() as pool:
        for result in tqdm(
                pool.imap_unordered(task, inputs),
                desc='Computing features',
                total=len(inputs),
                smoothing=.1,
                bar_format='{desc}: {percentage:3.0f}%|{bar}',
                leave=False):
            data.append(result)

    
    df_clusters = pandas.DataFrame(
        data=data, columns=['author','email', 'comments', 'empty comments', 'patterns', 'dispersion'])

    prediction_progress = tqdm(
        total=25, smoothing=.1, bar_format='{desc}: {percentage:3.0f}%|{bar}', leave=False)
    tasks = ['Loading model', 'Making prediction', 'Exporting result']
    prediction_progress.set_description(tasks[0])
    model = run_function_in_thread(prediction_progress, get_model, 5)
    if model is None:
        raise BodegitError('Could not load the model file')

    prediction_progress.set_description(tasks[1])
    result = run_function_in_thread(
        prediction_progress, predict, 25, args=(model, df_clusters))
    
    result = result.sort_values(['prediction', 'author'])
    prediction_progress.close()

    result = result.append(  
        (
            comments[lambda x: ~x['author'].isin(result.author)][['author','body']]
            .groupby('author', as_index=False)
            .count()
            .assign(
                empty=np.nan,
                patterns=np.nan,
                dispersion=np.nan,
                prediction="Few data",
            )
            .rename(columns={'body':'messages','empty':'empty messages'})
        ),ignore_index=True,sort=True)
    
    for author in (set(authors) - set(result.author)):
        result = result.append({
            'author': author,
            'comments':np.nan,
            'empty comments':np.nan,
            'patterns':np.nan,
            'dispersion':np.nan,
            'prediction':"Not Found",
        },ignore_index=True,sort=True)
    if verbose is False:
        result = result.set_index('author')[['email','prediction']]
    else:
        result = result.set_index('author')[['email','comments', 'empty comments', 'patterns', 'dispersion','prediction']]

    if output_type == 'json':
        return (result.reset_index().to_json(orient='records'))
    elif output_type == 'csv':
        return (result.to_csv())
    else:
        return (result)


# --- cli ---
def arg_parser():
    parser = argparse.ArgumentParser(description='BoDeGit - Bot detection in Git commit messages')
    parser.add_argument('--repositories', metavar='REPOSITORY',
        help='list of a repositories on GitHub in the form of ("owner/repo")',
        default=list(), type=str, nargs='*')
    parser.add_argument(
        '--authors', metavar='AUTHOR', required=False, default=list(), type=str, nargs='*',
        help='List of authors. Example: \
--authors "mehdi golzadeh" "alexandre decan" "tom mens"')
    parser.add_argument(
        '--authors-email', metavar='AUTHOR-EMAIL', required=False, default=list(), type=str, nargs='*',
        help='List of authors\' email. Example: \
--authors-email mehdigolzadeh@umons.ac.be alexandredecan@umons.ac.be tommens@umons.ac.be')
    parser.add_argument(
        '--exclude', metavar='AUTHOR', required=False, default=list(), type=str, nargs='*',
        help='List of authors to be excluded in the analysis. Example: \
--exclude "mehdi golzadeh" "alexandre decan" "tom mens"')
    parser.add_argument(
        '--exclude-emails', metavar='AUTHOR-EMAIL', required=False, default=list(), type=str, nargs='*',
        help='List of authors email to be excluded in the analysis. Example: \
--exclude-emails mehdigolzadeh@umons.ac.be alexandredecan@umons.ac.be tommens@umons.ac.be')
    parser.add_argument(
        '--start-date', type=str, required=False,
        default=None, help='Commits later than this date will be considered')
    parser.add_argument(
        '--verbose', action="store_true", required=False, default=False,
        help='To have verbose output result')
    parser.add_argument(
        '--min-commits', type=int, required=False, default=10,
        help='Minimum number of commits to analyze an author (default=10)')
    parser.add_argument(
        '--max-commits', type=int, required=False, default=100,
        help='Maximum number of commits to be used (default=100)')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--text', action='store_true', help='Print results as text.')
    group2.add_argument('--csv', action='store_true', help='Print results as csv.')
    group2.add_argument('--json', action='store_true', help='Print results as json.')

    return parser.parse_args()


def cli():
    args = arg_parser()

    date = datetime.datetime.now()+relativedelta(months=-6)
    if args.start_date is not None:
        date = dateutil.parser.parse(args.start_date)

    if args.min_commits < 10:
        sys.exit('Minimum number of required commits for the model is 10.')
    else:
        min_commits = args.min_commits

    if args.max_commits < 10:
        sys.exit('Maximum number of commits cannot be less than 10.')
    else:
        max_commits = args.max_commits

    if args.csv:
        output_type = 'csv'
    elif args.json:
        output_type = 'json'
    else:
        output_type = 'text'

    try:
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
            print(
                progress(
                    args.repositories,
                    args.authors,
                    args.authors_email,
                    args.exclude,
                    args.exclude_emails,
                    date,
                    args.verbose,
                    min_commits,
                    max_commits,
                    output_type
                ))
    except BodegitError as e:
        sys.exit(e)


if __name__ == '__main__':
    cli()
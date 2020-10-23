# BoDeGit
An automated tool to identify bots in Git repositories by analysing commit information.
The tool has been developed by Mehdi Golzadeh, researcher at the Software Engineering Lab of the University of Mons (Belgium) as part of his PhD research.

This tool accepts the name of a list of Git repositories and computes its output in three steps.
The first step consists of extracting all commit information from the specified Git repositories using git log. This step results in a list of authors and their corresponding commits.
The second step consists of computing the number of messages, empty messages, message patterns, and inequality between the number of messages within patterns.
The third step simply applies the model we developed on these examples and outputs the prediction made by the model


More details about the classification model can be found in a companion research article that is currently under peer review.

**Important note!** When running the tool on a Git repository of your choice, it is possible, for some human authors or bot authors to be misclassified by the classification model. If you would encounter such situations while running the tool, please inform us about it, so that we can strive to further improve the accurracy of the classification algorithm.


## Installation
To install BoDeGit, run the following command:
```
pip install git+https://github.com/mehdigolzadeh/BoDeGit
```
Given that this tool has many dependencies, and in order not to conflict with already installed packages, it is recommended to use a virtual environment before its installation. You can install and create a _Python virtual environment_ and then install and run the tool in this environment. You can use any virtual environment of your choice. Below are the steps to install and create a virtual environment with **virtualenv**.

Use the following command to install the virtual environment:
```
pip install virtualenv
```
Create a virtual environment in the folder where you want to place your files:
```
virtualenv <name>
```
Start using the environmnet by:
```
source <name>/bin/activate
```
After running this command your command line prompt will change to `(<name>) ...` and now you can install BoDeGit with the pip command.
When you are finished running the tool, you can quit the environment by:
```
deactivate
```


## Usage 

Here is the list of parameters:

`--repositories [REPOSITORY [REPOSITORY ...]]` 	**A list of repositories' path**
> Example: $ bodegit ./path/to/repo1 ./path/to/repo2

_At least one repository is required_

`--include [NAME [Name ...]]` 	**A list of name of authors**
> Example: $ bodegit ./path/to/repo1 --authors "jim golzadeh" "donald s" "m mens"

_By default all authors in the repository will be analysed_

`--committer` 	**To analyse committers instead of authors**
> Example: $ bodegit ./path/to/repo1 --committer
  
`--mapping [MAPPING]` 	**Mapping file to merge identities. This file must be a csv file where each line contains two values: the name to be merged, and the corresponding identity.**
> Example: $ bodegit ./path/to/repo1 --mapping ./mapping.csv

_Use "IGNORE" as identity to ignore specific names._

`--start-date START_DATE` 		**Start date of commit in the repository to be considered**
> Example: $ bodegit ./path/to/repo1 --start-date 01-01-2018
  
_The default start-date is 6 months before the current date._

`--verbose` **To have verbose output result**
> Example: $ bodegit ./path/to/repo1 --verbose
 
_The default value is false, if you don't pass this parameter the output will only be the accounts and their type_

`--only-predicted` **Only list accounts that the prediction is available**
> Example: $ bodegit ./path/to/repo1 --only-predicted
 
_The default value is true, if this parameter is not passed, the output would include all accounts resulting in "unknown" for accounts that were unable to make a prediction_
  
`--min-commits MIN_COMMITS` 		**Minimum number of commit messages that are required to analyze an account**
> Example: $ bodegit ./path/to/repo1 --min-commits 20
 
_The default value is 10 comments_

`--max-commits MAX_COMMITS` 		**Maximum number of commit messages to be considered for each account (default=100)**
> Example: $ bodegit ./path/to/repo1 --max-commits 120

_The default value is 100 comments_

`--text`                	Output results as plain text
`--csv`                		Output results in comma-separated values (csv) format
`--json`                	Output results in json format
> Example: $ bodegit ./path/to/repo1 --json

_This group of parameters is the type of output, e.g., if you pass --json you will get the result in JSON format_



## Examples of bodegit output (for illustration purposes only)
```
$ bodegit ./path/to/repo1  --verbose --committer
                  messages  patterns  dispersion prediction
committer
Travis CI[bot]          20         1       0.026        Bot
greenkeeper[bot]        10         3       0.141        Bot
blablabla               69        58       0.040      Human
blablabla                5       NaN         NaN    Unknown
```

```
$ bodegit ./path/to/repo1 --start-date 01-01-2017  --verbose --min-commits 20 --max-commits 90 --json

[{"author":"Travis CI[bot]","messages":20,"patterns":1,"dispersion":0.026,"prediction":"Bot"},{"author":"blablabla","messages":69,"patterns":58,"dispersion":0.04,"prediction":"Human"},{"author":"greenkeeper[bot]","messages":10,"patterns":null,"dispersion":null,"prediction":"Unknown"},{"author":"blablabla","messages":5,"patterns":null,"dispersion":null,"prediction":"Unknown"}]
```

```
$ bodegit ./path/to/repo1 --verbose --max-commits 80 --csv

author,messages,patterns,dispersion,prediction
Travis CI[bot],20,1,0.026,Bot
greenkeeper[bot],10,3,0.141,Bot
blablabla,69,58,0.04,Human
blablabla,5,,,Unknown
```

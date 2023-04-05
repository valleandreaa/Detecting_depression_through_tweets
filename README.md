
# Detecting depression through tweets 

### Installation
```
pip install -r requirements.txt
```
* The optimal set size is 3'500 observations.
* The partition of the dataset is 80% - train set and 20 % test set 

| Model               | Approach    | kernel | solver    | Hyperparameters           | Accuracy | F1.5-Score | Recall | Precision | 
|---------------------|-------------|--------|-----------|---------------------------|----------|------------|-------|-----------|
| SVM                 | Bag of Words | poly   |           | gamma =1.0 , C=0.01       |          |            |       |           |
| SVM                 |  TFIDF       |   rbf     |           | gamma = 0.1, C=1          |  0.7108    |  0.6609      | 0.6266    |   0.75387     |
| Naive Bayes         |     Bag of Words         | -      |           | alpha=1.0 fit_prior=False | 0.717    | 0.722      | 0.726 | 0.713     |
| Naive Bayes         |         TFIDF | -      |           | alpha=2.0 fit_prior=False | 0.736    | 0.752      | 0.766 | 0.723     |
| Logistic Regression |     Bag of Words          | -      |   liblinear | C= 0.1, penality =l1      | 0.651    | 0.553      | 0.502 | 0.716     |
| Logistic Regression  |       TFIDF        | -      | liblinear | C= 1.0, penality =l1      | 0.700    | 0.655      | 0.625 | 0.735     |

[//]: # (an IDE for Python.)

[//]: # ()
[//]: # (Inspired by [awesome-python]&#40;https://github.com/vinta/awesome-python&#41;.)

[//]: # ()
[//]: # (## Contents)

[//]: # ()
[//]: # (- [Awesome PyCharm]&#40;#awesome-pycharm&#41;)

[//]: # (    - [Articles]&#40;#articles&#41;)

[//]: # (    - [Tutorials]&#40;#tutorials&#41;)

[//]: # (    - [Videos]&#40;#videos&#41;)

[//]: # (    - [Plugins]&#40;#plugins&#41;)

[//]: # (    - [Live Templates]&#40;#live-templates&#41;)

[//]: # (- [Contributing]&#40;#contributing&#41;)

[//]: # ()
[//]: # (- - -)

[//]: # ()
[//]: # (## Articles)

[//]: # ()
[//]: # (* [Configure PyCharm for Python/Django and Introduction to Django Rest Framework]&#40;https://medium.com/@srijan.pydev_21998/configure-pycharm-for-python-django-and-introduction-to-django-rest-framework-f9c1a7cb4ba0&#41; )

[//]: # ()
[//]: # (is a walkthrough to get the most out of PyCharm Professional while working with Django. *&#40;2018/04/07, Srijan Anand&#41;*)

[//]: # ()
[//]: # (* [A Look at PyCharm Python IDE for Linux]&#40;https://www.ghacks.net/2017/10/12/pycharm-python-ide-linux/&#41; )

[//]: # (discusses PyCharm as an IDE plus installation instructions for Linux Mint )

[//]: # (using a PPA. *&#40;2017/10/12, Mike Turcotte&#41;*)

[//]: # ()
[//]: # (* [Integrating PyCharm with Pyenv]&#40;http://vcrmartinez.com/2017/08/04/integrating-pycharm-with-pyenv/&#41;  shows how to use )

[//]: # ([pyenv]&#40;https://github.com/pyenv/pyenv&#41; &#40;the Python version management tool&#41; )

[//]: # (from within PyCharm. *&#40;2017/08/04, Viktor Martinez&#41;*)

[//]: # ()
[//]: # (* [Python Tool Review: Using PyCharm for Python Development - and More]&#40;https://www.caktusgroup.com/blog/2017/07/05/python-tool-review-using-pycharm-python-development-and-more/?utm_content=58335036&utm_medium=social&utm_source=twitter&#41; )

[//]: # (reviews PyCharm as an IDE, discussing performance, Python, Django, Git, )

[//]: # (code-checking, and more. *&#40;2017/07/05, Dan Poirier from Caktus Group&#41;*)

[//]: # ()
[//]: # (* [Best Python IDE, Complete Tutorial to setup Python With Pycharm]&#40;http://www.csestack.org/best-python-ide-complete-tutorial-to-setup-python-with-pycharm/&#41; shows )

[//]: # (complete steps under Windows to setup Python and PyCharm Community Edition, )

[//]: # (writing and running a simple program, and explains shortcuts.)

[//]: # (*&#40;2016/02/22, Aniruddha Chaudhari&#41;*)

[//]: # ()
[//]: # (## Tutorials)

[//]: # ()
[//]: # (* [MongoDB QuickStart with Python]&#40;http://freemongodbcourse.com&#41; is a free )

[//]: # (course by Michael Kennedy which features PyCharm. *&#40;2017/10/11, Michael Kennedy&#41;*)

[//]: # ()
[//]: # (## Videos)

[//]: # ()
[//]: # (### English)

[//]: # ()
[//]: # (* [Python Beginner Tutorial 1 - Install and Setup PyCharm IDE]&#40;https://youtu.be/0y5XlNeFxNk&#41; )

[//]: # (covers the installation of the latest version of Python 3 and installation )

[//]: # (and setup of the Free JetBrains PyCharm IDE. After watching this video you )

[//]: # (will know how to run your very first Python script. *&#40;2017-05-19&#41;*)

[//]: # ()
[//]: # (* [PyCharm Terminal]&#40;https://youtu.be/i1js96Ha_OQ&#41; covers usage of the )

[//]: # (embedded Terminal tool in PyCharm Community Edition, under Windows. Demos )

[//]: # (running Django's `manage.py`. *&#40;2017/08/03, Chris Mahn&#41;*)

[//]: # ()
[//]: # (* [PyCharm tips and tricks]&#40;https://youtu.be/SVxuUGjB8YU&#41; demonstrates many not so obvious, but super useful features and hotkeys *&#40;2017/07/12, Dmitry Trofimov&#41;* )

[//]: # ()
[//]: # (* [django-bootstrap3 Pycharm Module Install]&#40;https://youtu.be/5y9Z_BhEr5Q&#41;  Use PyCharm to install this plugin )

[//]: # ( into an existing Django project. *&#40;2017/08/19, Chris Mahn&#41;*)

[//]: # ()
[//]: # ()
[//]: # (* [Pycharm Reformatting]&#40;https://youtu.be/JZ_xuPiK-UA&#41; shows )

[//]: # (reformatting files, generating imports with quick fix. *&#40;2017/08/05, Chris Mahn&#41;*)

[//]: # ()
[//]: # (* [Productive pytest with PyCharm]&#40;https://youtu.be/ixqeebhUa-w&#41; helps level up pytest skill. *&#40;2018/02/26, Brian Okken&#41;*)

[//]: # ()
[//]: # (### Spanish)

[//]: # ()
[//]: # (* [Django Creación de nuestro primer proyecto con Django en PyCharm]&#40;https://youtu.be/oX0SoU9OHnE&#41; )

[//]: # (*&#40;2017/06/14, KeepCoding&#41;*)

[//]: # ()
[//]: # (### Portuguese)

[//]: # ()
[//]: # (* [Curso Python #05 - Instalando o PyCharm e o QPython3]&#40;https://youtu.be/ElRd0cbXIv4&#41; )

[//]: # (Nesta aula, veremos como instalar e configurar a IDE &#40;Integrated Development )

[//]: # (Environment&#41; Python chamada PyCharm no Windows, MacOS e )

[//]: # (Linux.  *&#40;2017/05/05, Curso em Video&#41;*)

[//]: # ()
[//]: # (## Plugins)

[//]: # ()
[//]: # (* Database and Frameworks)

[//]: # (  * [MongoDB Plugin for IntelliJ]&#40;https://plugins.jetbrains.com/plugin/7141-mongo-plugin&#41; )

[//]: # (integrates MongoDB Servers with database/collections tree, Query Runner and )

[//]: # (Shell console. *&#40;2017-12-12&#41;*)

[//]: # (  * [JS GraphQL]&#40;https://plugins.jetbrains.com/plugin/8097-js-graphql/&#41; provides GraphQL support directly inside PyCharm.)

[//]: # (* Editor)

[//]: # (  * [CodeGlance]&#40;https://plugins.jetbrains.com/plugin/7275-codeglance&#41; provides a minimap for your editor, similar to Sublime.)

[//]: # (  * [Open in Splitted Tab]&#40;https://plugins.jetbrains.com/plugin/7407-open-in-splitted-tab/&#41; adds a PyCharm command to open a definition in a new splitted tab.)

[//]: # (* Code Analysis)

[//]: # (  * [Sourcery]&#40;https://plugins.jetbrains.com/plugin/12631-sourcery&#41; provides a list of refactoring recommendations to simplify your codebase. [freemium])

[//]: # (  * [Grazie]&#40;https://plugins.jetbrains.com/plugin/12175-grazie/&#41; provides grammar and advanced spell checking.)

[//]: # (  * [Python Security]&#40;https://plugins.jetbrains.com/plugin/13609-python-security/&#41; helps you spot security problems in libraries and code.)

[//]: # (* Integration)

[//]: # (  * [Code Review for BitBucket]&#40;https://plugins.jetbrains.com/plugin/13538-code-review-for-bitbucket/&#41; lets you manage BitBucket pull requests from inside the IDE. [paid])

[//]: # (* File Type Support)

[//]: # (  * [Requirements]&#40;https://plugins.jetbrains.com/plugin/10837-requirements/&#41; adds extra support for `requirements.txt` files.)

[//]: # (  * [Idealog]&#40;https://plugins.jetbrains.com/plugin/9746-ideolog/&#41; views log files.)

[//]: # (  * [.ignore]&#40;https://plugins.jetbrains.com/plugin/7495--ignore/&#41; provides support for `.gitignore` and other ignore file lists.)

[//]: # (  * [.env]&#40;https://plugins.jetbrains.com/plugin/9525--env-files-support/&#41; support for `.env` environment variable definitions.)

[//]: # (  * [Pug]&#40;https://plugins.jetbrains.com/plugin/7094-pug-ex-jade-/&#41; template support plugin.)

[//]: # (  * [Extra Icons]&#40;https://plugins.jetbrains.com/plugin/11058-extra-icons/&#41; provides icons for a lot more file types.)

[//]: # ()
[//]: # (## Live Templates)

[//]: # ()
[//]: # (* [Flask PyCharm Templates]&#40;https://github.com/mstuttgart/flask-pycharm-templates&#41;)

[//]: # (Collection of live templates to help you develop Flask web applications. *&#40;2017-10-08, Michell Stuttgart&#41;*)

[//]: # ()
[//]: # (## Themes)

[//]: # ()
[//]: # (* [PyCharm Color Schemes]&#40;https://github.com/mstuttgart/pycharm-color-scheme&#41;)

[//]: # (Collection of themes adapted to use with this IDE. *&#40;2019-10-01, Michell Stuttgart&#41;*)

[//]: # ()
[//]: # (# Contributing)

[//]: # ()
[//]: # (Your contributions are always welcome! Please take a look at the )

[//]: # ([contribution guidelines]&#40;https://github.com/JetBrains/awesome-pycharm/blob/master/CONTRIBUTING.md&#41; )

[//]: # (first.)

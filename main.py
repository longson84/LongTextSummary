

import trafilatura
from plugin.chromeplugin.summary import funcs

if __name__ == '__main__':
    url = "https://www.slingacademy.com/article/python-ways-to-extract-plain-text-from-a-webpage/"

    text = trafilatura.fetch_url(url)



    print(trafilatura.extract(text))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

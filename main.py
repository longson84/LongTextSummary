
import trafilatura
from plugin.chromeplugin.summary import funcs

if __name__ == '__main__':

    # url = "https://www.cnbc.com/2024/01/26/tech-layoffs-jump-in-january-as-alphabet-meta-microsoft-reach-high.html"
    #
    # text = funcs.extract_text_from_url(url)
    #
    # embedding_map = funcs.embed_text(text)
    #
    # funcs.save_to_vector_db(embedding_map)

    file_path = "docs/doc1.pdf"
    funcs.summarize_pdf(file_path)

    # funcs.test_calc_rouge()

